"""Club matches of many competitions from ESPN's public scoreboard (the v1C club Elo).

The quiniela tickets carry club matches of other competitions: the European
cups and the Copa del Rey on weekday tickets, the Swedish, Norwegian and
Brazilian leagues on summer tickets, the Premier League and Serie A now and
then. ESPN's scoreboard (the JSON its own website uses, the same as the
Liga F's, ``quiniela.io.espn``) has all of them with one name and one id per
team across competitions, so the European cups link the leagues of every
country into one rating (``quiniela.prediction.elo_clubs``).

Every month of every competition in ``COMPETITIONS`` is read once (about
6,500 small requests for the whole history, from ``FIRST_YEAR``) and only the
matches are kept, one Parquet file per competition and month in
``data/clubs/espn/<competition>/<YYYYMM>.parquet``. Completed months are never
asked again; the last month, the current one and the next ones up to
``AHEAD_MONTHS`` (fixtures) are refreshed on every update.

The endpoint is not an official, documented API: it may change without
notice, so everything downloaded is kept locally.
"""

import http.client
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quiniela.config import CLUBS_DIR

logger = logging.getLogger(__name__)

SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{competition}/scoreboard"
    "?dates={month}&limit=1000"
)

# ESPN competitions of the club rating: the leagues whose teams reach the tickets (and
# Spain's, to rate the Spanish clubs of the cups), and the cups that link them
LEAGUES = [
    "esp.1",
    "esp.2",
    "eng.1",
    "eng.2",
    "ger.1",
    "ita.1",
    "fra.1",
    "por.1",
    "ned.1",
    "bel.1",
    "tur.1",
    "gre.1",
    "sco.1",
    "sco.2",
    "aut.1",
    "sui.1",
    "den.1",
    "swe.1",
    "nor.1",
    "rus.1",
    "bra.1",
    "arg.1",
    "usa.1",
    "mex.1",
]
CUPS = [
    "uefa.champions",
    "uefa.champions_qual",
    "uefa.europa",
    "uefa.europa_qual",
    "uefa.europa.conf",
    "uefa.europa.conf_qual",
    "uefa.super_cup",
    "esp.copa_del_rey",
    "esp.super_cup",
    "eng.fa",
    "eng.league_cup",
    "ger.dfb_pokal",
    "fifa.cwc",
    "conmebol.libertadores",
]
COMPETITIONS = [*LEAGUES, *CUPS]

# First year downloaded (2009: one year to settle the ratings before the 2010-11 tickets)
FIRST_YEAR = 2009

# Months after the current one asked for their fixtures
AHEAD_MONTHS = 1

# Statuses of a finished match, parallel requests and a polite pause after each one
FINISHED = {"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
WORKERS = 4
PAUSE_SECONDS = 0.2

# Attempts of every request and the wait before trying again (doubled every time)
ATTEMPTS = 3
RETRY_SECONDS = 2.0

ESPN_DIR = CLUBS_DIR / "espn"

# Every monthly file joined, in ESPN_DIR (rebuilt by load_club_matches when one is newer)
ALL_FILE = "all_matches.parquet"

COLUMNS = [
    "competition",
    "espn_id",
    "date",
    "home_id",
    "away_id",
    "home_team",
    "away_team",
    "FTHG",
    "FTAG",
    "FTR",
    "status",
    "neutral",
]


def parse_events(data: dict[str, Any], competition: str) -> pd.DataFrame:
    """Matches of a scoreboard JSON.

    Args:
        data: ESPN scoreboard JSON.
        competition: ESPN code of its competition.

    Returns:
        One row per match with ``COLUMNS`` (scores empty unless finished;
        ``date`` in Madrid time).
    """
    rows = []
    for event in data.get("events", []) or []:
        competitions = event.get("competitions") or [{}]
        sides = {c.get("homeAway"): c for c in competitions[0].get("competitors", [])}
        if "home" not in sides or "away" not in sides:
            continue
        status = event["status"]["type"]["name"]
        finished = status in FINISHED
        home_goals = float(sides["home"]["score"]) if finished else None
        away_goals = float(sides["away"]["score"]) if finished else None
        result = None
        if home_goals is not None and away_goals is not None:
            result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
        rows.append(
            {
                "competition": competition,
                "espn_id": str(event["id"]),
                "date": pd.Timestamp(event["date"]).tz_convert("Europe/Madrid").tz_localize(None),
                "home_id": str(sides["home"]["team"]["id"]),
                "away_id": str(sides["away"]["team"]["id"]),
                "home_team": sides["home"]["team"]["displayName"],
                "away_team": sides["away"]["team"]["displayName"],
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": result,
                "status": status,
                "neutral": bool(competitions[0].get("neutralSite", False)),
            }
        )
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.astype({"FTHG": float, "FTAG": float, "neutral": bool})


def _fetch(url: str) -> dict[str, Any]:
    """Download one scoreboard JSON, trying again after a failure (ESPN times out at times).

    Raises:
        OSError: The last attempt failed too.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "quiniela (personal research)"})
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed host
                data: dict[str, Any] = json.loads(response.read())
            return data
        except (OSError, ValueError, http.client.HTTPException) as error:
            if attempt == ATTEMPTS - 1:
                raise OSError(f"{url}: {error}") from error
            time.sleep(RETRY_SECONDS * (attempt + 1))
    raise OSError(url)  # pragma: no cover - the loop always returns or raises


def months(today: date, first_year: int = FIRST_YEAR) -> list[str]:
    """Months to have, ``YYYYMM``, from January of ``first_year`` to ``AHEAD_MONTHS`` ahead."""
    last = pd.Period(today, "M") + AHEAD_MONTHS
    return [p.strftime("%Y%m") for p in pd.period_range(f"{first_year}-01", last, freq="M")]


def _fresh(month: str, today: date) -> bool:
    """Whether a month must be asked again: the last one, the current one and the fixtures."""
    last: str = (pd.Period(today, "M") - 1).strftime("%Y%m")
    return month >= last


def download_competition(
    competition: str,
    today: date | None = None,
    output_dir: Path = ESPN_DIR,
    first_year: int = FIRST_YEAR,
) -> int:
    """Download the missing (and refresh the recent) months of one competition.

    Args:
        competition: ESPN code.
        today: Reference date. Defaults to today.
        output_dir: Folder of the monthly files.
        first_year: First year downloaded.

    Returns:
        Number of months downloaded.
    """
    today = today or date.today()
    folder = output_dir / competition
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for month in months(today, first_year):
        path = folder / f"{month}.parquet"
        if path.exists() and not _fresh(month, today):
            continue
        try:
            data = _fetch(SCOREBOARD.format(competition=competition, month=month))
        except OSError as error:
            # Left for the next update: the month has no file, so it is asked again
            logger.warning("ESPN %s %s not downloaded: %s", competition, month, error)
            continue
        parse_events(data, competition).to_parquet(path, index=False)
        done += 1
        time.sleep(PAUSE_SECONDS)
    return done


def update_clubs(today: date | None = None, output_dir: Path = ESPN_DIR) -> int:
    """Download every competition of ``COMPETITIONS`` (a few in parallel).

    Args:
        today: Reference date. Defaults to today.
        output_dir: Folder of the monthly files.

    Returns:
        Number of months downloaded.
    """
    with ThreadPoolExecutor(WORKERS) as pool:
        counts = list(pool.map(lambda c: download_competition(c, today, output_dir), COMPETITIONS))
    total = sum(counts)
    logger.info("ESPN club matches: %d months downloaded", total)
    return total


def load_club_matches(output_dir: Path = ESPN_DIR) -> pd.DataFrame:
    """Every club match downloaded (played and fixtures), in date order.

    The thousands of monthly files are slow to read one by one, so they are
    joined into ``ALL_FILE`` (next to the competition folders), rebuilt
    whenever a monthly file is newer.

    Args:
        output_dir: Folder of the monthly files.

    Returns:
        ``COLUMNS``, one row per ESPN match; empty when nothing was downloaded.
    """
    paths = sorted(output_dir.glob("*/*.parquet")) if output_dir.exists() else []
    if not paths:
        return pd.DataFrame(columns=COLUMNS)
    joined = output_dir / ALL_FILE
    newest = max(p.stat().st_mtime for p in paths)
    if joined.exists() and joined.stat().st_mtime >= newest:
        return pd.read_parquet(joined)
    frames = [frame for frame in (pd.read_parquet(p) for p in paths) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    matches = pd.concat(frames, ignore_index=True).drop_duplicates("espn_id", keep="last")
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    matches.to_parquet(joined, index=False)
    return matches
