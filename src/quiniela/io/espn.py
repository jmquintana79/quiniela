"""Liga F (women's league) results and fixtures from ESPN's public scoreboard.

ESPN publishes the Liga F (``esp.w.1``) in the JSON its own website uses,
without bot protection, from 2022-23 (the first professional season) on:
every match with its date, teams, score and status, the fixtures of the
season in progress included. soccerdata's ESPN reader asks for every match
day and has no scores, so this module reads the monthly scoreboard instead
(about twelve small requests per season).

Every month is cached in ``data/cache/espn/`` (completed months are never
asked again) and every season is saved in ``data/women/espn/<season>.parquet``
with the same columns as the men's matches: ``league`` (``LIGAF``),
``season``, ``game``, ``date``, ``home_team``, ``away_team``, ``FTHG``,
``FTAG``, ``FTR``, the shots and shots on target of each team (``HS``, ``AS``,
``HST``, ``AST``, as football-data names them; the scoreboard has them for every
finished match), plus ``status`` and ``espn_id``.

The endpoint is not an official, documented API: it may change without
notice, so everything downloaded is kept locally.
"""

import json
import logging
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quiniela.config import CACHE_DIR, WOMEN_DIR

logger = logging.getLogger(__name__)

# ESPN's code of the Liga F and the first season it covers
LIGA_F = "esp.w.1"
FIRST_SEASON = 2022
SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={month}"
)

# League code of the Liga F matches in the project's tables
LEAGUE_CODE = "LIGAF"

ESPN_DIR = WOMEN_DIR / "espn"
ESPN_CACHE = CACHE_DIR / "espn"

# Statuses of a finished match, and a polite pause between requests
FINISHED = {"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
PAUSE_SECONDS = 0.3

COLUMNS = [
    "league",
    "season",
    "game",
    "date",
    "home_team",
    "away_team",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "status",
    "espn_id",
]

# Team statistics of the scoreboard kept for every finished match: column -> ESPN name
STATISTICS = {"S": "totalShots", "ST": "shotsOnTarget"}


def _statistic(competitor: dict[str, Any], name: str) -> float | None:
    """A team statistic of a scoreboard match (``None`` when missing)."""
    for item in competitor.get("statistics") or []:
        if item.get("name") == name:
            try:
                return float(item.get("displayValue"))
            except (TypeError, ValueError):
                return None
    return None


def season_label(start: int) -> str:
    """Season label of a start year (``2026`` -> ``"2026-27"``)."""
    return f"{start}-{(start + 1) % 100:02d}"


def season_months(start: int) -> list[str]:
    """Months of a season, August to July, as ``YYYYMM``."""
    return [f"{start}{m:02d}" for m in range(8, 13)] + [f"{start + 1}{m:02d}" for m in range(1, 8)]


def _fetch(url: str) -> dict[str, Any]:
    """Download one scoreboard JSON."""
    request = urllib.request.Request(url, headers={"User-Agent": "quiniela (personal research)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data: dict[str, Any] = json.loads(response.read())
    return data


def _month(month: str, refresh: bool, cache_dir: Path) -> dict[str, Any]:
    """Scoreboard of a month, from the cache unless it must be refreshed."""
    path = cache_dir / LIGA_F / f"{month}.json"
    if path.exists() and not refresh:
        cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cached
    data = _fetch(SCOREBOARD.format(league=LIGA_F, month=month))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(PAUSE_SECONDS)
    return data


def parse_events(data: dict[str, Any], season: str) -> pd.DataFrame:
    """Matches of a scoreboard JSON.

    Args:
        data: ESPN scoreboard JSON.
        season: Season label of its matches.

    Returns:
        One row per match with ``COLUMNS`` (scores and statistics empty unless finished).
    """
    rows = []
    for event in data.get("events", []) or []:
        competition = event["competitions"][0]
        sides = {c["homeAway"]: c for c in competition["competitors"]}
        if "home" not in sides or "away" not in sides:
            continue
        status = event["status"]["type"]["name"]
        finished = status in FINISHED
        home_goals = float(sides["home"]["score"]) if finished else None
        away_goals = float(sides["away"]["score"]) if finished else None
        result = None
        if finished:
            result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"  # type: ignore[operator]
        rows.append(
            {
                "league": LEAGUE_CODE,
                "season": season,
                "date": pd.Timestamp(event["date"]).tz_convert("Europe/Madrid").tz_localize(None),
                "home_team": sides["home"]["team"]["displayName"],
                "away_team": sides["away"]["team"]["displayName"],
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": result,
                **{
                    f"{side[0].upper()}{column}": _statistic(sides[side], name)
                    if finished
                    else None
                    for column, name in STATISTICS.items()
                    for side in ("home", "away")
                },
                "status": status,
                "espn_id": str(event["id"]),
            }
        )
    out = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "game"])
    if out.empty:
        return pd.DataFrame(columns=COLUMNS)
    out["game"] = (
        out["date"].dt.strftime("%Y-%m-%d") + " " + out["home_team"] + "-" + out["away_team"]
    )
    return out[COLUMNS]


def download_season(
    start: int,
    current: bool = False,
    output_dir: Path = ESPN_DIR,
    cache_dir: Path = ESPN_CACHE,
    today: date | None = None,
) -> Path | None:
    """Download one Liga F season and save it as ``<season>.parquet``.

    Args:
        start: Start year of the season.
        current: The season in progress: months from the current one on are
            asked again (results and fixtures change), earlier ones come
            from the cache.
        output_dir: Folder of the season files.
        cache_dir: Folder of the monthly JSON cache.
        today: Reference date (which months are past). Defaults to today.

    Returns:
        The file written, or ``None`` when the season has no match.
    """
    today = today or date.today()
    this_month = f"{today.year}{today.month:02d}"
    frames = []
    for month in season_months(start):
        if current and month > this_month:
            # Future months only have fixtures: ask for them every time too
            refresh = True
        else:
            # The current (and last) month changes; older ones are final
            refresh = current and month >= f"{today.year}{today.month - 1:02d}"
        frames.append(parse_events(_month(month, refresh, cache_dir), season_label(start)))
    matches = pd.concat(frames, ignore_index=True).drop_duplicates("espn_id", keep="last")
    if matches.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{season_label(start)}.parquet"
    matches.sort_values("date").reset_index(drop=True).to_parquet(path, index=False)
    logger.info("Liga F %s: %d matches saved in %s", season_label(start), len(matches), path)
    return path


def update_liga_f(
    today: date | None = None, output_dir: Path = ESPN_DIR, cache_dir: Path = ESPN_CACHE
) -> list[Path]:
    """Download the missing completed Liga F seasons and refresh the current one.

    Args:
        today: Reference date. Defaults to today.
        output_dir: Folder of the season files.
        cache_dir: Folder of the monthly JSON cache.

    Returns:
        Files written.
    """
    today = today or date.today()
    current = today.year if today.month >= 8 else today.year - 1
    written = []
    for start in range(FIRST_SEASON, current):
        if not (output_dir / f"{season_label(start)}.parquet").exists():
            path = download_season(start, output_dir=output_dir, cache_dir=cache_dir, today=today)
            if path is not None:
                written.append(path)
    path = download_season(
        current, current=True, output_dir=output_dir, cache_dir=cache_dir, today=today
    )
    if path is not None:
        written.append(path)
    return written


def load_liga_f_matches(output_dir: Path = ESPN_DIR) -> pd.DataFrame:
    """Every Liga F match downloaded from ESPN (played and fixtures), by date.

    Args:
        output_dir: Folder of the season files.

    Returns:
        ``COLUMNS``; empty when nothing was downloaded.
    """
    paths = sorted(output_dir.glob("*.parquet")) if output_dir.exists() else []
    if not paths:
        return pd.DataFrame(columns=COLUMNS)
    matches = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return matches.sort_values("date").reset_index(drop=True)
