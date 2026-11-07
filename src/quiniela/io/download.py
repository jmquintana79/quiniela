"""Download Spanish league match data from football-data.co.uk.

Data is retrieved through the ``soccerdata`` library (``MatchHistory`` reader),
which normalises column and team names and caches the raw CSV files. The
upcoming fixtures file is not covered by soccerdata, so it is read directly
from football-data.co.uk and normalised to the same format.

Seasons are identified by their **start year**: ``2015`` means the 2015-16
season.

Output layout::

    data/historical/<code>/<season>.parquet   # completed seasons
    data/current/<code>/results.parquet       # season in progress: played matches
    data/current/<code>/fixtures.parquet      # season in progress: upcoming matches

where ``<code>`` is the football-data.co.uk league code (``SP1``, ``SP2``) and
``<season>`` is the season label (e.g. ``2015-16``).
"""

import csv
import io
import logging
from datetime import date
from pathlib import Path
from typing import IO

import pandas as pd
import requests
import soccerdata as sd
from soccerdata import _config as sd_config
from soccerdata import match_history as sd_match_history

from quiniela.config import CACHE_DIR, CURRENT_DIR, HISTORICAL_DIR, ensure_data_dirs

logger = logging.getLogger(__name__)

# soccerdata league id -> football-data.co.uk league code
LEAGUES: dict[str, str] = {
    "ESP-La Liga": "SP1",
    "ESP-Segunda": "SP2",
}

# First season (start year) available on football-data.co.uk for each league
FIRST_SEASON: dict[str, int] = {
    "SP1": 1993,
    "SP2": 1996,
}

# Month in which a new season starts to be considered the current one
SEASON_START_MONTH = 7

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
REQUEST_TIMEOUT = 30  # seconds

# Same column names soccerdata uses for the results files
FIXTURE_COLUMNS: dict[str, str] = {
    "Div": "league",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Referee": "referee",
}


def _register_leagues() -> None:
    """Add the leagues missing from soccerdata's built-in configuration.

    soccerdata only ships "ESP-La Liga". Segunda División is registered here
    instead of in ``~/soccerdata/config/league_dict.json`` so the project does
    not depend on a file outside the repository.
    """
    sd_config.LEAGUE_DICT.setdefault(
        "ESP-Segunda",
        {
            "MatchHistory": "SP2",
            "season_start": "Aug",
            # Promotion play-offs run into June
            "season_end": "Jun",
        },
    )
    # soccerdata caches the league mapping per reader class; reset it so the
    # newly registered league is picked up
    if hasattr(sd.MatchHistory, "_all_leagues_dict"):
        del sd.MatchHistory._all_leagues_dict


def _parse_csv(raw_data: IO[bytes], lkey: str, skey: str) -> pd.DataFrame:
    """Parse a football-data.co.uk CSV without dropping malformed rows.

    Replaces soccerdata's parser, which skips rows with more fields than the
    header. Several old files (e.g. La Liga 2004-05, Segunda 2002-03) contain
    such rows: the extra fields are trailing odds or empty values, while the
    match columns stay aligned with the header. Reading with as many columns
    as the longest row keeps every match; the extra columns are named
    ``Unnamed: <n>`` so soccerdata drops them afterwards.

    Args:
        raw_data: Raw CSV file.
        lkey: football-data.co.uk league code (only used for logging).
        skey: Season code, e.g. ``"0405"`` (only used for logging).

    Returns:
        One row per line of the file.
    """
    logger.debug("Parsing league=%s season=%s", lkey, skey)
    raw = raw_data.read()
    try:
        # Files since 2024-25 start with a BOM; "utf-8-sig" also reads files without one
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    width = max(len(row) for row in rows)
    names = header + [f"Unnamed: {i}" for i in range(len(header), width)]
    # Blank header cells (trailing commas) must also be unique and droppable
    names = [name or f"Unnamed: {i}" for i, name in enumerate(names)]
    # Skip the original header: with header=0 pandas would still size rows from it
    return pd.read_csv(io.StringIO(text), header=None, skiprows=1, names=names)


_register_leagues()
# MatchHistory.read_games() looks the parser up at call time, so this replaces it
sd_match_history._parse_csv = _parse_csv


def current_season(today: date | None = None) -> int:
    """Return the start year of the season in progress.

    Args:
        today: Reference date. Defaults to the current date.

    Returns:
        Start year of the current season (e.g. ``2026`` for 2026-27).
    """
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def season_label(start_year: int) -> str:
    """Format a season start year as a label (``2015`` -> ``"2015-16"``)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _season_code(start_year: int) -> str:
    """Format a season start year as a football-data.co.uk code (``2015`` -> ``"1516"``)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _resolve_leagues(leagues: list[str] | None) -> dict[str, str]:
    """Validate the requested leagues and map them to their league codes.

    Args:
        leagues: soccerdata league ids. ``None`` selects every league in ``LEAGUES``.

    Returns:
        Mapping from soccerdata league id to football-data.co.uk code.

    Raises:
        ValueError: If a league is not supported.
    """
    if leagues is None:
        return dict(LEAGUES)
    unknown = sorted(set(leagues) - set(LEAGUES))
    if unknown:
        raise ValueError(f"Unsupported leagues {unknown}. Valid leagues: {sorted(LEAGUES)}")
    return {league: LEAGUES[league] for league in leagues}


def _read_games(league: str, seasons: list[int]) -> pd.DataFrame:
    """Read match results for one league through soccerdata.

    soccerdata re-downloads the season in progress on every call and serves
    completed seasons from its cache.

    Args:
        league: soccerdata league id.
        seasons: Season start years.

    Returns:
        One row per match, indexed by ``(league, season, game)``.
    """
    reader = sd.MatchHistory(
        leagues=league,
        seasons=[f"{year}-{year + 1}" for year in seasons],
        data_dir=CACHE_DIR / "MatchHistory",
    )
    return reader.read_games()


def _save(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to parquet, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def download_historical(
    start_year: int,
    end_year: int,
    leagues: list[str] | None = None,
) -> list[Path]:
    """Download completed seasons and store them in the historical folder.

    Seasons before a league's first available season are skipped with a
    warning, so the same range can be used for both divisions.

    Args:
        start_year: Start year of the first season to download (inclusive).
        end_year: Start year of the last season to download (inclusive). It
            must be earlier than the current season, which is handled by
            :func:`download_current`.
        leagues: soccerdata league ids. ``None`` downloads every league in ``LEAGUES``.

    Returns:
        Paths of the files written, one per league and season.

    Raises:
        ValueError: If the range is empty or reaches the current season.
    """
    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year})")
    last_completed = current_season() - 1
    if end_year > last_completed:
        raise ValueError(
            f"end_year ({end_year}) must be <= {last_completed}: season "
            f"{season_label(last_completed + 1)} is in progress, use download_current()"
        )

    ensure_data_dirs()
    written: list[Path] = []
    for league, code in _resolve_leagues(leagues).items():
        first = max(start_year, FIRST_SEASON[code])
        if first > start_year:
            logger.warning(
                "%s has no data before %s; skipping earlier seasons", league, season_label(first)
            )
        seasons = list(range(first, end_year + 1))
        if not seasons:
            continue

        logger.info(
            "Downloading %s seasons %s to %s",
            league,
            season_label(seasons[0]),
            season_label(seasons[-1]),
        )
        games = _read_games(league, seasons)
        # One file per season makes it easy to add or refresh a single season
        for year in seasons:
            season_games = games[games.index.get_level_values("season") == _season_code(year)]
            if season_games.empty:
                logger.warning("No matches found for %s %s", league, season_label(year))
                continue
            path = HISTORICAL_DIR / code / f"{season_label(year)}.parquet"
            written.append(_save(season_games, path))
    return written


def _read_fixtures(leagues: dict[str, str]) -> pd.DataFrame:
    """Download the upcoming fixtures for the given leagues.

    The fixtures file lists every league covered by football-data.co.uk; only
    the requested ones are kept. Columns, team names and the index are
    normalised to match soccerdata's results.

    Args:
        leagues: Mapping from soccerdata league id to football-data.co.uk code.

    Returns:
        One row per upcoming match (including pre-match odds), indexed by
        ``(league, season, game)``.
    """
    response = requests.get(FIXTURES_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    fixtures = pd.read_csv(io.BytesIO(response.content), encoding="UTF-8-SIG")

    codes_to_league = {code: league for league, code in leagues.items()}
    fixtures = fixtures[fixtures["Div"].isin(codes_to_league)].rename(columns=FIXTURE_COLUMNS)
    if fixtures.empty:
        # Keep the same index so callers can filter it like a non-empty result
        return fixtures.assign(season="", game="").set_index(["league", "season", "game"])

    fixtures["time"] = fixtures["time"].fillna("12:00")
    fixtures["date"] = pd.to_datetime(
        fixtures["date"] + " " + fixtures["time"], format="mixed", dayfirst=True
    )
    fixtures = fixtures.drop(columns="time").assign(
        league=fixtures["league"].map(codes_to_league),
        season=_season_code(current_season()),
        home_team=fixtures["home_team"].replace(sd_config.TEAMNAME_REPLACEMENTS),
        away_team=fixtures["away_team"].replace(sd_config.TEAMNAME_REPLACEMENTS),
    )
    fixtures["game"] = (
        fixtures["date"].dt.strftime("%Y-%m-%d")
        + " "
        + fixtures["home_team"]
        + "-"
        + fixtures["away_team"]
    )
    return fixtures.set_index(["league", "season", "game"]).sort_index()


def download_current(leagues: list[str] | None = None) -> list[Path]:
    """Download the season in progress and store it in the current folder.

    Two files are written per league: the results of the matches already
    played and the upcoming fixtures (the next matchday to predict). Run it
    weekly to refresh both; existing files are overwritten.

    Args:
        leagues: soccerdata league ids. ``None`` downloads every league in ``LEAGUES``.

    Returns:
        Paths of the files written.
    """
    ensure_data_dirs()
    selected = _resolve_leagues(leagues)
    season = current_season()
    written: list[Path] = []

    for league, code in selected.items():
        logger.info("Downloading %s %s results", league, season_label(season))
        results = _read_games(league, [season])
        written.append(_save(results, CURRENT_DIR / code / "results.parquet"))

    logger.info("Downloading upcoming fixtures")
    fixtures = _read_fixtures(selected)
    for league, code in selected.items():
        league_fixtures = fixtures[fixtures.index.get_level_values("league") == league]
        if league_fixtures.empty:
            logger.warning("No upcoming fixtures published for %s", league)
        written.append(_save(league_fixtures, CURRENT_DIR / code / "fixtures.parquet"))
    return written
