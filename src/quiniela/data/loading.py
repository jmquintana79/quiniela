"""Load the downloaded match files into tidy DataFrames.

Every loader returns one row per match with the same columns, so historical
and current matches can be concatenated directly:

- ``league``: football-data.co.uk league code (``SP1``, ``SP2``).
- ``season``: season label, e.g. ``"2015-16"``.
- ``game``: match id, ``"<date> <home>-<away>"`` (unique within a league).
- ``date``, ``home_team``, ``away_team``.
- ``FTHG``, ``FTAG``, ``FTR``: full-time goals and result (``H``, ``D``, ``A``);
  empty for matches not played yet.
- Pre-match 1X2 odds (``ODDS_COLUMNS``) of the bookmakers available in the
  file; seasons without a bookmaker get empty values.
"""

from pathlib import Path

import pandas as pd

from quiniela.config import CURRENT_DIR, HISTORICAL_DIR

MATCH_COLUMNS = [
    "league",
    "season",
    "game",
    "date",
    "home_team",
    "away_team",
    "FTHG",
    "FTAG",
    "FTR",
]

# Bookmakers whose 1X2 odds are kept, from most to least preferred: market
# average (since 2019-20), Betbrain average (2005-2019), Pinnacle, then single
# bookmakers available in older seasons
ODDS_PREFIXES = ["Avg", "BbAv", "PS", "B365", "BW", "IW", "WH", "LB"]
ODDS_COLUMNS = [f"{prefix}{outcome}" for prefix in ODDS_PREFIXES for outcome in "HDA"]

# A league pairing (home team at its ground) is played once per season
FIXTURE_KEY = ["league", "home_team", "away_team"]


def season_label(code: str) -> str:
    """Convert a football-data.co.uk season code into a label (``"9900"`` -> ``"1999-00"``).

    Args:
        code: Four-digit season code, e.g. ``"1516"``.

    Returns:
        Season label, e.g. ``"2015-16"``.
    """
    start = int(code[:2])
    year = 1900 + start if start >= 90 else 2000 + start
    return f"{year}-{code[2:]}"


def _read_matches(path: Path, league: str) -> pd.DataFrame:
    """Read one parquet file and keep the match and odds columns.

    Args:
        path: Results or fixtures parquet file written by ``quiniela.io.download``.
        league: League code stored in the ``league`` column.

    Returns:
        Matches with ``MATCH_COLUMNS`` plus the available ``ODDS_COLUMNS``
        (result columns are added empty if missing).
    """
    df = pd.read_parquet(path).reset_index()
    for column in ("FTHG", "FTAG", "FTR"):
        if column not in df.columns:
            df[column] = pd.NA
    df["league"] = league
    df["season"] = df["season"].astype(str).map(season_label)
    odds = [column for column in ODDS_COLUMNS if column in df.columns]
    return df[MATCH_COLUMNS + odds]


def _league_dirs(root: Path) -> list[Path]:
    """Return the league folders (``SP1``, ``SP2``...) inside a data folder."""
    return sorted(path for path in root.glob("*") if path.is_dir())


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate match frames, always returning every match and odds column."""
    if not frames:
        return pd.DataFrame(columns=MATCH_COLUMNS + ODDS_COLUMNS)
    matches = pd.concat(frames, ignore_index=True)
    return matches.reindex(columns=MATCH_COLUMNS + ODDS_COLUMNS)


def load_historical(historical_dir: Path = HISTORICAL_DIR) -> pd.DataFrame:
    """Load every completed season.

    Args:
        historical_dir: Folder with ``<code>/<season>.parquet`` files.

    Returns:
        All historical matches sorted by date.
    """
    frames = [
        _read_matches(path, league_dir.name)
        for league_dir in _league_dirs(historical_dir)
        for path in sorted(league_dir.glob("*.parquet"))
    ]
    return _concat(frames).sort_values("date", ignore_index=True)


def load_current(current_dir: Path = CURRENT_DIR) -> pd.DataFrame:
    """Load the season in progress: played matches plus upcoming fixtures.

    A fixture that already appears among the results is dropped, so each match
    appears once. The source's fixtures file can lag behind (it may still list
    last weekend's matches) and a rescheduled match changes its date, so a
    fixture is matched by league, home team and away team: each pairing is
    played once per season at the same ground.

    Args:
        current_dir: Folder with ``<code>/results.parquet`` and ``<code>/fixtures.parquet``.

    Returns:
        Current season matches sorted by date.
    """
    frames = []
    for league_dir in _league_dirs(current_dir):
        for name in ("results.parquet", "fixtures.parquet"):
            path = league_dir / name
            if path.exists():
                frames.append(_read_matches(path, league_dir.name))
    matches = _concat(frames)
    # Results come first, so keep="first" keeps the played version of a match
    matches = matches.drop_duplicates(subset=FIXTURE_KEY, keep="first")
    return matches.sort_values("date", ignore_index=True)


def pending_fixtures(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Drop the fixtures that have already been played.

    Args:
        fixtures: Upcoming matches of one league (``home_team``, ``away_team``).
        results: Played matches of the same league and season.

    Returns:
        The fixtures whose home/away pairing is not among the results.
    """
    played = pd.MultiIndex.from_frame(results[["home_team", "away_team"]])
    pairs = pd.MultiIndex.from_frame(fixtures[["home_team", "away_team"]])
    return fixtures[~pairs.isin(played)]
