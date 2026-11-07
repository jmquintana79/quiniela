"""Load the downloaded match files into tidy DataFrames.

Every loader returns one row per match with the same columns (``MATCH_COLUMNS``),
so historical and current matches can be concatenated directly:

- ``league``: football-data.co.uk league code (``SP1``, ``SP2``).
- ``game``: match id, ``"<date> <home>-<away>"`` (unique within a league).
- ``date``, ``home_team``, ``away_team``.
- ``FTHG``, ``FTAG``, ``FTR``: full-time goals and result (``H``, ``D``, ``A``);
  empty for matches not played yet.
"""

from pathlib import Path

import pandas as pd

from quiniela.config import CURRENT_DIR, HISTORICAL_DIR

MATCH_COLUMNS = ["league", "game", "date", "home_team", "away_team", "FTHG", "FTAG", "FTR"]


def _read_matches(path: Path, league: str) -> pd.DataFrame:
    """Read one parquet file and keep the match columns.

    Args:
        path: Results or fixtures parquet file written by ``quiniela.io.download``.
        league: League code stored in the ``league`` column.

    Returns:
        Matches with ``MATCH_COLUMNS`` (result columns are added empty if missing).
    """
    df = pd.read_parquet(path).reset_index()
    for column in ("FTHG", "FTAG", "FTR"):
        if column not in df.columns:
            df[column] = pd.NA
    df["league"] = league
    return df[MATCH_COLUMNS]


def _league_dirs(root: Path) -> list[Path]:
    """Return the league folders (``SP1``, ``SP2``...) inside a data folder."""
    return sorted(path for path in root.glob("*") if path.is_dir())


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
    if not frames:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values("date", ignore_index=True)


def load_current(current_dir: Path = CURRENT_DIR) -> pd.DataFrame:
    """Load the season in progress: played matches plus upcoming fixtures.

    A fixture that already appears among the results (same league and game id)
    is dropped, so each match appears once.

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
    if not frames:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    matches = pd.concat(frames, ignore_index=True)
    # Results come first, so keep="first" keeps the played version of a match
    matches = matches.drop_duplicates(subset=["league", "game"], keep="first")
    return matches.sort_values("date", ignore_index=True)
