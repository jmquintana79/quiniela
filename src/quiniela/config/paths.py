"""Filesystem paths used across the project.

All paths are derived from the project root, so they work regardless of the
current working directory. The data root can be overridden with the
``QUINIELA_DATA_DIR`` environment variable (e.g. to keep data on another disk).
"""

import os
from pathlib import Path

# Project root: <root>/src/quiniela/config/paths.py -> parents[3] == <root>
ROOT_DIR: Path = Path(__file__).resolve().parents[3]

# Root folder for every dataset handled by the project
DATA_DIR: Path = Path(os.environ.get("QUINIELA_DATA_DIR", ROOT_DIR / "data"))

# Completed seasons, one file per league and season
HISTORICAL_DIR: Path = DATA_DIR / "historical"

# Season in progress: results played so far and upcoming fixtures
CURRENT_DIR: Path = DATA_DIR / "current"

# Model predictions for the season in progress, one file per model
PREDICTIONS_DIR: Path = DATA_DIR / "predictions"

# Walk-forward predictions of past seasons (backtest), all models in one file
BACKTEST_DIR: Path = DATA_DIR / "backtest"

# Raw files cached by soccerdata (avoids downloading completed seasons twice)
CACHE_DIR: Path = DATA_DIR / "cache"

# Generated HTML dashboards (rebuilt from the data folders)
DASHBOARDS_DIR: Path = ROOT_DIR / "dashboards"


def ensure_data_dirs() -> None:
    """Create the data folders if they do not exist yet."""
    for path in (HISTORICAL_DIR, CURRENT_DIR, PREDICTIONS_DIR, BACKTEST_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
