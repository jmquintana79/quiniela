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

# Official quiniela draws (real tickets, winners and prizes) saved from Loterías y
# Apuestas del Estado, one JSON file per season (see quiniela.data.prizes)
LAE_DIR: Path = DATA_DIR / "lae"

# Every next-ticket file saved (data/lae/upcoming.json), one per jornada: the
# history of the public's picks, the v1F predictions of the Liga F
PUBLIC_DIR: Path = LAE_DIR / "public"

# Liga F (women's) matches of the official tickets with results and public picks
WOMEN_DIR: Path = DATA_DIR / "women"

# Results of every men's national-team match since 1872 (quiniela.io.international)
NATIONAL_DIR: Path = DATA_DIR / "national"

# Metrics and KPIs of every official ticket, per match and per draw (quiniela.evaluation.ticket_metrics)
METRICS_DIR: Path = DATA_DIR / "metrics"

# Raw files cached by soccerdata (avoids downloading completed seasons twice)
CACHE_DIR: Path = DATA_DIR / "cache"

# Generated HTML dashboards (rebuilt from the data folders)
DASHBOARDS_DIR: Path = ROOT_DIR / "dashboards"


def ensure_data_dirs() -> None:
    """Create the data folders if they do not exist yet."""
    folders = (HISTORICAL_DIR, CURRENT_DIR, PREDICTIONS_DIR, BACKTEST_DIR, LAE_DIR, CACHE_DIR)
    for path in (*folders, PUBLIC_DIR, WOMEN_DIR, NATIONAL_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)
