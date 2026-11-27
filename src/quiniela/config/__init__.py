"""Project configuration: filesystem paths and other static settings."""

from quiniela.config import paths
from quiniela.config.paths import (
    BACKTEST_DIR,
    CACHE_DIR,
    CLUBS_DIR,
    CURRENT_DIR,
    DASHBOARDS_DIR,
    DATA_DIR,
    HISTORICAL_DIR,
    LAE_DIR,
    METRICS_DIR,
    NATIONAL_DIR,
    PREDICTIONS_DIR,
    PUBLIC_DIR,
    ROOT_DIR,
    WOMEN_DIR,
    ensure_data_dirs,
)

# The submodule is listed so the API docs (pdoc) include it
__all__ = [
    "BACKTEST_DIR",
    "CACHE_DIR",
    "CLUBS_DIR",
    "CURRENT_DIR",
    "DASHBOARDS_DIR",
    "DATA_DIR",
    "HISTORICAL_DIR",
    "LAE_DIR",
    "METRICS_DIR",
    "NATIONAL_DIR",
    "PREDICTIONS_DIR",
    "PUBLIC_DIR",
    "ROOT_DIR",
    "WOMEN_DIR",
    "ensure_data_dirs",
    "paths",
]
