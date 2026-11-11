"""Project configuration: filesystem paths and other static settings."""

from quiniela.config import paths
from quiniela.config.paths import (
    BACKTEST_DIR,
    CACHE_DIR,
    CURRENT_DIR,
    DASHBOARDS_DIR,
    DATA_DIR,
    HISTORICAL_DIR,
    PREDICTIONS_DIR,
    ROOT_DIR,
    ensure_data_dirs,
)

# The submodule is listed so the API docs (pdoc) include it
__all__ = [
    "BACKTEST_DIR",
    "CACHE_DIR",
    "CURRENT_DIR",
    "DASHBOARDS_DIR",
    "DATA_DIR",
    "HISTORICAL_DIR",
    "PREDICTIONS_DIR",
    "ROOT_DIR",
    "ensure_data_dirs",
    "paths",
]
