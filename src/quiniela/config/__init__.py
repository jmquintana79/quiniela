"""Project configuration: filesystem paths and other static settings."""

from quiniela.config.paths import (
    CACHE_DIR,
    CURRENT_DIR,
    DASHBOARDS_DIR,
    DATA_DIR,
    HISTORICAL_DIR,
    PREDICTIONS_DIR,
    ROOT_DIR,
    ensure_data_dirs,
)

__all__ = [
    "CACHE_DIR",
    "CURRENT_DIR",
    "DASHBOARDS_DIR",
    "DATA_DIR",
    "HISTORICAL_DIR",
    "PREDICTIONS_DIR",
    "ROOT_DIR",
    "ensure_data_dirs",
]
