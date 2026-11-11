"""Model evaluation: probabilistic metrics and the multi-season backtest.

Submodules:

- ``metrics``: hit rate, log-loss, Brier score, RPS and calibration.
- ``backtest``: walk-forward predictions of every registered model over past seasons.
"""

from quiniela.evaluation import backtest, metrics

# Submodules are listed so the API docs (pdoc) include them
__all__ = ["backtest", "metrics"]
