"""Model evaluation: probabilistic metrics and the multi-season backtest.

Submodules:

- ``metrics``: hit rate, log-loss, Brier score, RPS and calibration.
- ``backtest``: walk-forward predictions of every registered model over past seasons.
- ``money``: the real tickets of past draws played in euros (v3+v2F·D, v3+v2F·R, v3+v2F·B, v3+v2F·E).
- ``women``: the Liga F models (v1F...), evaluated apart on the women's ticket matches.
- ``ticket_metrics``: metrics and KPIs of every official ticket, per match and per draw.
"""

from quiniela.evaluation import backtest, metrics, money, ticket_metrics, women

# Submodules are listed so the API docs (pdoc) include them
__all__ = ["backtest", "metrics", "money", "ticket_metrics", "women"]
