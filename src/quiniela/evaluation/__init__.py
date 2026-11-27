"""Model evaluation: probabilistic metrics and the multi-season backtest.

Submodules:

- ``metrics``: hit rate, log-loss, Brier score, RPS and calibration.
- ``backtest``: walk-forward predictions of every registered model over past seasons.
- ``money``: the real tickets of past draws played in euros (v3+v3F·D, v3+v3F·R, v3+v3F·B, v3+v3F·E).
- ``women``: the Liga F models (v1F...), evaluated apart on the women's ticket matches.
- ``clubs``: the club models (v1C...) of the other competitions, evaluated apart on every
  ESPN club match since 2011 and on the club ticket matches outside La Liga / Segunda.
- ``national``: the national-team models (v1N...), evaluated apart on every international
  since 2010 and on the national-team ticket matches.
- ``ticket_metrics``: metrics and KPIs of every official ticket, per match and per draw.
"""

from quiniela.evaluation import backtest, clubs, metrics, money, national, ticket_metrics, women

# Submodules are listed so the API docs (pdoc) include them
__all__ = ["backtest", "clubs", "metrics", "money", "national", "ticket_metrics", "women"]
