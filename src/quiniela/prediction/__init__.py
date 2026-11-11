"""Prediction models and the pipeline that applies them to the current season.

Every model follows the scikit-learn estimator API (``fit``, ``predict``,
``predict_proba`` and ``classes_`` in ``H``, ``D``, ``A`` order). To add a
model, implement it in its own module and register it in
``quiniela.prediction.registry.MODELS``.

Submodules:

- ``registry``: model interface and registered models.
- ``head_to_head``: v1, head-to-head benchmark.
- ``dixon_coles``: v2, Dixon-Coles team strengths with time decay.
- ``market``: market reference (bookmakers' implied probabilities).
- ``pipeline``: walk-forward predictions of matchdays and their storage.
- ``ticket``: quiniela ticket composition and prize rules.
"""

from quiniela.prediction import (
    dixon_coles,
    head_to_head,
    market,
    pipeline,
    registry,
    ticket,
)
from quiniela.prediction.dixon_coles import DixonColesModel
from quiniela.prediction.head_to_head import HeadToHeadModel
from quiniela.prediction.market import MarketModel
from quiniela.prediction.registry import DEFAULT_MODEL, MODELS, ModelSpec, Predictor

# Submodules are listed so the API docs (pdoc) include them
__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "DixonColesModel",
    "HeadToHeadModel",
    "MarketModel",
    "ModelSpec",
    "Predictor",
    "dixon_coles",
    "head_to_head",
    "market",
    "pipeline",
    "registry",
    "ticket",
]
