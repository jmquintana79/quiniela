"""Prediction models and the pipeline that applies them to the current season.

Every model follows the scikit-learn estimator API (``fit``, ``predict``,
``predict_proba`` and ``classes_`` in ``H``, ``D``, ``A`` order). To add a
model, implement it in its own module and register it in
``quiniela.prediction.registry.MODELS``.

Submodules:

- ``registry``: model interface and registered models.
- ``head_to_head``: head-to-head benchmark model.
- ``pipeline``: predict current-season matchdays and store the predictions.
- ``ticket``: quiniela ticket composition and prize rules.
"""

from quiniela.prediction import head_to_head, pipeline, registry, ticket
from quiniela.prediction.head_to_head import HeadToHeadModel
from quiniela.prediction.registry import DEFAULT_MODEL, MODELS, ModelSpec, Predictor

# Submodules are listed so the API docs (pdoc) include them
__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "HeadToHeadModel",
    "ModelSpec",
    "Predictor",
    "head_to_head",
    "pipeline",
    "registry",
    "ticket",
]
