"""Model registry: the interface every model follows and the available models.

To add a model, implement it in its own module (scikit-learn estimator API)
and register it in ``MODELS``; tasks, the pipeline and the dashboard pick it
up by its key.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd

from quiniela.prediction.head_to_head import HeadToHeadModel


class Predictor(Protocol):
    """Interface shared by every model (scikit-learn classifier API)."""

    classes_: np.ndarray

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Train on played matches."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted result (``H``, ``D`` or ``A``) of each match."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the probability of each result, in ``classes_`` order."""
        ...


@dataclass(frozen=True)
class ModelSpec:
    """A registered model.

    Attributes:
        name: Display name (dashboard column header).
        factory: Builds a new, unfitted model.
        description: One-line summary shown in the dashboard.
    """

    name: str
    factory: Callable[[], Any]
    description: str


# Registry: key (used in tasks and file names) -> model
MODELS: dict[str, ModelSpec] = {
    "head_to_head": ModelSpec(
        name="Head-to-head",
        factory=HeadToHeadModel,
        description="Past results between the same teams at the same ground, smoothed "
        "towards the league's home/draw/away shares.",
    ),
}

DEFAULT_MODEL = "head_to_head"
