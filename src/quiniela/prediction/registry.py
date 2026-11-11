"""Model registry: the interface every model follows and the available models.

To add a model, implement it in its own module (scikit-learn estimator API)
and register it in ``MODELS``; tasks, the pipeline, the backtest and the
dashboards pick it up by its key. The registry order is the column order in
the dashboards (oldest version first, references last).

Models receive the whole match table as ``X`` (league, date, teams, odds and,
when fitting, goals) and read the columns they need; ``y`` is the full-time
result (``H``, ``D``, ``A``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd

from quiniela.prediction.dixon_coles import DixonColesModel
from quiniela.prediction.head_to_head import HeadToHeadModel
from quiniela.prediction.market import MarketModel


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
        version: Model version (``v1``, ``v2``...); empty for references.
        reference: True for baselines that are not candidate models (the market).
    """

    name: str
    factory: Callable[[], Any]
    description: str
    version: str = ""
    reference: bool = False


# Registry: key (used in tasks and file names) -> model
MODELS: dict[str, ModelSpec] = {
    "head_to_head": ModelSpec(
        name="Head-to-head",
        factory=HeadToHeadModel,
        description="Past results between the same teams at the same ground, smoothed "
        "towards the league's home/draw/away shares.",
        version="v1",
    ),
    "dixon_coles": ModelSpec(
        name="Dixon-Coles",
        factory=DixonColesModel,
        description="Poisson goals from time-weighted team attack and defence strengths, "
        "with league goal levels, home advantage and a low-score correction.",
        version="v2",
    ),
    "market": ModelSpec(
        name="Market",
        factory=MarketModel,
        description="Probabilities implied by the bookmakers' pre-match odds (margin "
        "removed): the reference every model is compared against.",
        reference=True,
    ),
}

DEFAULT_MODEL = "head_to_head"
