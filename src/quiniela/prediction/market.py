r"""Market reference: probabilities implied by the bookmakers' pre-match odds.

Not a model to beat the market with, but the realistic ceiling every model is
compared against. For decimal odds $o_H, o_D, o_A$ the implied probabilities,
normalised so the bookmaker margin is removed, are:

$$
p_k = \frac{1 / o_k}{1/o_H + 1/o_D + 1/o_A}, \qquad k \in \lbrace H, D, A \rbrace
$$

For each match the first bookmaker with the three odds available is used, in
``quiniela.data.loading.ODDS_PREFIXES`` order (market average first). Matches
without odds get no prediction (``NaN`` probabilities).
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from quiniela.data.loading import ODDS_PREFIXES

OUTCOMES = ["H", "D", "A"]


# scikit-learn ships without type hints, so its base classes are typed as Any
class MarketModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Turn the bookmakers' 1X2 odds of each match into probabilities.

    Args:
        prefixes: Bookmaker column prefixes in order of preference (e.g.
            ``"Avg"`` for ``AvgH``, ``AvgD``, ``AvgA``).
    """

    def __init__(self, prefixes: tuple[str, ...] = tuple(ODDS_PREFIXES)) -> None:
        self.prefixes = prefixes

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Nothing to learn: the odds of each match are its prediction.

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            The model.
        """
        self.classes_ = np.array(OUTCOMES)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Implied probabilities of each result.

        Args:
            X: Matches with odds columns (``<prefix>H``, ``<prefix>D``, ``<prefix>A``).

        Returns:
            Array of shape ``(n_matches, 3)`` (``H``, ``D``, ``A``); rows are
            ``NaN`` for matches without any complete set of odds.
        """
        probs = np.full((len(X), len(OUTCOMES)), np.nan)
        for prefix in self.prefixes:
            columns = [f"{prefix}{outcome}" for outcome in OUTCOMES]
            if not set(columns).issubset(X.columns):
                continue
            odds = X[columns].astype(float).to_numpy()
            usable = np.isnan(probs[:, 0]) & np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
            inverse = 1 / odds[usable]
            probs[usable] = inverse / inverse.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely result according to the market (``None`` without odds).

        Args:
            X: Matches with odds columns.

        Returns:
            Array of results (``H``, ``D``, ``A``) or ``None``.
        """
        probs = self.predict_proba(X)
        available = ~np.isnan(probs).any(axis=1)
        predicted = np.full(len(X), None, dtype=object)
        predicted[available] = np.asarray(OUTCOMES)[probs[available].argmax(axis=1)]
        return predicted
