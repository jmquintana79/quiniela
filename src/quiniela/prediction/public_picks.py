r"""Public picks (v1F): the Liga F model, from the share of bettors marking each sign.

The women's league has no data source in the project yet, so its model is
the quiniela public: the "Pronósticos más apostados" of the betting page,
the share $s_k$ of bettors marking each sign $k \in \lbrace 1, X, 2 \rbrace$,
normalised to probabilities,

$$
P(k) = \frac{s_k}{s_1 + s_X + s_2}
$$

and the predicted sign is the most marked one. Women's league models have
their own version counter with an ``F`` (v1F, v2F...) and are evaluated apart
from the men's (``quiniela.evaluation.women``).

**Without public picks** (tickets whose betting page was not saved) the
model falls back to the shares of 1, X and 2 of the training results, which
is also how the tickets plan those matches.
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

OUTCOMES = ["H", "D", "A"]

# Columns with the public's share of each sign (percent or any scale)
PUBLIC_COLUMNS = ["public_H", "public_D", "public_A"]


class PublicPicksModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Predict 1X2 results from the share of bettors marking each sign (v1F)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Learn the result shares used when a match has no public picks.

        Args:
            X: Played matches (only their number matters).
            y: Full-time result of each match (``H``, ``D`` or ``A``).

        Returns:
            The fitted model.
        """
        self.classes_ = np.array(OUTCOMES)
        results = pd.Series(np.asarray(y))
        shares = results.value_counts(normalize=True)
        # Without training results, the three signs are equally likely
        self.prior_ = np.array([float(shares.get(o, 1 / 3)) for o in OUTCOMES])
        if not np.isclose(self.prior_.sum(), 1.0):
            self.prior_ = np.full(3, 1 / 3)
        return self

    def has_picks(self, X: pd.DataFrame) -> np.ndarray:
        """Whether each match has the public's picks.

        Args:
            X: Matches, with ``PUBLIC_COLUMNS`` when the picks are known.

        Returns:
            Boolean array.
        """
        if not set(PUBLIC_COLUMNS) <= set(X.columns):
            return np.zeros(len(X), dtype=bool)
        known: np.ndarray = X[PUBLIC_COLUMNS].notna().all(axis=1).to_numpy()
        return known

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Normalised public shares, or the result shares without them.

        Args:
            X: Matches with ``PUBLIC_COLUMNS`` (missing values allowed).

        Returns:
            Array of shape ``(n_matches, 3)`` in ``H``, ``D``, ``A`` order.
        """
        check_is_fitted(self, "prior_")
        probs = np.tile(self.prior_, (len(X), 1))
        known = self.has_picks(X)
        if known.any():
            shares = X.loc[known, PUBLIC_COLUMNS].to_numpy(dtype=float)
            probs[known] = shares / shares.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most marked sign of each match (the most frequent result without picks).

        Args:
            X: Matches with ``PUBLIC_COLUMNS``.

        Returns:
            Array of results.
        """
        predicted: np.ndarray = self.classes_[self.predict_proba(X).argmax(axis=1)]
        return predicted
