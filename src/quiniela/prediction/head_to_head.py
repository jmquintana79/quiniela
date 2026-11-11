r"""Head-to-head benchmark: conditional probability of the result given the fixture.

For a match between home team $h$ and away team $a$ in league $\ell$, played
at $h$'s ground, the model estimates:

$$
P(r \mid h, a) = \frac{n_{h,a}(r) + \alpha \, \pi_\ell(r)}{n_{h,a} + \alpha},
\qquad r \in \lbrace H, D, A \rbrace
$$

where:

- $n_{h,a}(r)$ counts the past matches of $h$ at home against $a$ that ended
  with result $r$ (home win, draw or away win), and $n_{h,a}$ is the total
  number of those meetings;
- $\pi_\ell(r)$ is the share of result $r$ in league $\ell$ (the prior, e.g.
  about 46% home wins), or the global share for an unseen league;
- $\alpha \ge 0$ (additive smoothing) is how many "virtual matches" the prior
  is worth.

So teams that never met get the league prior, and with a long history the
observed frequencies dominate. The predicted sign is the most likely result:

$$
\hat{r} = \underset{r}{\arg\max} \; P(r \mid h, a)
$$

The formulas are rendered by the API docs (``invoke dev.docs``, pdoc with
``--math``).
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Possible results: home win (1), draw (X), away win (2)
OUTCOMES = ["H", "D", "A"]

# Columns the model reads from X
FEATURES = ["home_team", "away_team"]


# scikit-learn ships without type hints, so its base classes are typed as Any
class HeadToHeadModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Predict 1X2 results from past meetings between the same teams at the same ground.

    Args:
        alpha: Smoothing strength, in "virtual matches" drawn from the league
            prior. ``0`` uses raw frequencies (undefined for teams that never met).
    """

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Count past results per fixture and the league priors.

        Args:
            X: Played matches with ``home_team`` and ``away_team`` columns, and
                optionally ``league`` (priors are then computed per league).
            y: Full-time result of each match (``H``, ``D`` or ``A``).

        Returns:
            The fitted model.
        """
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        X = self._validate(X)
        # Align by position, whatever index y comes with
        y = pd.Series(np.asarray(y), index=X.index)
        unknown = set(y.unique()) - set(OUTCOMES)
        if unknown:
            raise ValueError(f"Unknown results {sorted(unknown)}; expected {OUTCOMES}")

        self.classes_ = np.array(OUTCOMES)
        outcomes = pd.get_dummies(y).reindex(columns=OUTCOMES, fill_value=0).astype(float)

        # Result counts per (home, away) fixture
        self.counts_ = outcomes.groupby([X["home_team"], X["away_team"]]).sum()

        # Overall result shares, globally and per league
        self.prior_ = outcomes.mean()
        self.league_priors_ = (
            outcomes.groupby(X["league"]).mean() if "league" in X.columns else pd.DataFrame()
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate the probability of each result.

        Args:
            X: Matches with ``home_team`` and ``away_team`` columns (and
                ``league`` to use the league prior).

        Returns:
            Array of shape ``(n_matches, 3)`` with the probabilities of
            ``classes_`` (``H``, ``D``, ``A``); each row adds up to 1.
        """
        check_is_fitted(self, "counts_")
        X = self._validate(X)

        fixtures = pd.MultiIndex.from_frame(X[FEATURES])
        counts = self.counts_.reindex(fixtures, fill_value=0.0).to_numpy()

        priors = pd.DataFrame([self.prior_] * len(X), columns=OUTCOMES)
        if "league" in X.columns and not self.league_priors_.empty:
            league_priors = self.league_priors_.reindex(X["league"]).reset_index(drop=True)
            priors = league_priors.fillna(priors)
        priors_array = priors.to_numpy()

        n = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs: np.ndarray = (counts + self.alpha * priors_array) / (n + self.alpha)
        # alpha == 0 and no history: fall back to the prior
        no_data = (n + self.alpha).ravel() == 0
        probs[no_data] = priors_array[no_data]
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the most likely result of each match.

        Args:
            X: Matches with ``home_team`` and ``away_team`` columns.

        Returns:
            Array of results (``H``, ``D`` or ``A``). Ties go to the first
            outcome in ``H``, ``D``, ``A`` order.
        """
        # predict_proba first: it raises NotFittedError before classes_ is read
        best = self.predict_proba(X).argmax(axis=1)
        predicted: np.ndarray = self.classes_[best]
        return predicted

    def history_size(self, X: pd.DataFrame) -> np.ndarray:
        """Return how many past meetings back each prediction.

        Args:
            X: Matches with ``home_team`` and ``away_team`` columns.

        Returns:
            Number of past matches between the same teams at the same ground.
        """
        check_is_fitted(self, "counts_")
        fixtures = pd.MultiIndex.from_frame(self._validate(X)[FEATURES])
        sizes: np.ndarray = self.counts_.reindex(fixtures, fill_value=0.0).sum(axis=1).to_numpy()
        return sizes.astype(int)

    @staticmethod
    def _validate(X: pd.DataFrame) -> pd.DataFrame:
        """Check that X is a DataFrame with the required columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with home_team and away_team columns")
        missing = sorted(set(FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"X is missing columns {missing}")
        return X
