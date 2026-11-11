r"""Outcome calibration: correct a model's probabilities before choosing the sign.

Models tend to under-estimate draws, and choosing the most likely outcome then
almost never picks an X. The calibrator is a post-processing step applied to
the probabilities of any model (a final decision layer): it learns how much
each outcome is over- or under-estimated and re-weights the probabilities
before the sign is chosen.

**Calibrated probabilities.** With $p_k$ the model's probability of outcome
$k$ and learned offsets $b_k$ (the home win is the reference, $b_H = 0$):

$$
q_k = \frac{p_k\, e^{b_k}}{\sum_j p_j\, e^{b_j}}, \qquad k \in \lbrace H, D, A \rbrace
$$

The offsets minimise the log-loss $-\frac{1}{N}\sum_i \log q_{i,y_i}$ on
out-of-sample predictions, so a positive $b_D$ means the model gave draws too
little probability.

**Draw rule (the "X rule").** Even with calibrated probabilities a draw is
rarely the single most likely outcome. The sign is therefore chosen with a
draw margin $m \ge 0$:

$$
\hat r =
\begin{cases}
D & \text{if } q_D + m \ge \max(q_H, q_A) \\\\
\arg\max_{k \in \lbrace H, A \rbrace} q_k & \text{otherwise}
\end{cases}
$$

$m$ is learned on the same out-of-sample predictions as the value that
maximises the number of hits (the smallest one on ties), so $m = 0$, plain
"most likely outcome", is kept unless picking more draws actually helps.

Two pieces follow the scikit-learn API:

- ``OutcomeCalibrator``: ``fit(probabilities, results)`` learns the offsets,
  ``predict_proba`` / ``predict`` apply them.
- ``CalibratedModel``: wraps any model and a calibrator into a single
  estimator. To learn from predictions the model has not seen, ``fit`` trains
  the model without its last ``calibration_days`` of data, predicts those
  matches, fits the calibrator on them and finally refits the model on all
  the data.
"""

from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

OUTCOMES = ["H", "D", "A"]

# Smallest probability used inside logarithms
EPSILON = 1e-12


# scikit-learn ships without type hints, so its base classes are typed as Any
class OutcomeCalibrator(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Re-weight 1X2 probabilities with one learned offset per outcome.

    ``X`` is not a match table but the probabilities of a model: an array of
    shape ``(n_matches, 3)`` in ``H``, ``D``, ``A`` order.

    Args:
        l2: Strength of the penalty that keeps the offsets near 0 (useful
            with few calibration matches).
        draw_margin: Margin $m$ of the draw rule. ``"auto"`` learns it,
            a number fixes it (``0`` disables the rule).
        margin_grid: Candidate margins tried when ``draw_margin="auto"``.
    """

    def __init__(
        self,
        l2: float = 0.01,
        draw_margin: float | str = "auto",
        margin_grid: tuple[float, ...] = tuple(np.round(np.arange(0, 0.205, 0.01), 2)),
    ) -> None:
        self.l2 = l2
        self.draw_margin = draw_margin
        self.margin_grid = margin_grid

    def fit(self, X: np.ndarray, y: pd.Series | np.ndarray) -> Self:
        """Learn the offsets that minimise the log-loss of the calibrated probabilities.

        Args:
            X: Model probabilities, shape ``(n_matches, 3)``.
            y: Real results (``H``, ``D``, ``A``).

        Returns:
            The fitted calibrator.
        """
        probs, labels = self._validate(X), np.asarray(y)
        keep = ~np.isnan(probs).any(axis=1)
        probs, labels = probs[keep], labels[keep]
        if len(labels) == 0:
            raise ValueError("No predictions to calibrate")
        observed = np.stack([labels == o for o in OUTCOMES], axis=1).astype(float)
        logits = np.log(np.clip(probs, EPSILON, 1.0))

        def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
            offsets = np.concatenate([[0.0], free])
            q = self._softmax(logits + offsets)
            value = -np.mean((observed * np.log(q)).sum(axis=1)) + self.l2 * free @ free
            grad = np.mean(q - observed, axis=0)[1:] + 2 * self.l2 * free
            return float(value), grad

        result = minimize(objective, np.zeros(len(OUTCOMES) - 1), jac=True, method="BFGS")
        self.classes_ = np.array(OUTCOMES)
        self.offsets_ = np.concatenate([[0.0], result.x])

        if self.draw_margin == "auto":
            calibrated = self._softmax(logits + self.offsets_)
            # Most hits wins; the grid is increasing, so ties keep the smallest margin
            hits = [np.sum(self._decide(calibrated, m) == labels) for m in self.margin_grid]
            self.draw_margin_ = float(self.margin_grid[int(np.argmax(hits))])
        else:
            self.draw_margin_ = float(self.draw_margin)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (rows with missing probabilities stay ``NaN``).

        Args:
            X: Model probabilities, shape ``(n_matches, 3)``.

        Returns:
            Calibrated probabilities, same shape.
        """
        check_is_fitted(self, "offsets_")
        probs = self._validate(X)
        missing = np.isnan(probs).any(axis=1)
        # Placeholder for missing rows (blanked again below) so the softmax stays finite
        logits = np.log(np.clip(np.where(missing[:, None], 1.0, probs), EPSILON, 1.0))
        calibrated = self._softmax(logits + self.offsets_)
        calibrated[missing] = np.nan
        return calibrated

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sign chosen with the draw rule (``None`` where probabilities are missing).

        Args:
            X: Model probabilities, shape ``(n_matches, 3)``.

        Returns:
            Array of results.
        """
        probs = self.predict_proba(X)
        available = ~np.isnan(probs).any(axis=1)
        predicted = np.full(len(probs), None, dtype=object)
        predicted[available] = self._decide(probs[available], self.draw_margin_)
        return predicted

    @staticmethod
    def _decide(probs: np.ndarray, margin: float) -> np.ndarray:
        """Apply the draw rule: X when the draw is within ``margin`` of the favourite."""
        home, draw, away = probs[:, 0], probs[:, 1], probs[:, 2]
        signs = np.where(home >= away, "H", "A").astype(object)
        signs[draw + margin >= np.maximum(home, away)] = "D"
        return signs

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Row-wise softmax, stable for large values."""
        exp = np.exp(logits - np.nanmax(logits, axis=1, keepdims=True))
        normalised: np.ndarray = exp / exp.sum(axis=1, keepdims=True)
        return normalised

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        """Check that X holds three probabilities per match."""
        probs = np.asarray(X, dtype=float)
        if probs.ndim != 2 or probs.shape[1] != len(OUTCOMES):
            raise ValueError(f"X must have shape (n_matches, 3), got {probs.shape}")
        return probs


# scikit-learn ships without type hints, so its base classes are typed as Any
class CalibratedModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """A model followed by an outcome calibrator, as a single estimator.

    Args:
        estimator: Model to calibrate (unfitted; it is cloned).
        calibrator: Calibrator applied to its probabilities. Defaults to
            ``OutcomeCalibrator()``.
        calibration_days: Most recent days of the training data held out to
            fit the calibrator on out-of-sample predictions.
    """

    def __init__(
        self,
        estimator: Any,
        calibrator: Any = None,
        calibration_days: int = 365,
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.calibration_days = calibration_days

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the calibrator on held-out predictions, then the model on all the data.

        Args:
            X: Played matches with a ``date`` column and whatever the model needs.
            y: Full-time results (``H``, ``D``, ``A``).

        Returns:
            The fitted estimator.
        """
        y = pd.Series(np.asarray(y), index=X.index)
        cutoff = X["date"].max() - pd.Timedelta(days=self.calibration_days)
        before, after = X["date"] < cutoff, X["date"] >= cutoff
        if not before.any() or not after.any():
            raise ValueError("Not enough history to hold out calibration matches")

        held_out_model = clone(self.estimator).fit(X[before], y[before])
        held_out_probs = held_out_model.predict_proba(X[after])
        calibrator = OutcomeCalibrator() if self.calibrator is None else self.calibrator
        self.calibrator_ = clone(calibrator).fit(held_out_probs, y[after])

        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.array(OUTCOMES)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities of each result.

        Args:
            X: Matches to predict.

        Returns:
            Array of shape ``(n_matches, 3)`` in ``H``, ``D``, ``A`` order.
        """
        check_is_fitted(self, "estimator_")
        calibrated: np.ndarray = self.calibrator_.predict_proba(self.estimator_.predict_proba(X))
        return calibrated

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely calibrated result of each match.

        Args:
            X: Matches to predict.

        Returns:
            Array of results (``H``, ``D``, ``A``).
        """
        check_is_fitted(self, "estimator_")
        predicted: np.ndarray = self.calibrator_.predict(self.estimator_.predict_proba(X))
        return predicted
