r"""Metrics for 1X2 probability forecasts.

Every metric takes the predicted probabilities as an array of shape
``(n_matches, 3)`` in ``H``, ``D``, ``A`` order (home win, draw, away win) and
the real results as ``H``, ``D`` or ``A``. With $p_{ik}$ the probability given
to outcome $k$ in match $i$ and $o_{ik} = 1$ if outcome $k$ happened (0
otherwise), over $N$ matches:

- **Hit rate** (higher is better): share of matches whose most likely outcome
  happened.
- **Log-loss** (lower is better): $-\frac{1}{N}\sum_i \log p_{i,y_i}$, the
  average surprise; heavily penalises confident mistakes.
- **Brier score** (lower is better, 0 to 2):
  $\frac{1}{N}\sum_i \sum_k (p_{ik} - o_{ik})^2$.
- **Ranked probability score** (lower is better, 0 to 1): compares cumulative
  probabilities, so predicting a draw when the home team wins is "closer" than
  predicting an away win:
  $\frac{1}{N}\sum_i \frac{1}{2}\sum_{k=1}^{2}
  \Big(\sum_{j \le k} p_{ij} - \sum_{j \le k} o_{ij}\Big)^2$.
- **Calibration**: predicted probabilities grouped in bins, compared with how
  often the outcome happened; a calibrated model has both close to each other.
"""

import numpy as np
import pandas as pd

# Outcome order of every probability array
OUTCOMES = ["H", "D", "A"]

# Smallest probability used in the log-loss (avoids log(0))
EPSILON = 1e-15


def _one_hot(y: pd.Series | np.ndarray) -> np.ndarray:
    """Encode results (``H``, ``D``, ``A``) as a ``(n, 3)`` indicator matrix."""
    labels = np.asarray(y)
    return np.stack([labels == outcome for outcome in OUTCOMES], axis=1).astype(float)


def hit_rate(probs: np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Share of matches whose most likely outcome happened."""
    predicted = np.asarray(OUTCOMES)[np.asarray(probs).argmax(axis=1)]
    return float(np.mean(predicted == np.asarray(y)))


def log_loss(probs: np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Average negative log-probability given to the real result."""
    observed = (np.asarray(probs) * _one_hot(y)).sum(axis=1)
    return float(-np.mean(np.log(np.clip(observed, EPSILON, 1.0))))


def brier_score(probs: np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Average squared distance between probabilities and the observed outcome."""
    return float(np.mean(((np.asarray(probs) - _one_hot(y)) ** 2).sum(axis=1)))


def ranked_probability_score(probs: np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Average ranked probability score (outcomes ordered H < D < A)."""
    cumulative = np.cumsum(np.asarray(probs), axis=1)[:, :-1]
    observed = np.cumsum(_one_hot(y), axis=1)[:, :-1]
    return float(np.mean(((cumulative - observed) ** 2).sum(axis=1) / (len(OUTCOMES) - 1)))


def calibration_table(
    probs: np.ndarray, y: pd.Series | np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Compare predicted probabilities with observed frequencies.

    The three outcomes of every match are pooled (one-vs-rest): each
    probability is a forecast "this outcome happens", grouped in equal-width bins.

    Args:
        probs: Predicted probabilities, shape ``(n, 3)``.
        y: Real results.
        n_bins: Number of bins between 0 and 1.

    Returns:
        One row per non-empty bin with ``bin`` (lower edge), ``predicted``
        (mean probability), ``observed`` (share of times the outcome happened)
        and ``count``.
    """
    forecast = np.asarray(probs).ravel()
    happened = _one_hot(y).ravel()
    bins = np.minimum((forecast * n_bins).astype(int), n_bins - 1)
    table = pd.DataFrame({"bin": bins / n_bins, "predicted": forecast, "observed": happened})
    grouped = table.groupby("bin").agg(
        predicted=("predicted", "mean"), observed=("observed", "mean"), count=("observed", "size")
    )
    return grouped.reset_index()


def evaluate(
    probs: np.ndarray,
    y: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute every scalar metric for a set of predictions.

    Args:
        probs: Predicted probabilities, shape ``(n, 3)`` in ``H``, ``D``, ``A`` order.
        y: Real results.
        predicted: Predicted signs. Defaults to the most likely outcome; pass
            them when a model uses another decision rule.

    Returns:
        ``matches``, ``hit_rate``, ``log_loss``, ``brier``, ``rps``, plus
        ``draw_picks`` (share of matches where a draw is the predicted sign)
        and ``draw_rate`` (share of real draws), to check how draws are handled.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(y)
    if len(labels) == 0:
        return {"matches": 0}
    signs = (
        np.asarray(OUTCOMES)[probs.argmax(axis=1)] if predicted is None else np.asarray(predicted)
    )
    return {
        "matches": len(labels),
        "hit_rate": float(np.mean(signs == labels)),
        "log_loss": log_loss(probs, labels),
        "brier": brier_score(probs, labels),
        "rps": ranked_probability_score(probs, labels),
        "draw_picks": float(np.mean(signs == "D")),
        "draw_rate": float(np.mean(labels == "D")),
    }
