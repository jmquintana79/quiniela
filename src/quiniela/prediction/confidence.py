r"""Confidence of a prediction: how predictable a match is according to a model.

Computed from the 1X2 probabilities of any model, without touching the model.
With the probabilities of a match sorted as $p_{(1)} \ge p_{(2)} \ge p_{(3)}$:

- **Confidence**: the probability of the predicted sign, $p_{(1)}$ for a
  model that picks the most likely one. For a calibrated model it is literally
  the chance of hitting the match with a single sign (65% is hit about 65
  times out of 100).
- **Margin**: $p_{(1)} - p_{(2)}$, how clear the favourite is over the second
  option. A 45/40/15 match has a small margin even if 45% is not low.

The confidence is the measure used to rank matches and to spend doubles and
triples (``quiniela.prediction.planner``); the margin is shown next to it.
"""

import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]

# Columns added to every prediction
CONFIDENCE_COLUMNS = ["confidence", "margin"]

# Confidence tiers shown in the dashboards and the evaluation (lower edges)
TIERS: dict[str, float] = {"high": 0.60, "medium": 0.45, "low": 0.0}


def confidence(probs: np.ndarray, predicted: np.ndarray | pd.Series | None = None) -> pd.DataFrame:
    """Confidence and margin of each prediction.

    Args:
        probs: Probabilities of shape ``(n_matches, 3)`` in ``H``, ``D``, ``A``
            order; rows may be ``NaN`` (no prediction).
        predicted: Predicted signs (``H``, ``D``, ``A``). Defaults to the most
            likely one; pass them when a model uses another decision rule.

    Returns:
        DataFrame with ``confidence`` (probability of the predicted sign) and
        ``margin`` (largest minus second largest probability), ``NaN`` for
        matches without probabilities.
    """
    probs = np.asarray(probs, dtype=float).reshape(-1, len(OUTCOMES))
    available = ~np.isnan(probs).any(axis=1)
    ordered = np.sort(np.where(available[:, None], probs, 0.0), axis=1)[:, ::-1]
    margin = np.where(available, ordered[:, 0] - ordered[:, 1], np.nan)

    if predicted is None:
        chosen = np.where(available, ordered[:, 0], np.nan)
    else:
        signs = np.asarray(predicted, dtype=object)
        chosen = np.full(len(probs), np.nan)
        for k, outcome in enumerate(OUTCOMES):
            picked = available & (signs == outcome)
            chosen[picked] = probs[picked, k]
    return pd.DataFrame({"confidence": chosen, "margin": margin})


def tier(value: float | None) -> str | None:
    """Confidence tier (``high``, ``medium`` or ``low``) of a confidence value.

    Args:
        value: Confidence between 0 and 1, or ``None``/``NaN``.

    Returns:
        The tier name, or ``None`` without a value.
    """
    if value is None or np.isnan(value):
        return None
    return next(name for name, low in TIERS.items() if value >= low)
