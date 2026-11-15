"""Evaluation of the Liga F (women's league) models, apart from the men's backtest.

The Liga F history (``quiniela.data.women``) holds the women's matches of the
official tickets with their results and, from the tickets whose betting page
was saved, the public's picks. Every registered women's model
(``quiniela.prediction.registry.WOMEN_MODELS``) predicts each played match
walk-forward (trained on the Liga F matches played before it) and is scored
only on the matches where it has its input: v1F (public picks) on the matches
with public picks. A reference, the result shares of the earlier Liga F
matches (what v1F falls back to without picks), is scored on every played
match, so the tables are never empty while the public-picks history grows.
"""

from typing import Any

import numpy as np
import pandas as pd

from quiniela.evaluation.metrics import contingency_table, evaluate
from quiniela.prediction.public_picks import PUBLIC_COLUMNS, PublicPicksModel
from quiniela.prediction.registry import WOMEN_MODELS

OUTCOMES = ["H", "D", "A"]

# Reference of the Liga F evaluation: shares of 1, X and 2 of the earlier matches
SHARES_KEY = "liga_f_shares"
SHARES_SPEC: dict[str, Any] = {
    "key": SHARES_KEY,
    "name": "Result shares",
    "version": "",
    "reference": True,
    "description": "Shares of 1, X and 2 of the Liga F matches played before (what v1F uses "
    "without public picks); its sign is always the most frequent result.",
}


def women_predictions(history: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward predictions of every women's model on the played Liga F matches.

    Args:
        history: Output of ``quiniela.data.women.liga_f_history``.

    Returns:
        One row per played match and model with ``season``, ``jornada``,
        ``position``, ``home``, ``away``, ``FTR``, ``model``, ``p_H``, ``p_D``,
        ``p_A``, ``prediction`` and ``has_input`` (the model had its data).
    """
    played = history.dropna(subset=["FTR"]).sort_values("date").reset_index(drop=True)
    rows = []
    for i, match in played.iterrows():
        earlier = played.iloc[:i]
        # Matches of the same day are not known before this one
        earlier = earlier[earlier["date"] < match["date"]]
        X = played.iloc[[i]]
        models: dict[str, tuple[Any, bool]] = {}
        for key, spec in WOMEN_MODELS.items():
            model = spec.factory().fit(earlier, earlier["FTR"])
            has_input = bool(model.has_picks(X)[0]) if hasattr(model, "has_picks") else True
            models[key] = (model, has_input)
        reference = PublicPicksModel().fit(earlier, earlier["FTR"])
        models[SHARES_KEY] = (reference, True)
        for key, (model, has_input) in models.items():
            inputs = X.drop(columns=PUBLIC_COLUMNS) if key == SHARES_KEY else X
            probs = np.asarray(model.predict_proba(inputs), dtype=float)[0]
            rows.append(
                {
                    **match[["season", "jornada", "position", "home", "away", "FTR"]].to_dict(),
                    "model": key,
                    "p_H": probs[0],
                    "p_D": probs[1],
                    "p_A": probs[2],
                    "prediction": OUTCOMES[int(probs.argmax())],
                    "has_input": has_input,
                }
            )
    columns = ["season", "jornada", "position", "home", "away", "FTR", "model"]
    columns += ["p_H", "p_D", "p_A", "prediction", "has_input"]
    return pd.DataFrame(rows, columns=columns)


def summarize_women(predictions: pd.DataFrame) -> dict[str, Any]:
    """Metrics and contingency tables of the Liga F models, overall and by season.

    Args:
        predictions: Output of :func:`women_predictions`.

    Returns:
        ``results`` and ``contingency`` keyed by season (``"all"`` for every
        one) and model key; each model only on the matches with its input.
    """
    results: dict[str, dict[str, Any]] = {}
    contingency: dict[str, dict[str, list[list[int]]]] = {}
    scored = predictions[predictions["has_input"]]
    for season, group in [("all", scored), *scored.groupby("season")]:
        for model, rows in group.groupby("model"):
            metrics = evaluate(
                rows[["p_H", "p_D", "p_A"]].to_numpy(), rows["FTR"], rows["prediction"]
            )
            results.setdefault(str(season), {})[str(model)] = metrics
            contingency.setdefault(str(season), {})[str(model)] = contingency_table(
                rows["prediction"], rows["FTR"]
            )
    return {"results": results, "contingency": contingency}
