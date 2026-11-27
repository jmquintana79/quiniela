"""Evaluation of the Liga F (women's league) models, apart from the men's backtest.

Two sources, two kinds of model:

- **Every Liga F match** since 2022-23 (ESPN, ``quiniela.io.espn``): the
  models that learn from results (v2F, pi-ratings) predict every week
  walk-forward, trained on the matches played before it; the first season
  only trains. The reference, the **result shares** of the earlier matches
  (always the most frequent result), is scored the same way.
- **The Liga F matches of the official tickets** with the public's picks
  archived (``quiniela.data.women``): v1F (public picks) is only scored there.

Every model is scored on the matches where it has its input (``all``), and
all of them together on the matches where every one has it (``common``:
today the ticket matches with public picks), the fair comparison of v1F and
v2F.
"""

from typing import Any

import numpy as np
import pandas as pd

from quiniela.evaluation.metrics import contingency_table, evaluate
from quiniela.prediction.public_picks import PUBLIC_COLUMNS, PublicPicksModel
from quiniela.prediction.registry import WOMEN_MODELS

OUTCOMES = ["H", "D", "A"]

# Keys of the women's models
PUBLIC_KEY = "public_picks"
PI_KEY = "pi_ratings_f"
SHOTS_KEY = "pi_ratings_shots_f"
# Liga F models learned from the ESPN results (v2F, v3F...): every one is backtested weekly
RESULT_KEYS = [key for key in WOMEN_MODELS if key != PUBLIC_KEY]

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

PREDICTION_COLUMNS = [
    "season",
    "game",
    "date",
    "home_team",
    "away_team",
    "FTR",
    "model",
    "p_H",
    "p_D",
    "p_A",
    "prediction",
    "has_input",
]


def _rows(matches: pd.DataFrame, key: str, probs: np.ndarray, has_input: bool) -> pd.DataFrame:
    """Prediction rows of one model for some matches."""
    return matches[["season", "game", "date", "home_team", "away_team", "FTR"]].assign(
        model=key,
        p_H=probs[:, 0],
        p_D=probs[:, 1],
        p_A=probs[:, 2],
        prediction=np.array(OUTCOMES)[probs.argmax(axis=1)],
        has_input=has_input,
    )


def results_backtest(espn: pd.DataFrame) -> pd.DataFrame:
    """Weekly walk-forward predictions of the result-based models on every played Liga F match.

    Args:
        espn: ESPN Liga F matches (``quiniela.io.espn.load_liga_f_matches``).

    Returns:
        ``PREDICTION_COLUMNS`` for every model of ``RESULT_KEYS`` (v2F, v3F...)
        and the result-shares reference, from the second season on (the first
        one only trains).
    """
    played = espn.dropna(subset=["FTR"]).sort_values("date").reset_index(drop=True)
    if played.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    first = played["season"].min()
    week = played["date"].dt.to_period("W-SUN").dt.start_time
    frames = []
    for start, matches in played[played["season"] > first].groupby(week[played["season"] > first]):
        train = played[played["date"] < start]
        if train.empty:
            continue
        for key in RESULT_KEYS:
            model = WOMEN_MODELS[key].factory().fit(train, train["FTR"])
            frames.append(_rows(matches, key, model.predict_proba(matches), True))
        shares = PublicPicksModel().fit(train, train["FTR"])
        frames.append(_rows(matches, SHARES_KEY, shares.predict_proba(matches), True))
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]


def women_predictions(history: pd.DataFrame, espn: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every women's model on the played Liga F matches.

    Args:
        history: Liga F ticket matches (``quiniela.data.women.liga_f_history``).
        espn: ESPN Liga F matches; without them only v1F and the reference
            (on the ticket matches) are scored.

    Returns:
        ``PREDICTION_COLUMNS``; ``game`` is the ESPN match (or
        ``<season>-<jornada>-<position>`` for a ticket match not in ESPN).
    """
    frames = []
    backtest = results_backtest(espn) if espn is not None and not espn.empty else None
    if backtest is not None:
        frames.append(backtest)
    played = history.dropna(subset=["FTR"]).sort_values("date").reset_index(drop=True)
    if not played.empty:
        key = (
            played["espn_game"]
            if "espn_game" in played.columns
            else pd.Series(None, index=played.index)
        )
        fallback = (
            played["season"]
            + "-"
            + played["jornada"].astype(str)
            + "-"
            + played["position"].astype(str)
        )
        tickets = played.assign(
            game=key.fillna(fallback),
            home_team=(
                played["home_team"] if "home_team" in played.columns else played["home"]
            ).fillna(played["home"]),
            away_team=(
                played["away_team"] if "away_team" in played.columns else played["away"]
            ).fillna(played["away"]),
        )
        for i, match in tickets.iterrows():
            earlier = tickets[tickets["date"] < match["date"]]
            model = PublicPicksModel().fit(earlier, earlier["FTR"])
            row = tickets.loc[[i]]
            has_picks = bool(model.has_picks(row)[0])
            frames.append(_rows(row, PUBLIC_KEY, model.predict_proba(row), has_picks))
            if backtest is None:
                inputs = row.drop(columns=PUBLIC_COLUMNS)
                frames.append(_rows(row, SHARES_KEY, model.predict_proba(inputs), True))
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]


def _scores(predictions: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Metrics and contingency tables by season (``"all"``) and model."""
    results: dict[str, dict[str, Any]] = {}
    contingency: dict[str, dict[str, list[list[int]]]] = {}
    for season, group in [("all", predictions), *predictions.groupby("season")]:
        for model, rows in group.groupby("model"):
            probs = rows[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
            results.setdefault(str(season), {})[str(model)] = evaluate(
                probs, rows["FTR"], rows["prediction"]
            )
            contingency.setdefault(str(season), {})[str(model)] = contingency_table(
                rows["prediction"], rows["FTR"]
            )
    return results, contingency


def summarize_women(predictions: pd.DataFrame) -> dict[str, Any]:
    """Metrics of the Liga F models, each on its own matches and all on the common ones.

    Args:
        predictions: Output of :func:`women_predictions`.

    Returns:
        ``results`` and ``contingency`` (each model on the matches with its
        input) and ``common`` / ``common_contingency`` (every model on the
        matches where all of them have their input), keyed by season
        (``"all"``) and model.
    """
    scored = predictions[predictions["has_input"]]
    results, contingency = _scores(scored)
    # Every women's model and the reference must have scored the match
    expected = {*WOMEN_MODELS, SHARES_KEY}
    per_game = scored.groupby("game")["model"].agg(set)
    common_games = per_game[per_game.apply(lambda models: expected <= models)].index
    common, common_contingency = _scores(scored[scored["game"].isin(common_games)])
    return {
        "results": results,
        "contingency": contingency,
        "common": common,
        "common_contingency": common_contingency,
    }


def upcoming_predictions(
    espn: pd.DataFrame, before: pd.Timestamp | None = None, key: str = PI_KEY
) -> pd.DataFrame:
    """Predictions of the Liga F matches from a date on, trained on the matches before it.

    Args:
        espn: ESPN Liga F matches.
        before: Only matches played before it train the model, and the
            matches from it on are predicted (e.g. a ticket's sales deadline:
            its matches keep their pre-match prediction once played).
            Defaults to the first match not played yet.
        key: Results-based model (v2F by default; v3F, the tickets' one...).

    Returns:
        The predicted matches with ``p_H``, ``p_D``, ``p_A`` and ``prediction``.
    """
    pending = espn[espn["FTR"].isna()]
    if before is None:
        before = pending["date"].min() if not pending.empty else espn["date"].max()
    train = espn[espn["FTR"].notna() & (espn["date"] < before)]
    target = espn[espn["date"] >= before]
    if train.empty or target.empty:
        return target.assign(p_H=np.nan, p_D=np.nan, p_A=np.nan, prediction=None)
    model = WOMEN_MODELS[key].factory().fit(train, train["FTR"])
    probs = model.predict_proba(target)
    return target.assign(
        p_H=probs[:, 0],
        p_D=probs[:, 1],
        p_A=probs[:, 2],
        prediction=np.array(OUTCOMES)[probs.argmax(axis=1)],
    )
