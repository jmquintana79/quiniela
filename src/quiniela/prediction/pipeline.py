"""Predict matchdays of the current season and store the predictions.

For each matchday the model is trained only on matches played **before** it
(every historical season plus the current season's earlier matches), so a
prediction never uses information that was unknown at the time.

Predictions are stored in ``data/predictions/<model>.parquet`` with one row
per match, keyed by ``league`` and ``game``. Predicting a single matchday
replaces that matchday's rows and keeps the rest.
"""

import logging
from pathlib import Path

import pandas as pd

from quiniela.config import CURRENT_DIR, HISTORICAL_DIR, PREDICTIONS_DIR
from quiniela.data.loading import load_current, load_historical
from quiniela.data.matchdays import assign_matchdays
from quiniela.prediction.registry import DEFAULT_MODEL, MODELS

logger = logging.getLogger(__name__)

# Columns passed to the models
FEATURE_COLUMNS = ["league", "home_team", "away_team"]

# Columns stored for every prediction
PREDICTION_COLUMNS = [
    "league",
    "game",
    "matchday",
    "date",
    "home_team",
    "away_team",
    "prediction",
    "p_H",
    "p_D",
    "p_A",
]


def current_matchdays(current_dir: Path = CURRENT_DIR) -> pd.DataFrame:
    """Load the current season with its league rounds and quiniela matchdays.

    Args:
        current_dir: Folder with the current season files.

    Returns:
        Current matches (played and upcoming) with ``round`` and ``matchday`` columns.
    """
    matches = load_current(current_dir)
    return matches.join(assign_matchdays(matches))


def predict_matchdays(
    model_key: str = DEFAULT_MODEL,
    matchdays: list[int] | None = None,
    history: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict the given matchdays of the current season.

    Args:
        model_key: Key of the model in ``MODELS``.
        matchdays: Matchday numbers. ``None`` predicts only the last matchday.
        history: Historical matches. Defaults to the downloaded seasons.
        current: Current season with a ``matchday`` column. Defaults to the
            downloaded current season.

    Returns:
        One row per match with ``PREDICTION_COLUMNS``.

    Raises:
        KeyError: If the model is not registered.
        ValueError: If there is no current data or a matchday does not exist.
    """
    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Available: {sorted(MODELS)}")
    history = load_historical(HISTORICAL_DIR) if history is None else history
    current = current_matchdays() if current is None else current
    if current.empty:
        raise ValueError("No current season data. Run `invoke download.current` first.")

    available = sorted(current["matchday"].unique())
    selected = [available[-1]] if matchdays is None else sorted(set(matchdays))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Matchdays {missing} not found. Available: 1-{available[-1]}")

    played = pd.concat([history, current], ignore_index=True).dropna(subset=["FTR"])
    predictions = []
    for matchday in selected:
        target = current[current["matchday"] == matchday]
        # Train only on matches finished before the matchday starts
        cutoff = target["date"].min().normalize()
        train = played[played["date"] < cutoff]

        model = MODELS[model_key].factory()
        model.fit(train[FEATURE_COLUMNS], train["FTR"])
        probs = pd.DataFrame(
            model.predict_proba(target[FEATURE_COLUMNS]),
            columns=[f"p_{c}" for c in model.classes_],
            index=target.index,
        )
        predicted = target.assign(prediction=model.predict(target[FEATURE_COLUMNS])).join(probs)
        predictions.append(predicted[PREDICTION_COLUMNS])
        logger.info(
            "Matchday %s: %d matches predicted (trained on %d matches)",
            matchday,
            len(target),
            len(train),
        )
    return pd.concat(predictions, ignore_index=True)


def save_predictions(
    predictions: pd.DataFrame,
    model_key: str,
    replace_all: bool = False,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> Path:
    """Store predictions, merging them with the ones already saved.

    Args:
        predictions: Output of :func:`predict_matchdays`.
        model_key: Model the predictions come from (file name).
        replace_all: Overwrite the file instead of merging.
        predictions_dir: Folder where prediction files are kept.

    Returns:
        Path of the predictions file.
    """
    path = predictions_dir / f"{model_key}.parquet"
    if path.exists() and not replace_all:
        previous = pd.read_parquet(path)
        # New predictions win for the matches they cover
        keys = pd.MultiIndex.from_frame(predictions[["league", "game"]])
        keep = ~pd.MultiIndex.from_frame(previous[["league", "game"]]).isin(keys)
        predictions = pd.concat([previous[keep], predictions], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.sort_values(["date", "league"]).to_parquet(path, index=False)
    logger.info("Saved %d predictions to %s", len(predictions), path)
    return path


def load_predictions(predictions_dir: Path = PREDICTIONS_DIR) -> dict[str, pd.DataFrame]:
    """Load the stored predictions of every registered model.

    Args:
        predictions_dir: Folder where prediction files are kept.

    Returns:
        Mapping from model key to its predictions (models without a file are omitted).
    """
    return {
        key: pd.read_parquet(predictions_dir / f"{key}.parquet")
        for key in MODELS
        if (predictions_dir / f"{key}.parquet").exists()
    }


def run(
    model_key: str = DEFAULT_MODEL,
    matchday: int | None = None,
    all_matchdays: bool = False,
) -> Path:
    """Predict and save: one matchday (default: the last one) or all of them.

    Args:
        model_key: Key of the model in ``MODELS``.
        matchday: Matchday to predict. Ignored when ``all_matchdays`` is set.
        all_matchdays: Predict every matchday of the current season.

    Returns:
        Path of the predictions file.
    """
    current = current_matchdays()
    if all_matchdays:
        selected: list[int] | None = sorted(current["matchday"].unique())
    else:
        selected = None if matchday is None else [matchday]
    predictions = predict_matchdays(model_key, selected, current=current)
    return save_predictions(predictions, model_key, replace_all=all_matchdays)
