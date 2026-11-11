"""Walk-forward predictions of matchdays, for the current season and the backtest.

For each matchday a model is trained only on matches played **before** it
(every earlier season plus the earlier matches of the same season), so a
prediction never uses information that was unknown at the time. The same
:func:`walk_forward` routine predicts the current season and past seasons
(``quiniela.evaluation.backtest``).

Models receive the whole match table as ``X``: league, date, teams, odds and,
for the training matches, goals. The goals and result of the matches being
predicted are removed before calling ``predict``.

Current-season predictions are stored in ``data/predictions/<model>.parquet``
with one row per match, keyed by ``league`` and ``game``. Predicting a single
matchday replaces that matchday's rows and keeps the rest.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quiniela.config import CURRENT_DIR, HISTORICAL_DIR, PREDICTIONS_DIR
from quiniela.data.loading import load_current, load_historical
from quiniela.data.matchdays import assign_matchdays
from quiniela.prediction.registry import MODELS

logger = logging.getLogger(__name__)

# Columns of the match being predicted that a model must not see
RESULT_COLUMNS = ["FTHG", "FTAG", "FTR"]

# Exact score and Pleno al 15 picks (only for models with ``predict_score``)
SCORE_COLUMNS = ["score_home", "score_away", "pleno_home", "pleno_away"]

# Columns stored for every prediction
PREDICTION_COLUMNS = [
    "league",
    "season",
    "game",
    "matchday",
    "date",
    "home_team",
    "away_team",
    "prediction",
    "p_H",
    "p_D",
    "p_A",
    *SCORE_COLUMNS,
]

# Value accepted wherever a model key is expected, meaning every registered model
ALL_MODELS = "all"


def resolve_models(model_key: str) -> list[str]:
    """Expand a model key (or ``"all"``) into registered model keys.

    Args:
        model_key: A key of ``MODELS`` or ``"all"``.

    Returns:
        The selected keys, in registry order.

    Raises:
        KeyError: If the key is not registered.
    """
    if model_key == ALL_MODELS:
        return list(MODELS)
    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Available: {sorted(MODELS)} or 'all'")
    return [model_key]


def current_matchdays(current_dir: Path = CURRENT_DIR) -> pd.DataFrame:
    """Load the current season with its league rounds and quiniela matchdays.

    Args:
        current_dir: Folder with the current season files.

    Returns:
        Current matches (played and upcoming) with ``round`` and ``matchday`` columns.
    """
    matches = load_current(current_dir)
    return matches.join(assign_matchdays(matches))


def walk_forward(model_key: str, target: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """Predict the matchdays in ``target`` one by one, training on earlier matches.

    Args:
        model_key: Key of the model in ``MODELS``.
        target: Matches to predict, with a ``matchday`` column.
        played: Pool of played matches to train on; only the ones before each
            matchday's first day are used.

    Returns:
        One row per target match with ``PREDICTION_COLUMNS``. Matchdays
        without any earlier match to train on get no prediction (empty values).
    """
    spec = MODELS[model_key]
    predictions = []
    for matchday, matches in target.groupby("matchday", sort=True):
        # Train only on matches finished before the matchday starts
        cutoff = matches["date"].min().normalize()
        train = played[played["date"] < cutoff]
        out = matches.assign(prediction=None).reindex(
            columns=list(dict.fromkeys([*matches.columns, *PREDICTION_COLUMNS]))
        )
        if train.empty:
            predictions.append(out[PREDICTION_COLUMNS])
            continue

        model = spec.factory()
        model.fit(train, train["FTR"])
        X = matches.drop(columns=RESULT_COLUMNS, errors="ignore")
        probs = np.asarray(model.predict_proba(X), dtype=float)
        out[["p_H", "p_D", "p_A"]] = probs
        out["prediction"] = model.predict(X)
        if hasattr(model, "predict_score"):
            out[SCORE_COLUMNS] = model.predict_score(X)[SCORE_COLUMNS].to_numpy()
        predictions.append(out[PREDICTION_COLUMNS])
        logger.debug(
            "%s matchday %s: %d matches predicted (trained on %d matches)",
            model_key,
            matchday,
            len(matches),
            len(train),
        )
    if not predictions:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(predictions, ignore_index=True)


def predict_matchdays(
    model_key: str,
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
    resolve_models(model_key)
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
    target = current[current["matchday"].isin(selected)]
    predictions = walk_forward(model_key, target, played)
    logger.info("%s: %d matches predicted (matchdays %s)", model_key, len(predictions), selected)
    return predictions


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
    model_key: str = ALL_MODELS,
    matchday: int | None = None,
    all_matchdays: bool = False,
) -> list[Path]:
    """Predict and save: one matchday (default: the last one) or all of them.

    Args:
        model_key: Key of the model in ``MODELS``, or ``"all"`` for every model.
        matchday: Matchday to predict. Ignored when ``all_matchdays`` is set.
        all_matchdays: Predict every matchday of the current season.

    Returns:
        Paths of the predictions files, one per model.
    """
    current = current_matchdays()
    history = load_historical(HISTORICAL_DIR)
    if all_matchdays:
        selected: list[int] | None = sorted(current["matchday"].unique())
    else:
        selected = None if matchday is None else [matchday]
    paths = []
    for key in resolve_models(model_key):
        predictions = predict_matchdays(key, selected, history=history, current=current)
        paths.append(save_predictions(predictions, key, replace_all=all_matchdays))
    return paths
