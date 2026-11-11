"""Backtest: walk-forward predictions of every model over past seasons.

Each completed season is split into quiniela matchdays and every registered
model predicts them one by one, trained only on matches played before each
matchday (``quiniela.prediction.pipeline.walk_forward``). This is exactly how
the models are used during a season, repeated over thousands of matches so
that differences between models can be told apart from luck.

The predictions of all models are stored in ``data/backtest/predictions.parquet``
(one row per model and match, with the real result) and summarised with the
metrics of ``quiniela.evaluation.metrics``, overall, per season and per
league, plus the quiniela prizes the 14-match tickets would have won.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from quiniela.config import BACKTEST_DIR, HISTORICAL_DIR
from quiniela.data.loading import load_historical
from quiniela.data.matchdays import assign_matchdays
from quiniela.evaluation.metrics import evaluate
from quiniela.prediction.pipeline import ALL_MODELS, resolve_models, walk_forward
from quiniela.prediction.ticket import prize_category, ticket_positions

logger = logging.getLogger(__name__)

# First season (start year) of the default backtest: 11 seasons, ~9,000 matches,
# every one with bookmaker odds for the market reference
DEFAULT_START = 2015

BACKTEST_FILE = "predictions.parquet"

# Real result columns added to every prediction
RESULT_COLUMNS = ["FTHG", "FTAG", "FTR"]

# Pleno al 15 goal categories: 0, 1, 2 and M (3 or more)
PLENO_LABELS = np.array(["0", "1", "2", "M"])


def season_label(start_year: int) -> str:
    """Format a season start year as a label (``2015`` -> ``"2015-16"``)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _season_matches(history: pd.DataFrame, label: str) -> pd.DataFrame:
    """Matches of one season with their league rounds and quiniela matchdays."""
    matches = history[history["season"] == label]
    return matches.join(assign_matchdays(matches))


def run_backtest(
    start: int = DEFAULT_START,
    end: int | None = None,
    model_key: str = ALL_MODELS,
    history: pd.DataFrame | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Predict every matchday of the selected seasons with the selected models.

    Args:
        start: Start year of the first season (2015 = 2015-16).
        end: Start year of the last season (inclusive). Defaults to the last
            downloaded season.
        model_key: Key of a model in ``MODELS`` or ``"all"``.
        history: Historical matches. Defaults to the downloaded seasons.
        n_jobs: Parallel jobs (``-1`` uses every CPU core); one job per model
            and season.

    Returns:
        One row per model and match: ``model``, the prediction columns and
        the real result (``FTHG``, ``FTAG``, ``FTR``).

    Raises:
        ValueError: If no downloaded season falls in the range.
    """
    history = load_historical(HISTORICAL_DIR) if history is None else history
    available = sorted(history["season"].unique())
    last = int(available[-1][:4]) if end is None else end
    labels = [season_label(year) for year in range(start, last + 1)]
    labels = [label for label in labels if label in available]
    if not labels:
        raise ValueError(f"No downloaded seasons between {start} and {last}")
    keys = resolve_models(model_key)

    played = history.dropna(subset=["FTR"])
    seasons = {label: _season_matches(history, label) for label in labels}
    logger.info(
        "Backtest of %s over %d seasons (%s to %s)", keys, len(labels), labels[0], labels[-1]
    )
    jobs = [(key, label) for key in keys for label in labels]
    results = Parallel(n_jobs=n_jobs)(
        delayed(walk_forward)(key, seasons[label], played) for key, label in jobs
    )

    frames = []
    for (key, label), predictions in zip(jobs, results, strict=True):
        actual = seasons[label][["league", "game", *RESULT_COLUMNS]]
        frames.append(predictions.assign(model=key).merge(actual, on=["league", "game"]))
    return pd.concat(frames, ignore_index=True)


def save_backtest(backtest: pd.DataFrame, backtest_dir: Path = BACKTEST_DIR) -> Path:
    """Store backtest predictions, replacing the models and seasons they cover.

    Args:
        backtest: Output of :func:`run_backtest`.
        backtest_dir: Folder of the backtest file.

    Returns:
        Path of the backtest file.
    """
    path = backtest_dir / BACKTEST_FILE
    if path.exists():
        previous = pd.read_parquet(path)
        covered = pd.MultiIndex.from_frame(backtest[["model", "season"]].drop_duplicates())
        keep = ~pd.MultiIndex.from_frame(previous[["model", "season"]]).isin(covered)
        backtest = pd.concat([previous[keep], backtest], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    backtest.sort_values(["model", "date", "league"]).to_parquet(path, index=False)
    logger.info("Saved %d backtest predictions to %s", len(backtest), path)
    return path


def load_backtest(backtest_dir: Path = BACKTEST_DIR) -> pd.DataFrame:
    """Load the stored backtest predictions (empty if the backtest was never run)."""
    path = backtest_dir / BACKTEST_FILE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _pleno(goals: pd.Series) -> np.ndarray:
    """Goals as Pleno al 15 categories (0, 1, 2, M)."""
    labels: np.ndarray = PLENO_LABELS[np.minimum(goals.to_numpy(dtype=int), 3)]
    return labels


def ticket_results(backtest: pd.DataFrame) -> pd.DataFrame:
    """Hits of every model's 14-match quiniela ticket in each matchday.

    Args:
        backtest: Backtest predictions with real results.

    Returns:
        One row per model, season and matchday with full tickets: ``hits``
        and ``category`` (prize won, or ``None``).
    """
    rows = []
    for (model, season, matchday), matches in backtest.groupby(["model", "season", "matchday"]):
        positions = ticket_positions(matches)
        ticket = matches[positions.notna()]
        if len(ticket) < 14:
            continue
        hits = int((ticket["prediction"] == ticket["FTR"]).sum())
        rows.append((model, season, matchday, hits, prize_category(hits)))
    return pd.DataFrame(rows, columns=["model", "season", "matchday", "hits", "category"])


def summarize(backtest: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics of every model, optionally broken down by season and/or league.

    Matches a model could not predict (e.g. the market without odds) are
    counted in ``missing`` and left out of the other metrics.

    Args:
        backtest: Backtest predictions with real results.
        by: Extra grouping columns besides ``model`` (``"season"``, ``"league"``).

    Returns:
        One row per group with ``matches``, ``missing``, ``hit_rate``,
        ``log_loss``, ``brier``, ``rps``, ``draw_picks``, ``draw_rate``,
        ``score_hit_rate`` and ``pleno_hit_rate`` (models with score
        predictions), plus ``tickets`` and ``prizes`` when not grouped by league.
    """
    groups = ["model", *(by or [])]
    rows = []
    for key, matches in backtest.groupby(groups, sort=False):
        predicted = matches.dropna(subset=["p_H", "p_D", "p_A"])
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,), strict=True))
        row["missing"] = len(matches) - len(predicted)
        row.update(
            evaluate(
                predicted[["p_H", "p_D", "p_A"]].to_numpy(),
                predicted["FTR"],
                predicted["prediction"],
            )
        )
        scored = predicted.dropna(subset=["score_home"])
        if not scored.empty:
            row["score_hit_rate"] = float(
                (
                    (scored["score_home"].astype(int) == scored["FTHG"])
                    & (scored["score_away"].astype(int) == scored["FTAG"])
                ).mean()
            )
            row["pleno_hit_rate"] = float(
                (
                    (scored["pleno_home"].to_numpy() == _pleno(scored["FTHG"]))
                    & (scored["pleno_away"].to_numpy() == _pleno(scored["FTAG"]))
                ).mean()
            )
        rows.append(row)
    summary = pd.DataFrame(rows)

    if "league" not in groups and not summary.empty:
        # Tickets mix both leagues, so prizes only exist per model and season
        tickets = ticket_results(backtest)
        if not tickets.empty:
            counts = tickets.groupby(groups).agg(
                tickets=("hits", "size"), prizes=("category", lambda c: int(c.notna().sum()))
            )
            summary = summary.merge(counts.reset_index(), on=groups, how="left")
    return summary
