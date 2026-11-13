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

The tickets are also played with doubles and triples (``ticket_strategies``):
without them, on random matches, "by eye" on the least predictable matches,
and placed by the planner (``quiniela.prediction.planner``, v4), to measure
how many more prizes each way of spending the same bets wins.
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
from quiniela.prediction import planner
from quiniela.prediction.pipeline import ALL_MODELS, resolve_models, walk_forward
from quiniela.prediction.ticket import (
    PRIZE_CATEGORIES,
    TICKET_SIZE,
    prize_category,
    ticket_positions,
)

logger = logging.getLogger(__name__)

# First season (start year) of the default backtest: 11 seasons, ~9,000 matches,
# every one with bookmaker odds for the market reference
DEFAULT_START = 2015

BACKTEST_FILE = "predictions.parquet"

# Tickets with doubles and triples, derived from the predictions when they are saved
TICKETS_FILE = "tickets.parquet"

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

    The tickets with doubles and triples (:func:`ticket_strategies`) are
    recomputed from the stored predictions and saved next to them, so the
    dashboards do not have to replay them.

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
    tickets = ticket_strategies(backtest)
    tickets.to_parquet(backtest_dir / TICKETS_FILE, index=False)
    logger.info("Saved %d backtest tickets to %s", len(tickets), backtest_dir / TICKETS_FILE)
    return path


def load_backtest(backtest_dir: Path = BACKTEST_DIR) -> pd.DataFrame:
    """Load the stored backtest predictions (empty if the backtest was never run)."""
    path = backtest_dir / BACKTEST_FILE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_tickets(backtest_dir: Path = BACKTEST_DIR) -> pd.DataFrame:
    """Load the stored backtest tickets with doubles and triples (empty if missing)."""
    path = backtest_dir / TICKETS_FILE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=TICKET_COLUMNS)


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


# %% Tickets with doubles and triples

# Ways of playing each matchday's ticket: key -> (label, kind, objective in hits).
# Every strategy but "single" spends the same budget, so the prizes compare
# where the bets go, not how many there are.
TICKET_STRATEGIES: dict[str, tuple[str, str, int]] = {
    "single": ("No doubles or triples (1 bet)", "single", planner.DEFAULT_MIN_HITS),
    "random": ("Doubles and triples on random matches", "random", planner.DEFAULT_MIN_HITS),
    "heuristic": (
        "By eye: on the least predictable matches",
        "heuristic",
        planner.DEFAULT_MIN_HITS,
    ),
    "optimal_14": ("Optimal placement, aiming at 14 hits", "optimal", 14),
    "optimal": ("Optimal placement, aiming at 10+ hits", "optimal", planner.DEFAULT_MIN_HITS),
    "v4": ("v4: best mix within the budget, aiming at 10+ hits", "mix", planner.DEFAULT_MIN_HITS),
}

# Random placements drawn per matchday (their prizes are averaged)
RANDOM_DRAWS = 20

TICKET_COLUMNS = [
    "model",
    "season",
    "matchday",
    "strategy",
    "draw",
    "doubles",
    "triples",
    "bets",
    "hits",
    "category",
    "p_prize",
    "p_top",
    "expected_hits",
    "games",
    "picks",
]

# Separator of the ticket games ("<league>/<game>") and picks stored as one string
LIST_SEPARATOR = "|"


def _ticket_matches(matches: pd.DataFrame) -> pd.DataFrame | None:
    """The 14 ticket matches of a matchday in ticket order, or None without a full ticket."""
    positions = ticket_positions(matches)
    ticket = matches.assign(_position=positions).dropna(subset=["_position"])
    ticket = ticket.sort_values("_position").dropna(subset=["p_H", "p_D", "p_A", "FTR"])
    return ticket if len(ticket) == TICKET_SIZE else None


def _season_tickets(
    predictions: pd.DataFrame, doubles: int, triples: int, max_bets: int, seed: int
) -> list[tuple[object, ...]]:
    """Ticket rows of every strategy for the matchdays of one model and season."""
    rows: list[tuple[object, ...]] = []
    for (model, season, matchday), matches in predictions.groupby(["model", "season", "matchday"]):
        ticket = _ticket_matches(matches)
        if ticket is None:
            continue
        probs = ticket[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
        results = ticket["FTR"].tolist()
        # Which matches the ticket had and what each strategy marked, so per-match
        # hit rates (e.g. v4 in the evaluation dashboard) can be computed later
        games = LIST_SEPARATOR.join(ticket["league"] + "/" + ticket["game"])
        # One generator per matchday, so results do not depend on the job split
        rng = np.random.default_rng([seed, int(matchday), int(str(season)[:4])])
        for key, (_, kind, min_hits) in TICKET_STRATEGIES.items():
            if kind == "single":
                plans = [planner.single_plan(probs, min_hits)]
            elif kind == "random":
                plans = [
                    planner.random_plan(probs, rng, doubles, triples, min_hits)
                    for _ in range(RANDOM_DRAWS)
                ]
            elif kind == "heuristic":
                plans = [planner.heuristic_plan(probs, doubles, triples, min_hits)]
            elif kind == "optimal":
                plans = [planner.plan_ticket(probs, doubles, triples, min_hits=min_hits)]
            else:
                plans = [planner.plan_ticket(probs, max_bets=max_bets, min_hits=min_hits)]
            for draw, plan in enumerate(plans):
                hits = plan.hits(results)
                rows.append(
                    (
                        model,
                        season,
                        matchday,
                        key,
                        draw,
                        plan.doubles,
                        plan.triples,
                        plan.bets,
                        hits,
                        prize_category(hits),
                        plan.at_least(planner.DEFAULT_MIN_HITS),
                        plan.at_least(TICKET_SIZE),
                        float(plan.coverage.sum()),
                        games,
                        LIST_SEPARATOR.join(plan.picks),
                    )
                )
    return rows


def ticket_strategies(
    backtest: pd.DataFrame,
    model: str = planner.PLANNER_MODEL,
    doubles: int = planner.DEFAULT_DOUBLES,
    triples: int = planner.DEFAULT_TRIPLES,
    max_bets: int = planner.DEFAULT_MAX_BETS,
    seed: int = 0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Play every full ticket of the backtest with each strategy of ``TICKET_STRATEGIES``.

    Args:
        backtest: Backtest predictions with real results.
        model: Model whose probabilities choose the signs (v3 by default).
        doubles: Doubles of the random, by-eye and optimal strategies.
        triples: Triples of the random, by-eye and optimal strategies.
        max_bets: Budget of the v4 strategy, which also chooses the mix.
        seed: Seed of the random placements.
        n_jobs: Parallel jobs (one per season).

    Returns:
        One row per strategy, matchday and draw (``RANDOM_DRAWS`` for the
        random strategy, one for the rest) with ``TICKET_COLUMNS``:
        ``p_prize`` and ``p_top`` are the planned probabilities of 10+ and 14
        hits, ``hits`` and ``category`` what happened, ``games`` the ticket
        matches (``<league>/<game>``) and ``picks`` the signs marked in each
        (``H``, ``HD``, ``HDA``...), both joined with ``LIST_SEPARATOR``.
    """
    predictions = backtest[backtest["model"] == model]
    if predictions.empty:
        return pd.DataFrame(columns=TICKET_COLUMNS)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_season_tickets)(group, doubles, triples, max_bets, seed)
        for _, group in predictions.groupby("season")
    )
    rows = [row for season_rows in results for row in season_rows]
    return pd.DataFrame(rows, columns=TICKET_COLUMNS)


def summarize_tickets(tickets: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Prizes won by each ticket strategy.

    Random placements are averaged over their draws, so their counts can be
    fractional.

    Args:
        tickets: Output of :func:`ticket_strategies`.
        by: Extra grouping columns besides ``strategy`` (e.g. ``"season"``).

    Returns:
        One row per strategy (in ``TICKET_STRATEGIES`` order) with ``label``,
        ``bets`` (average cost), ``tickets`` (matchdays), ``prizes`` (10+
        hits), one column per category (``14`` ... ``10``), ``mean_hits``,
        ``planned_prize`` (average planned probability of 10+ hits) and
        ``prize_rate`` (share of tickets with a prize).
    """
    groups = ["strategy", *(by or [])]
    rows = []
    for key, group in tickets.groupby(groups, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(groups, values, strict=True))
        draws = group["draw"].nunique()
        row["label"] = TICKET_STRATEGIES.get(str(values[0]), (str(values[0]),))[0]
        row["bets"] = float(group["bets"].mean())
        row["tickets"] = int(len(group) // draws)
        prizes = group["hits"] >= planner.DEFAULT_MIN_HITS
        row["prizes"] = float(prizes.sum() / draws)
        for hits in PRIZE_CATEGORIES:
            row[str(hits)] = float((group["hits"] == hits).sum() / draws)
        row["mean_hits"] = float(group["hits"].mean())
        row["planned_prize"] = float(group["p_prize"].mean())
        row["prize_rate"] = float(prizes.mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = {key: i for i, key in enumerate(TICKET_STRATEGIES)}
    return summary.sort_values(
        [*groups[1:], "strategy"], key=lambda c: c.map(order) if c.name == "strategy" else c
    ).reset_index(drop=True)


def ticket_picks(tickets: pd.DataFrame, strategy: str = "v4") -> pd.DataFrame:
    """Signs marked in every ticket match by one strategy (first draw).

    Args:
        tickets: Output of :func:`ticket_strategies`.
        strategy: Key of ``TICKET_STRATEGIES``.

    Returns:
        One row per ticket match with ``season``, ``matchday``, ``league``,
        ``game`` and ``picks``; empty if the tickets have no picks stored.
    """
    columns = ["season", "matchday", "league", "game", "picks"]
    if "picks" not in tickets.columns:
        return pd.DataFrame(columns=columns)
    chosen = tickets[(tickets["strategy"] == strategy) & (tickets["draw"] == 0)]
    rows = []
    for season, matchday, games, picks in zip(
        chosen["season"], chosen["matchday"], chosen["games"], chosen["picks"], strict=True
    ):
        for game, pick in zip(
            games.split(LIST_SEPARATOR), picks.split(LIST_SEPARATOR), strict=True
        ):
            league, key = game.split("/", 1)
            rows.append((season, matchday, league, key, pick))
    return pd.DataFrame(rows, columns=columns)
