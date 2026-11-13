"""Backtest in euros: the real quiniela tickets of past seasons, played with real prizes.

For every past draw whose ticket has at least ``MIN_MATCHED`` La Liga /
Segunda matches (``quiniela.data.prizes``), the ticket is played with:

- ``single``: one column with the v3 signs (0.75 EUR, a reference: the
  minimum ticket is two columns);
- ``v4``: doubles and triples for the highest chance of 11+ hits within 48 bets;
- ``v5``: the prudent money planner (only 10 to 13-hit prizes count), only
  when its expected net prize is positive;
- ``v5_full``: the money planner counting every prize (14 hits and the Pleno
  al 15 too), only when its expected net prize is positive;
- ``v5_always``: the full planner's ticket every draw, to measure what
  choosing the draws adds;
- ``v5_big``: the big-win mode, every draw: at most 50 EUR, the ticket with
  the best chance of a prize of 10,000 EUR or more.

The other ticket matches (other competitions, women's league) get the league
shares of 1, X and 2 as probabilities and the home sign. Payouts use the real
winners and takings of the draw, sharing each pool with our winning columns
(``quiniela.prediction.money.ticket_payout``). Every input is known before
the draw: the v3 walk-forward probabilities, the market odds, a prize model
fitted on earlier draws only, the expected bets (average of recent draws) and
the jackpot announced for that draw.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from quiniela.config import BACKTEST_DIR, CURRENT_DIR, HISTORICAL_DIR, LAE_DIR
from quiniela.data.prizes import BET_PRICE, SPECIAL_SHARE, draw_coverage, match_tickets
from quiniela.prediction import money, planner
from quiniela.prediction.market import MarketModel
from quiniela.prediction.ticket import TICKET_SIZE

logger = logging.getLogger(__name__)

OUTCOMES = ["H", "D", "A"]

# Smallest number of the 14 ticket matches that must be La Liga / Segunda
MIN_MATCHED = 10

# Draws whose bets are averaged to expect the bets of the next one
VOLUME_WINDOW = 5

MONEY_STRATEGIES: dict[str, str] = {
    "single": "One column with the v3 signs (0.75 EUR)",
    "v4": "v4: best chance of 11+ hits within 48 bets, every draw",
    "v5": "v5 (prudent): only 10-13 hit prizes count, plays only when they pay",
    "v5_full": "v5 counting 14 hits and the Pleno al 15 too, only when it pays",
    "v5_always": "The full v5 ticket every draw, worth it or not",
    "v5_big": "v5 big win: best chance of 10,000+ EUR within 50 EUR, every draw",
}

MONEY_COLUMNS = [
    "season",
    "jornada",
    "date",
    "draw_id",
    "matched",
    "strategy",
    "played",
    "doubles",
    "triples",
    "bets",
    "cost",
    "payout",
    "net",
    "best_hits",
    "expected_net",
    "expected_top",
    "payout_top",
    "p_big",
    "jackpot",
]

PLENO_LABELS = np.array(["0", "1", "2", "M"])


def draw_surprises(
    draws: pd.DataFrame, matched: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """Surprise of the real results of every draw, from the market probabilities.

    Args:
        draws: Output of ``quiniela.data.prizes.load_draws``.
        matched: Output of ``quiniela.data.prizes.match_tickets``.
        matches: football-data matches with odds (``league``, ``game``, ``FTR``).

    Returns:
        ``draws`` with ``matched`` (ticket matches found) and ``surprise``
        (``NaN`` when a ticket result is missing).
    """
    crowd = _match_probabilities(matches)
    first14 = matched[matched["position"] <= 14].merge(crowd, on=["league", "game"], how="left")
    prior = _prior(matches)
    for k, outcome in enumerate(OUTCOMES):
        first14[f"m_{outcome}"] = first14[f"m_{outcome}"].fillna(prior[k])
    first14["result"] = first14["sign"].map({o: i for i, o in enumerate(OUTCOMES)})
    complete = first14.groupby("draw_id")["result"].transform(lambda r: r.notna().all())
    rows = first14[complete]
    probs = rows[[f"m_{o}" for o in OUTCOMES]].to_numpy(dtype=float)
    picked = probs[np.arange(len(rows)), rows["result"].to_numpy(dtype=int)]
    rows = rows.assign(_log=-np.log(np.clip(picked, money.PROB_FLOOR, 1)))
    values = rows.groupby("draw_id")["_log"].sum().rename("surprise").reset_index()
    out = draws.merge(draw_coverage(matched), on="draw_id", how="left")
    return out.merge(values, on="draw_id", how="left")


def _match_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """Market probabilities of every football-data match (``m_H``, ``m_D``, ``m_A``)."""
    probs = MarketModel().fit(matches).predict_proba(matches)
    return matches[["league", "game"]].assign(m_H=probs[:, 0], m_D=probs[:, 1], m_A=probs[:, 2])


def _prior(matches: pd.DataFrame) -> np.ndarray:
    """Shares of 1, X and 2 among played matches (probabilities of unknown matches)."""
    shares = matches["FTR"].value_counts(normalize=True)
    return np.array([float(shares.get(o, 1 / 3)) for o in OUTCOMES])


def _pleno_hit(row: pd.Series, picks: pd.DataFrame) -> bool:
    """Whether the Dixon-Coles Pleno al 15 pick of the 15th match was right."""
    if pd.isna(row.get("game")) or pd.isna(row.get("FTHG")):
        return False
    pick = picks[(picks["league"] == row["league"]) & (picks["game"] == row["game"])]
    if pick.empty or pd.isna(pick["pleno_home"].iloc[0]):
        return False
    actual = PLENO_LABELS[[min(int(row["FTHG"]), 3), min(int(row["FTAG"]), 3)]]
    return bool(pick["pleno_home"].iloc[0] == actual[0] and pick["pleno_away"].iloc[0] == actual[1])


def _season_money(
    season_draws: pd.DataFrame,
    tickets: pd.DataFrame,
    model: money.PrizeModel,
    prior: np.ndarray,
    max_bets: int,
    seed: int,
) -> list[tuple[object, ...]]:
    """Money backtest rows of the draws of one season."""
    rows: list[tuple[object, ...]] = []
    for _, draw in season_draws.iterrows():
        ticket = tickets[tickets["draw_id"] == draw["draw_id"]].sort_values("position")
        first14 = ticket[ticket["position"] <= 14]
        if len(first14) != TICKET_SIZE or first14["sign"].isna().any():
            continue
        probs = first14[["p_H", "p_D", "p_A"]].to_numpy(dtype=float, copy=True)
        crowd = first14[["m_H", "m_D", "m_A"]].to_numpy(dtype=float, copy=True)
        known = ~np.isnan(probs).any(axis=1)
        probs[~known] = prior
        crowd[np.isnan(crowd).any(axis=1)] = probs[np.isnan(crowd).any(axis=1)]
        results = first14["sign"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy(dtype=int)
        winners = {k: float(draw[f"winners_{k}"] or 0) for k in money.CATEGORY_HITS}
        pleno = bool(ticket["_pleno_hit"].iloc[-1]) if len(ticket) > 14 else False
        rng = np.random.default_rng([seed, int(draw["draw_id"]) % 2**31])

        nan = float("nan")
        plans: dict[str, tuple[planner.TicketPlan, float, float, bool, float]] = {
            "single": (planner.single_plan(probs, 11), nan, nan, True, nan),
            "v4": (planner.plan_ticket(probs, max_bets=max_bets, min_hits=11), nan, nan, True, nan),
        }
        args = (probs, crowd, model, float(draw["expected_bets"]), float(draw["jackpot"] or 0))
        prudent = money.plan_money(*args, max_bets=max_bets, rng=rng)
        full = money.plan_money(*args, max_bets=max_bets, top_weight=1.0, rng=rng)
        big = money.plan_big_win(*args, rng=rng)
        plans["v5"] = (prudent.plan, prudent.expected_net, prudent.expected_top, prudent.worth, nan)
        plans["v5_full"] = (full.plan, full.expected_net, full.expected_top, full.worth, nan)
        plans["v5_always"] = (full.plan, full.expected_net, full.expected_top, True, nan)
        plans["v5_big"] = (big.plan, big.expected_payout - big.cost, nan, True, big.chance)
        for key, (plan, expected, expected_top, played, p_big) in plans.items():
            payout, hits, top = money.ticket_payout(
                plan.levels, probs, results, winners, float(draw["takings"]),
                float(draw["jackpot"] or 0), float(draw["winners_special"] or 0), pleno,
            )  # fmt: skip
            cost = BET_PRICE * plan.bets
            rows.append(
                (
                    draw["season"],
                    draw["jornada"],
                    draw["date"],
                    draw["draw_id"],
                    int(draw["matched"]),
                    key,
                    played,
                    plan.doubles,
                    plan.triples,
                    plan.bets,
                    cost if played else 0.0,
                    payout if played else 0.0,
                    (payout - cost) if played else 0.0,
                    hits,
                    expected,
                    expected_top,
                    top if played else 0.0,
                    p_big,
                    float(draw["jackpot"] or 0),
                )
            )
    return rows


def money_backtest(
    backtest: pd.DataFrame,
    draws: pd.DataFrame,
    tickets: pd.DataFrame,
    matches: pd.DataFrame,
    seasons: list[str] | None = None,
    model_key: str = planner.PLANNER_MODEL,
    max_bets: int = money.DEFAULT_MAX_BETS,
    seed: int = 0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Play the real tickets of past draws with each strategy of ``MONEY_STRATEGIES``.

    Args:
        backtest: Walk-forward predictions (``quiniela.evaluation.backtest``).
        draws: Output of ``quiniela.data.prizes.load_draws``.
        tickets: Output of ``quiniela.data.prizes.load_tickets``.
        matches: football-data matches (odds and results) of every season.
        seasons: Seasons to play. Defaults to the seasons of ``backtest``.
        model_key: Model whose probabilities choose the signs (v3).
        max_bets: Budget of v4 and v5.
        seed: Seed of the simulations.
        n_jobs: Parallel jobs (one per season).

    Returns:
        One row per draw and strategy with ``MONEY_COLUMNS``.
    """
    matched = match_tickets(tickets, matches)
    info = draw_surprises(draws, matched, matches)
    # Expected bets: average of the previous draws (known before the draw)
    info = info.sort_values("date")
    info["expected_bets"] = (
        info["bets"].where(info["bets"] > 0).shift(1).rolling(VOLUME_WINDOW, min_periods=1).mean()
    )
    info["expected_bets"] = info["expected_bets"].fillna(info["bets"])
    # Missing winners or jackpot mean none
    amounts = ["jackpot", *[f"winners_{c}" for c in ("special", "14", "13", "12", "11", "10")]]
    info[amounts] = info[amounts].fillna(0)

    probs = backtest[backtest["model"] == model_key][["league", "game", "p_H", "p_D", "p_A"]]
    picks = backtest[backtest["model"] == "dixon_coles"][
        ["league", "game", "pleno_home", "pleno_away"]
    ]
    crowd = _match_probabilities(matches)
    playable = matched.merge(probs, on=["league", "game"], how="left").merge(
        crowd, on=["league", "game"], how="left"
    )
    pleno_rows = playable[playable["position"] == 15]
    hits = {
        draw_id: _pleno_hit(row, picks) for draw_id, row in zip(pleno_rows["draw_id"], pleno_rows.to_dict("records"), strict=True)
    }  # fmt: skip
    playable["_pleno_hit"] = playable["draw_id"].map(hits).fillna(False)

    seasons = sorted(backtest["season"].unique()) if seasons is None else seasons
    jobs = []
    for season in seasons:
        season_draws = info[(info["season"] == season) & (info["matched"] >= MIN_MATCHED)]
        if season_draws.empty:
            continue
        start = season_draws["date"].min()
        past = info[(info["date"] < start) & (info["matched"] >= MIN_MATCHED)]
        model = money.PrizeModel.fit(past)
        prior = _prior(matches[matches["date"] < start])
        tickets_season = playable[playable["draw_id"].isin(season_draws["draw_id"])]
        jobs.append((season_draws, tickets_season, model, prior))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_season_money)(d, t, m, p, max_bets, seed) for d, t, m, p in jobs
    )
    rows = [row for season_rows in results for row in season_rows]
    logger.info("Money backtest: %d draws played", len({r[3] for r in rows}))
    return pd.DataFrame(rows, columns=MONEY_COLUMNS)


def summarize_money(results: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Euros won and lost by each strategy.

    Args:
        results: Output of :func:`money_backtest`.
        by: Extra grouping columns besides ``strategy`` (e.g. ``"season"``).

    Returns:
        One row per strategy (in ``MONEY_STRATEGIES`` order) with ``label``,
        ``draws``, ``played``, ``cost``, ``payout``, ``net``, ``roi`` (net per
        euro spent), ``winning`` (draws with a payout above the cost),
        ``best_payout`` and the draws whose best column got 14 ... 10 hits.
    """
    groups = ["strategy", *(by or [])]
    rows = []
    for key, group in results.groupby(groups, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(groups, values, strict=True))
        played = group[group["played"]]
        row["label"] = MONEY_STRATEGIES.get(str(values[0]), str(values[0]))
        row["draws"] = len(group)
        row["played"] = len(played)
        row["cost"] = float(played["cost"].sum())
        row["payout"] = float(played["payout"].sum())
        row["net"] = row["payout"] - row["cost"]  # type: ignore[operator]
        row["roi"] = row["net"] / row["cost"] if row["cost"] else float("nan")  # type: ignore[operator]
        row["winning"] = int((played["net"] > 0).sum())
        row["best_payout"] = float(played["payout"].max()) if len(played) else 0.0
        # Prizes of 10,000 EUR or more: how many came, and how many the plans expected
        row["big_wins"] = int((played["payout"] >= money.BIG_WIN_THRESHOLD).sum())
        row["expected_big_wins"] = (
            float(played["p_big"].sum())
            if "p_big" in played and played["p_big"].notna().any()
            else None
        )
        for k in money.CATEGORY_HITS:
            row[str(k)] = int((played["best_hits"] == k).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = {key: i for i, key in enumerate(MONEY_STRATEGIES)}
    return summary.sort_values(
        [*groups[1:], "strategy"], key=lambda c: c.map(order) if c.name == "strategy" else c
    ).reset_index(drop=True)


MONEY_FILE = "money.parquet"


def save_money(
    backtest_dir: Path = BACKTEST_DIR,
    lae_dir: Path = LAE_DIR,
    historical_dir: Path = HISTORICAL_DIR,
    current_dir: Path = CURRENT_DIR,
) -> Path | None:
    """Run the money backtest on the saved predictions and LAE draws, and save it.

    Args:
        backtest_dir: Folder of the backtest predictions (and of the output).
        lae_dir: Folder with the saved LAE draws.
        historical_dir: Downloaded seasons (odds and results of the tickets).
        current_dir: Current season files.

    Returns:
        Path of ``money.parquet``, or ``None`` without LAE draws or predictions.
    """
    from quiniela.data.loading import load_current, load_historical
    from quiniela.data.prizes import load_draws
    from quiniela.data.prizes import load_tickets as load_lae_tickets
    from quiniela.evaluation.backtest import load_backtest

    predictions = load_backtest(backtest_dir)
    draws = load_draws(lae_dir)
    if predictions.empty or draws.empty:
        logger.warning("Money backtest skipped: no backtest predictions or LAE draws")
        return None
    matches = pd.concat(
        [load_historical(historical_dir), load_current(current_dir)], ignore_index=True
    )
    results = money_backtest(predictions, draws, load_lae_tickets(lae_dir), matches)
    path = backtest_dir / MONEY_FILE
    results.to_parquet(path, index=False)
    logger.info("Saved %d money backtest rows to %s", len(results), path)
    return path


def load_money(backtest_dir: Path = BACKTEST_DIR) -> pd.DataFrame:
    """Load the saved money backtest (empty if it was never run)."""
    path = backtest_dir / MONEY_FILE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=MONEY_COLUMNS)


@dataclass(frozen=True)
class PrizeContext:
    """What the dashboards need to plan v5 and value tickets in euros.

    Attributes:
        model: Prize model fitted on every saved draw.
        bets: Expected bets of the next draw (average of the last ones).
        jackpot: Estimated jackpot of the next weekend draw.
        draws: Saved draws with the number of La Liga / Segunda matches.
        matched: Ticket matches of the saved draws with their football-data match.
    """

    model: money.PrizeModel
    bets: float
    jackpot: float
    draws: pd.DataFrame
    matched: pd.DataFrame


def estimate_jackpot(draws: pd.DataFrame) -> float:
    """Jackpot of the next weekend draw, from the last weekend draw.

    The Pleno al 15 pool (jackpot plus 7.5% of the takings) carries over when
    nobody wins it; midweek draws have no jackpot.

    Args:
        draws: Saved draws (``date``, ``jackpot``, ``takings``, ``winners_special``).

    Returns:
        The estimated jackpot in EUR (0 when it was won or there is no data).
    """
    weekend = draws[draws["date"].dt.day_of_week >= 5]
    if weekend.empty:
        return 0.0
    last = weekend.iloc[-1]
    if float(last["winners_special"] or 0) > 0:
        return 0.0
    return float(last["jackpot"] or 0) + SPECIAL_SHARE * float(last["takings"] or 0)


def prize_context(
    lae_dir: Path = LAE_DIR,
    historical_dir: Path = HISTORICAL_DIR,
    current_dir: Path = CURRENT_DIR,
) -> PrizeContext | None:
    """Fit the prize model on every saved draw and estimate the next one.

    Args:
        lae_dir: Folder with the saved LAE draws.
        historical_dir: Downloaded seasons.
        current_dir: Current season files.

    Returns:
        The context, or ``None`` without saved draws.
    """
    from quiniela.data.loading import load_current, load_historical
    from quiniela.data.prizes import load_draws
    from quiniela.data.prizes import load_tickets as load_lae_tickets

    draws = load_draws(lae_dir)
    if draws.empty:
        return None
    matches = pd.concat(
        [load_historical(historical_dir), load_current(current_dir)], ignore_index=True
    )
    matched = match_tickets(load_lae_tickets(lae_dir), matches)
    info = draw_surprises(draws, matched, matches).fillna({"winners_special": 0, "jackpot": 0})
    usable = info[info["matched"] >= MIN_MATCHED]
    try:
        model = money.PrizeModel.fit(usable)
    except ValueError:
        return None
    recent = info[info["bets"] > 0].tail(VOLUME_WINDOW)
    return PrizeContext(
        model=model,
        bets=float(recent["bets"].mean()),
        jackpot=estimate_jackpot(info),
        draws=info,
        matched=matched,
    )


def matchday_payout(
    context: PrizeContext,
    games: pd.DataFrame,
    picks: list[str],
    results: list[str],
    crowd: np.ndarray,
) -> dict[str, object]:
    """Prize a ticket would have won in a played matchday.

    With the real draw of that matchday (10 or more of its matches on the
    ticket) the real winners and takings are used; otherwise the prize model
    at the real surprise of the results estimates them.

    Args:
        context: Prize context.
        games: ``league`` and ``game`` of the ticket matches, in ticket order.
        picks: Signs marked in each match.
        results: Real results (``H``, ``D``, ``A``).
        crowd: Market probabilities of the matches (for the estimate).

    Returns:
        ``payout`` in EUR, ``best_hits``, ``source`` (``"draw"`` or ``"model"``)
        and ``jornada`` of the real draw when found.
    """
    columns = money.picks_columns(picks, results)
    best_hits = int(np.max(np.nonzero(columns)[0])) if columns.any() else 0
    keys = set(zip(games["league"], games["game"], strict=True))
    found = context.matched[
        [
            (lg, g) in keys
            for lg, g in zip(context.matched["league"], context.matched["game"], strict=True)
        ]
    ]
    if not found.empty:
        overlap = found.groupby("draw_id").size()
        draw_id = overlap.idxmax()
        if overlap.max() >= MIN_MATCHED:
            draw = context.draws[context.draws["draw_id"] == draw_id].iloc[0]
            winners = {k: float(draw[f"winners_{k}"] or 0) for k in money.CATEGORY_HITS}
            payout, _ = money.prize_payout(columns, winners, float(draw["takings"]))
            return {
                "payout": payout,
                "best_hits": best_hits,
                "source": "draw",
                "jornada": int(draw["jornada"]),
            }
    index = np.array([OUTCOMES.index(r) for r in results])
    level = money.surprise(crowd, index)
    prizes, _ = context.model.prizes(level, context.bets, 0.0)
    return {
        "payout": float((columns[:15] * prizes[0]).sum()),
        "best_hits": best_hits,
        "source": "model",
        "jornada": None,
    }
