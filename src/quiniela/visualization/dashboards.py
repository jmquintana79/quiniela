"""Build static HTML dashboards from the downloaded match data.

Each dashboard is a single self-contained HTML file: the data is computed here
with pandas, serialised to JSON and embedded in an HTML template, so the file
opens with a double click (no server, no network access needed).

Dashboards:

- ``current.html``: current matchday (matches, results when played, one
  column per model (the market reference is left out: v3 follows it
  closely), the confidence of each
  match, hit rate, quiniela ticket with the chance of each prize, and the
  v4 ticket with doubles and triples) and the standings.
- ``season.html``: every matchday of the current season, season totals per
  model and the detail of any matchday (without the market reference).
- ``historical.html``: completed seasons (1X2 split and goals per season,
  plus standings and results of the selected season).
- ``evaluation.html``: backtest metrics of every model over past seasons,
  one column per model, with the market as reference; hit rate by
  confidence tier, and the prizes of the tickets played with doubles and
  triples (random, by eye, optimal and v4).
"""

import json
import logging
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quiniela.config import CURRENT_DIR, DASHBOARDS_DIR, HISTORICAL_DIR
from quiniela.io.download import LEAGUES

logger = logging.getLogger(__name__)

# football-data.co.uk league code -> display name
LEAGUE_NAMES: dict[str, str] = {
    LEAGUES["ESP-La Liga"]: "La Liga",
    LEAGUES["ESP-Segunda"]: "Segunda",
}

# Spain moved from 2 to 3 points per win in the 1995-96 season
THREE_POINTS_FROM = 1995

# Number of recent matches shown as "form" in the standings
FORM_LENGTH = 5

# Columns kept for every match (the rest of the ~200 columns are betting odds)
MATCH_COLUMNS = ["date", "home_team", "away_team", "FTHG", "FTAG", "FTR"]

# Placeholders replaced in the templates
STYLE_PLACEHOLDER = "/*__STYLE__*/"
SCRIPT_PLACEHOLDER = "/*__SCRIPT__*/"
DATA_PLACEHOLDER = "/*__DATA__*/null"


# %% DATA PREPARATION


def standings(matches: pd.DataFrame, points_per_win: int = 3) -> pd.DataFrame:
    """Compute a league table from match results.

    Teams are ranked by points, goal difference and goals scored. Head-to-head
    tiebreakers and point deductions are not applied, so positions can differ
    slightly from the official table.

    Args:
        matches: Played matches with ``home_team``, ``away_team``, ``FTHG``,
            ``FTAG`` and ``date`` columns.
        points_per_win: Points awarded for a win.

    Returns:
        One row per team with ``team``, ``played``, ``won``, ``drawn``,
        ``lost``, ``gf``, ``ga``, ``gd``, ``points`` and ``form`` (results of
        the last matches, oldest first, as a string such as ``"WDLWW"``).
    """
    played = matches.dropna(subset=["FTHG", "FTAG"])
    # One row per team and match, from that team's point of view
    home = pd.DataFrame(
        {
            "team": played["home_team"],
            "date": played["date"],
            "gf": played["FTHG"],
            "ga": played["FTAG"],
        }
    )
    away = pd.DataFrame(
        {
            "team": played["away_team"],
            "date": played["date"],
            "gf": played["FTAG"],
            "ga": played["FTHG"],
        }
    )
    rows = pd.concat([home, away], ignore_index=True)
    rows["result"] = "D"
    rows.loc[rows["gf"] > rows["ga"], "result"] = "W"
    rows.loc[rows["gf"] < rows["ga"], "result"] = "L"

    table = rows.groupby("team").agg(
        played=("result", "size"),
        won=("result", lambda r: int((r == "W").sum())),
        drawn=("result", lambda r: int((r == "D").sum())),
        lost=("result", lambda r: int((r == "L").sum())),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
    )
    table["gd"] = table["gf"] - table["ga"]
    table["points"] = table["won"] * points_per_win + table["drawn"]
    table["form"] = (
        rows.sort_values("date")
        .groupby("team")["result"]
        .agg(lambda r: "".join(r.tail(FORM_LENGTH)))
    )
    table = table.reset_index().sort_values(
        ["points", "gd", "gf", "team"], ascending=[False, False, False, True]
    )
    return table.astype({c: int for c in ["gf", "ga", "gd", "points"]}).reset_index(drop=True)


def season_summary(matches: pd.DataFrame) -> dict[str, float | int]:
    """Summarise the outcomes of a set of matches.

    Args:
        matches: Played matches with ``FTR``, ``FTHG`` and ``FTAG`` columns.

    Returns:
        Number of matches, share of home wins / draws / away wins and average
        goals per match.
    """
    outcomes = matches["FTR"].value_counts(normalize=True)
    return {
        "matches": len(matches),
        "home": round(float(outcomes.get("H", 0.0)), 4),
        "draw": round(float(outcomes.get("D", 0.0)), 4),
        "away": round(float(outcomes.get("A", 0.0)), 4),
        "goals": round(float((matches["FTHG"] + matches["FTAG"]).mean()), 3),
    }


def _season_start_year(path: Path) -> int:
    """Read the season start year from a historical file name (``2015-16.parquet``)."""
    return int(path.stem.split("-")[0])


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-friendly records (dates as ISO strings, NaN as null)."""
    out = df.copy()
    for column in out.select_dtypes(include="datetime").columns:
        out[column] = out[column].dt.strftime("%Y-%m-%d %H:%M")
    records: list[dict[str, Any]] = json.loads(out.to_json(orient="records"))
    return records


def _matches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the match columns used by the dashboards, sorted by date."""
    return df.reset_index(drop=True)[MATCH_COLUMNS].sort_values("date").reset_index(drop=True)


def historical_payload(historical_dir: Path = HISTORICAL_DIR) -> dict[str, Any]:
    """Collect the data shown in the historical dashboard.

    The per-season summary is precomputed for every season. Matches and
    standings are embedded too, but the page only renders the selected season.

    Args:
        historical_dir: Folder with ``<code>/<season>.parquet`` files.

    Returns:
        JSON-serialisable dictionary with one entry per league.
    """
    leagues = []
    for code, name in LEAGUE_NAMES.items():
        paths = sorted((historical_dir / code).glob("*.parquet"), key=_season_start_year)
        if not paths:
            logger.warning("No historical data for %s in %s", name, historical_dir / code)
            continue
        seasons = []
        for path in paths:
            start_year = _season_start_year(path)
            matches = _matches(pd.read_parquet(path))
            points = 3 if start_year >= THREE_POINTS_FROM else 2
            seasons.append(
                {
                    "season": path.stem,
                    "summary": season_summary(matches),
                    "standings": _records(standings(matches, points).drop(columns="form")),
                    "matches": _records(matches),
                }
            )
        leagues.append({"code": code, "name": name, "seasons": seasons})
    return {"leagues": leagues}


# Key and display information of the v4 planned ticket (not a registered model:
# a decision layer over the planner model's probabilities)
PLANNED_KEY = "planned"
PLANNED_SPEC = {
    "key": PLANNED_KEY,
    "name": "Planned ticket",
    "version": "v4",
    "reference": False,
    "description": "v3 signs plus the best doubles and triples for a 48-bet budget, chosen to "
    "maximise the chance of 11 or more hits. Outside the ticket, the v3 sign.",
    # A way of playing, not a probability model: never "best model" in the evaluation
    "ticket": True,
}


# The v5 money ticket: also a decision layer over the planner model, shown when
# the saved LAE draws allow fitting the prize model
MONEY_KEY = "money"
MONEY_SPEC = {
    "key": MONEY_KEY,
    "name": "Money ticket",
    "version": "v5",
    "reference": False,
    "description": "v3 signs plus the doubles and triples (up to 48 bets) with the highest "
    "expected prize in euros from 10 to 13 hits, played only when that prize is above the cost "
    "(prudent: the rare 14-hit and Pleno al 15 prizes do not count). Outside the ticket, the v3 "
    "sign.",
    "ticket": True,
}

# Seed of the v5 simulations (the same plan on every rebuild)
MONEY_SEED = 0


def _money_matchday(
    group: pd.DataFrame, key: str, context: Any
) -> tuple[Any, dict[str, Any]] | None:
    """Plan the v5 money ticket of a matchday, or None without a full ticket with probabilities."""
    from quiniela.prediction.money import plan_money
    from quiniela.prediction.ticket import TICKET_SIZE

    ticket = group.dropna(subset=["ticket"]).sort_values("ticket")
    columns = [f"{key}_p_{o}" for o in ("H", "D", "A")]
    if len(ticket) < TICKET_SIZE or ticket[columns].isna().any().any():
        return None
    probs = ticket[columns].to_numpy(dtype=float)
    crowd = ticket[[f"_crowd_{o}" for o in ("H", "D", "A")]].to_numpy(dtype=float, copy=True)
    missing = np.isnan(crowd).any(axis=1)
    crowd[missing] = probs[missing]
    best = plan_money(
        probs,
        crowd,
        context.model,
        context.bets,
        context.jackpot,
        rng=np.random.default_rng([MONEY_SEED, int(group["matchday"].iloc[0])]),
    )
    confidence = probs.max(axis=1)
    return best, {
        "picks": dict(zip(ticket.index, best.plan.picks, strict=True)),
        "coverage": dict(zip(ticket.index, best.plan.coverage, strict=True)),
        "summary": {
            "doubles": best.plan.doubles,
            "triples": best.plan.triples,
            "bets": best.plan.bets,
            "cost": round(best.cost, 2),
            "expected_payout": round(best.expected_payout, 2),
            "expected_top": round(best.expected_top, 2),
            "expected_net": round(best.expected_net, 2),
            "expected_common": round(best.expected_common, 2),
            "worth": bool(best.worth),
            # Counting the 14-hit and Pleno al 15 prizes too (not the verdict)
            "worth_full": bool(best.expected_net > 0),
            "prize_chance": round(best.prize_chance, 4),
            "jackpot": round(float(context.jackpot), 0),
            "expected_surprise": round(best.expected_surprise, 2),
            # How predictable the ticket is: average v3 confidence of its 14 matches
            "predictability": round(float(confidence.mean()), 4),
            "odds": _ticket_odds(best.plan.coverage),
        },
    }


def _ticket_odds(coverage: np.ndarray) -> dict[str, float]:
    """Chance of each prize category of a ticket, as JSON-friendly keys."""
    from quiniela.prediction.planner import category_probabilities

    return {str(k): round(v, 5) for k, v in category_probabilities(coverage).items()}


def _plan_matchday(group: pd.DataFrame, key: str) -> dict[str, Any] | None:
    """Plan the v4 ticket of a matchday from the probabilities of model ``key``.

    Args:
        group: Matches of the matchday with ``ticket`` positions and the
            model's ``<key>_p_H``, ``<key>_p_D``, ``<key>_p_A`` columns.
        key: Model whose probabilities choose the signs.

    Returns:
        The picks and coverage per match index, plus the plan summary, or
        ``None`` without a full ticket with probabilities.
    """
    from quiniela.prediction.planner import DEFAULT_MAX_BETS, plan_ticket
    from quiniela.prediction.ticket import TICKET_SIZE

    ticket = group.dropna(subset=["ticket"]).sort_values("ticket")
    columns = [f"{key}_p_{o}" for o in ("H", "D", "A")]
    if len(ticket) < TICKET_SIZE or ticket[columns].isna().any().any():
        return None
    plan = plan_ticket(ticket[columns].to_numpy(dtype=float), max_bets=DEFAULT_MAX_BETS)
    return {
        "picks": dict(zip(ticket.index, plan.picks, strict=True)),
        "coverage": dict(zip(ticket.index, plan.coverage, strict=True)),
        "summary": {
            "doubles": plan.doubles,
            "triples": plan.triples,
            "bets": plan.bets,
            "odds": _ticket_odds(plan.coverage),
        },
    }


def _score(hits: int, misses: int) -> dict[str, Any]:
    """Hits, misses and hit rate of a set of predictions."""
    played = hits + misses
    return {"hits": hits, "misses": misses, "rate": round(hits / played, 4) if played else None}


def season_payload(
    current: pd.DataFrame | None = None,
    predictions: dict[str, pd.DataFrame] | None = None,
    prizes: Any = None,
) -> dict[str, Any]:
    """Collect the data shown in the season dashboard.

    Every matchday of the current season is listed with the real result (when
    played) and one prediction column per model, plus each model's hits and
    the outcome of its 14-match quiniela ticket.

    Args:
        current: Current season with ``round`` and ``matchday`` columns.
            Defaults to the downloaded data.
        predictions: Stored predictions per model key. Defaults to the saved files.
        prizes: Prize context (``quiniela.evaluation.money.PrizeContext``) for
            the v5 money ticket and the euros won. Defaults to the saved LAE
            draws when the predictions are the saved ones too; without it, v5
            and the euros are left out.

    Returns:
        JSON-serialisable dictionary with the models, matchdays and season totals.
    """
    # Imported here: only the current and season dashboards need the prediction package
    from quiniela.prediction.confidence import TIERS, confidence
    from quiniela.prediction.pipeline import current_matchdays, load_predictions
    from quiniela.prediction.planner import PLANNER_MODEL
    from quiniela.prediction.registry import MODELS
    from quiniela.prediction.ticket import evaluate_ticket, ticket_positions

    current = current_matchdays() if current is None else current
    if predictions is None:
        predictions = load_predictions()
        if prizes is None:
            from quiniela.evaluation.money import prize_context

            prizes = prize_context()
    # Models hidden in the matchday views (the market reference) are only in the evaluation
    keys = [key for key in MODELS if key in predictions and MODELS[key].show_in_matchday]
    models = _model_specs(keys)
    # The v4 planned ticket needs the planner model's predictions; it goes
    # right after that model (before the references)
    planned = PLANNER_MODEL in keys
    if planned:
        position = next(i for i, m in enumerate(models) if m["key"] == PLANNER_MODEL) + 1
        models.insert(position, PLANNED_SPEC)
    with_money = planned and prizes is not None
    if with_money:
        models.insert(position + 1, MONEY_SPEC)

    # One column per model: predicted sign, its probability and, for models
    # that predict goals, the exact score and Pleno al 15 picks
    matches = current.copy()
    # Market probabilities stand for the public in the v5 prize estimates
    market = predictions.get("market")
    for outcome in ("H", "D", "A"):
        matches[f"_crowd_{outcome}"] = (
            market.set_index(["league", "game"])
            .reindex(pd.MultiIndex.from_frame(matches[["league", "game"]]))[f"p_{outcome}"]
            .to_numpy()
            if market is not None
            else np.nan
        )
    for key in keys:
        p = predictions[key].set_index(["league", "game"])
        aligned = p.reindex(pd.MultiIndex.from_frame(matches[["league", "game"]]))
        signs = aligned["prediction"]
        matches[f"{key}_sign"] = signs.to_numpy()
        # Probability of the predicted sign (a model may not always pick the largest one)
        prob = pd.Series(np.nan, index=aligned.index)
        for outcome in ("H", "D", "A"):
            chosen = (signs == outcome).to_numpy()
            prob[chosen] = aligned.loc[chosen, f"p_{outcome}"]
        matches[f"{key}_prob"] = prob.round(4).to_numpy()
        # Confidence (probability of the predicted sign) and margin over the second sign,
        # recomputed from the probabilities so older prediction files work too
        probs = aligned[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
        conf = confidence(probs, signs.to_numpy())
        matches[f"{key}_conf"] = conf["confidence"].round(4).to_numpy()
        matches[f"{key}_margin"] = conf["margin"].round(4).to_numpy()
        if key == PLANNER_MODEL:
            for outcome in ("H", "D", "A"):
                matches[f"{key}_p_{outcome}"] = aligned[f"p_{outcome}"].to_numpy()
        if "score_home" in aligned.columns and aligned["score_home"].notna().any():
            scored = aligned["score_home"].notna().to_numpy()
            score = pd.Series(None, index=aligned.index, dtype=object)
            pleno = pd.Series(None, index=aligned.index, dtype=object)
            score[scored] = (
                aligned.loc[scored, "score_home"].astype(int).astype(str)
                + "-"
                + aligned.loc[scored, "score_away"].astype(int).astype(str)
            )
            pleno[scored] = (
                aligned.loc[scored, "pleno_home"] + "-" + aligned.loc[scored, "pleno_away"]
            )
            matches[f"{key}_score"] = score.to_numpy()
            matches[f"{key}_pleno"] = pleno.to_numpy()

    columns_keys = [m["key"] for m in models]
    totals: dict[str, dict[str, Any]] = {
        key: {"hits": 0, "misses": 0, "missing": 0, "prizes": 0, "tickets": 0}
        for key in columns_keys
    }
    matchdays = []
    for number, group in matches.groupby("matchday"):
        group = group.assign(ticket=ticket_positions(group))
        group = group.sort_values(["ticket", "league", "date"], na_position="last")
        played = group["FTR"].notna()
        in_ticket = group["ticket"].notna()

        plan = _plan_matchday(group, PLANNER_MODEL) if planned else None
        if planned:
            # v4: the planned picks on the ticket, the planner model's sign elsewhere
            group[f"{PLANNED_KEY}_sign"] = group[f"{PLANNER_MODEL}_sign"]
            group[f"{PLANNED_KEY}_prob"] = group[f"{PLANNER_MODEL}_prob"]
            if plan is not None:
                picks = pd.Series(plan["picks"], dtype=object)
                group.loc[picks.index, f"{PLANNED_KEY}_sign"] = picks
                group.loc[picks.index, f"{PLANNED_KEY}_prob"] = (
                    pd.Series(plan["coverage"]).round(4).to_numpy()
                )
            group[f"{PLANNED_KEY}_conf"] = group[f"{PLANNED_KEY}_prob"]
            group[f"{PLANNED_KEY}_margin"] = group[f"{PLANNER_MODEL}_margin"]
        money_plan = _money_matchday(group, PLANNER_MODEL, prizes) if with_money else None
        if with_money:
            # v5: like v4, with the doubles and triples of the best expected prize
            group[f"{MONEY_KEY}_sign"] = group[f"{PLANNER_MODEL}_sign"]
            group[f"{MONEY_KEY}_prob"] = group[f"{PLANNER_MODEL}_prob"]
            if money_plan is not None:
                picks = pd.Series(money_plan[1]["picks"], dtype=object)
                group.loc[picks.index, f"{MONEY_KEY}_sign"] = picks
                group.loc[picks.index, f"{MONEY_KEY}_prob"] = (
                    pd.Series(money_plan[1]["coverage"]).round(4).to_numpy()
                )
            group[f"{MONEY_KEY}_conf"] = group[f"{MONEY_KEY}_prob"]
            group[f"{MONEY_KEY}_margin"] = group[f"{PLANNER_MODEL}_margin"]

        summary: dict[str, Any] = {}
        for key in columns_keys:
            predicted = group[f"{key}_sign"].notna()
            # A pick may cover several signs (doubles and triples of the v4 ticket)
            covered = [
                isinstance(sign, str) and isinstance(result, str) and result in sign
                for sign, result in zip(group[f"{key}_sign"], group["FTR"], strict=True)
            ]
            hit = played & predicted & pd.Series(covered, index=group.index)
            miss = played & predicted & ~hit
            ticket = evaluate_ticket(
                size=int(in_ticket.sum()),
                hits=int((hit & in_ticket).sum()),
                pending=int((in_ticket & ~played).sum()),
            )
            summary[key] = {
                **_score(int(hit.sum()), int(miss.sum())),
                "pending": int((predicted & ~played).sum()),
                # Matches the model could not predict: a third category next to
                # hits and misses, left out of the hit rate
                "missing": int((~predicted).sum()),
                # On the ticket they count as wrong lines (no sign marked)
                "missing_in_ticket": int((~predicted & in_ticket).sum()),
                "ticket": ticket.to_dict(),
                # Chance of each prize before the matchday (independent matches)
                "odds": _model_odds(group, key, in_ticket, predicted),
            }
            if key == PLANNED_KEY and plan is not None:
                summary[key]["plan"] = plan["summary"]
            if key == MONEY_KEY and money_plan is not None:
                summary[key]["plan"] = money_plan[1]["summary"]
                summary[key]["money"] = money_plan[1]["summary"]
            euros = _matchday_euros(group, key, prizes, money_plan) if with_money else None
            if euros is not None:
                summary[key]["euros"] = euros
                spent = totals[key].setdefault("euros", {"cost": 0.0, "payout": 0.0, "played": 0})
                spent["cost"] += euros["cost"]
                spent["payout"] += euros["payout"]
                spent["played"] += int(euros["played"])
            totals[key]["hits"] += int(hit.sum())
            totals[key]["misses"] += int(miss.sum())
            totals[key]["missing"] += int((~predicted).sum())
            if ticket.status in ("prize", "no_prize"):
                totals[key]["tickets"] += 1
                totals[key]["prizes"] += ticket.status == "prize"

        columns = ["ticket", "league", "round", "date", "home_team", "away_team"]
        columns += ["FTHG", "FTAG", "FTR"]
        columns += [
            f"{k}_{field}" for k in columns_keys for field in ("sign", "prob", "conf", "margin")
        ]
        columns += [c for k in keys for c in (f"{k}_score", f"{k}_pleno") if c in group.columns]
        matchdays.append(
            {
                "number": int(number),
                "start": group["date"].min().strftime("%Y-%m-%d"),
                "end": group["date"].max().strftime("%Y-%m-%d"),
                "played": int(played.sum()),
                "matches": _records(group[columns]),
                "summary": summary,
            }
        )

    return {
        "models": models,
        "leagues": LEAGUE_NAMES,
        # Model whose confidence ranks and colours the matches (v3)
        "confidence_model": PLANNER_MODEL if PLANNER_MODEL in keys else (keys[0] if keys else None),
        "confidence_tiers": TIERS,
        "matchdays": matchdays,
        "totals": {
            k: {
                **_score(t["hits"], t["misses"]),
                "missing": t["missing"],
                "prizes": t["prizes"],
                "tickets": t["tickets"],
                **(
                    {"euros": {k2: round(v, 2) for k2, v in t["euros"].items()}}
                    if "euros" in t
                    else {}
                ),
            }
            for k, t in totals.items()
        },
    }


def _matchday_euros(
    group: pd.DataFrame, key: str, context: Any, money_plan: Any
) -> dict[str, Any] | None:
    """Cost and prize of a model's ticket in a played matchday (v3, v4 and v5 only).

    v3 plays one column, v4 its doubles and triples and v5 its ticket only when
    it was worth it (otherwise nothing is spent or won).
    """
    from quiniela.evaluation.money import matchday_payout
    from quiniela.data.prizes import BET_PRICE
    from quiniela.prediction.planner import PLANNER_MODEL
    from quiniela.prediction.ticket import TICKET_SIZE

    if key not in (PLANNER_MODEL, PLANNED_KEY, MONEY_KEY):
        return None
    ticket = group.dropna(subset=["ticket"]).sort_values("ticket")
    if (
        len(ticket) < TICKET_SIZE
        or ticket["FTR"].isna().any()
        or ticket[f"{key}_sign"].isna().any()
    ):
        return None
    picks = [str(p) for p in ticket[f"{key}_sign"]]
    bets = int(np.prod([len(p) for p in picks]))
    played = key != MONEY_KEY or (money_plan is not None and money_plan[0].worth)
    if not played:
        return {"cost": 0.0, "payout": 0.0, "played": False, "source": None, "jornada": None}
    crowd = ticket[[f"_crowd_{o}" for o in ("H", "D", "A")]].to_numpy(dtype=float, copy=True)
    fallback = ticket[[f"{PLANNER_MODEL}_p_{o}" for o in ("H", "D", "A")]].to_numpy(dtype=float)
    missing = np.isnan(crowd).any(axis=1)
    crowd[missing] = fallback[missing]
    result = matchday_payout(context, ticket[["league", "game"]], picks, list(ticket["FTR"]), crowd)
    return {
        "cost": round(BET_PRICE * bets, 2),
        "payout": round(float(result["payout"]), 2),  # type: ignore[arg-type]
        "played": True,
        "source": result["source"],
        "jornada": result["jornada"],
    }


def _model_odds(
    group: pd.DataFrame, key: str, in_ticket: pd.Series, predicted: pd.Series
) -> dict[str, float] | None:
    """Chance of each prize of a model's ticket, or None without a full predicted ticket."""
    from quiniela.prediction.ticket import TICKET_SIZE

    ticket = group[in_ticket & predicted]
    if int(in_ticket.sum()) < TICKET_SIZE or len(ticket) < TICKET_SIZE:
        return None
    coverage = ticket[f"{key}_prob"].to_numpy(dtype=float)
    if np.isnan(coverage).any():
        return None
    return _ticket_odds(coverage)


def current_matchday_index(matchdays: list[dict[str, Any]]) -> int | None:
    """Pick the matchday shown as current: the first one with matches still to play.

    When every published match has been played (e.g. during an international
    break, before the source publishes the next fixtures) the last matchday is
    shown instead.

    Args:
        matchdays: Matchdays of ``season_payload`` in date order.

    Returns:
        Index of the current matchday, or ``None`` without matchdays.
    """
    if not matchdays:
        return None
    pending = [i for i, md in enumerate(matchdays) if md["played"] < len(md["matches"])]
    return pending[0] if pending else len(matchdays) - 1


def current_payload(
    current: pd.DataFrame | None = None,
    predictions: dict[str, pd.DataFrame] | None = None,
    current_dir: Path = CURRENT_DIR,
    season: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the data shown in the current matchday dashboard.

    Args:
        current: Current season with ``round`` and ``matchday`` columns.
            Defaults to the downloaded data.
        predictions: Stored predictions per model key. Defaults to the saved files.
        current_dir: Folder with the current season files (standings and the
            download time of the odds).
        season: Season payload already built (``season_payload``), to reuse it.

    Returns:
        JSON-serialisable dictionary with the models, the current matchday,
        its status and the standings of each league.
    """
    season = season_payload(current, predictions) if season is None else season
    index = current_matchday_index(season["matchdays"])
    matchday = None if index is None else season["matchdays"][index]
    if matchday is None:
        status = "empty"
    elif matchday["played"] == 0:
        status = "upcoming"
    elif matchday["played"] < len(matchday["matches"]):
        status = "in_progress"
    else:
        status = "completed"

    standings_by_league: dict[str, list[dict[str, Any]]] = {}
    odds_downloaded = None
    for code in LEAGUE_NAMES:
        results_path = current_dir / code / "results.parquet"
        if results_path.exists():
            results = _matches(pd.read_parquet(results_path))
            standings_by_league[code] = _records(standings(results))
        fixtures_path = current_dir / code / "fixtures.parquet"
        if fixtures_path.exists():
            # Odds change until kick-off: the page says when this snapshot was taken
            stamp = datetime.fromtimestamp(fixtures_path.stat().st_mtime)
            odds_downloaded = max(odds_downloaded or stamp, stamp)

    return {
        "models": season["models"],
        "leagues": season["leagues"],
        "confidence_model": season["confidence_model"],
        "confidence_tiers": season["confidence_tiers"],
        "matchday": matchday,
        "status": status,
        "matchdays": len(season["matchdays"]),
        "standings": standings_by_league,
        "odds_downloaded": odds_downloaded.strftime("%Y-%m-%d %H:%M") if odds_downloaded else None,
    }


def _model_specs(keys: list[str]) -> list[dict[str, Any]]:
    """Display information of the given registered models, in registry order."""
    from quiniela.prediction.registry import MODELS

    return [
        {
            "key": key,
            "name": MODELS[key].name,
            "version": MODELS[key].version,
            "reference": MODELS[key].reference,
            "description": MODELS[key].description,
        }
        for key in MODELS
        if key in keys
    ]


# Metrics shown in the evaluation dashboard: key, label, which direction is
# better, display format, and the explanation and formula shown in the page
EVALUATION_METRICS: list[dict[str, str]] = [
    {
        "key": "hit_rate",
        "label": "Hit rate",
        "better": "higher",
        "format": "pct",
        "description": "Share of matches whose predicted sign (1, X or 2) was the real result.",
        "formula": "hits / N",
    },
    {
        "key": "log_loss",
        "label": "Log-loss",
        "better": "lower",
        "format": "num",
        "description": "Average surprise: minus the log of the probability given to the real "
        "result. Punishes confident mistakes hard. Always predicting 1/3 each scores 1.099.",
        "formula": "−(1/N) Σᵢ log pᵢ,real",
    },
    {
        "key": "brier",
        "label": "Brier score",
        "better": "lower",
        "format": "num",
        "description": "Average squared distance between the three probabilities and what "
        "happened (1 for the real result, 0 for the others). From 0 (perfect) to 2.",
        "formula": "(1/N) Σᵢ Σₖ (pᵢₖ − oᵢₖ)²",
    },
    {
        "key": "rps",
        "label": "RPS",
        "better": "lower",
        "format": "num",
        "description": "Ranked probability score: like Brier but on cumulative probabilities, "
        "so 1 → X → 2 are ordered and predicting a draw is a smaller mistake than "
        "predicting the opposite result. From 0 to 1.",
        "formula": "(1/N) Σᵢ ½ Σₖ₌₁² (Σⱼ≤ₖ pᵢⱼ − Σⱼ≤ₖ oᵢⱼ)²",
    },
    {
        "key": "draw_picks",
        "label": "X picks",
        "better": "info",
        "format": "pct",
        "description": "Share of matches where the model predicted a draw, to compare with "
        "the real share of draws below.",
        "formula": "",
    },
    {
        "key": "draw_rate",
        "label": "Real draws",
        "better": "info",
        "format": "pct",
        "description": "Share of matches that ended in a draw.",
        "formula": "",
    },
    {
        "key": "score_hit_rate",
        "label": "Exact score",
        "better": "higher",
        "format": "pct",
        "description": "Share of matches whose most likely score was the real one (models "
        "that predict goals).",
        "formula": "",
    },
    {
        "key": "pleno_hit_rate",
        "label": "Pleno al 15",
        "better": "higher",
        "format": "pct",
        "description": "Share of matches where both teams' goals category (0, 1, 2, M) was "
        "right, as in the Pleno al 15.",
        "formula": "",
    },
    {
        "key": "prizes",
        "label": "Prize tickets",
        "better": "higher",
        "format": "int",
        "description": "Matchdays whose 14-match ticket got 10 or more hits (5th category or "
        "better), out of the complete tickets. Only for both leagues together.",
        "formula": "",
    },
    {
        "key": "matches",
        "label": "Matches",
        "better": "info",
        "format": "int",
        "description": "Matches predicted (matches without a prediction, e.g. no odds for the "
        "market, are left out of every metric).",
        "formula": "",
    },
]


def _clean(record: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN with None and numpy numbers with plain Python ones (JSON-friendly)."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float | np.floating) and np.isnan(value):
            out[key] = None
        elif isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = round(float(value), 5)
        elif isinstance(value, float):
            out[key] = round(value, 5)
        else:
            out[key] = value
    return out


def _confidence_tables(backtest: pd.DataFrame) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Hit rate of every model by confidence tier, keyed by ``"<league>|<season>"``."""
    from quiniela.prediction.confidence import TIERS, confidence

    predicted = backtest.dropna(subset=["p_H", "p_D", "p_A"])
    conf = confidence(
        predicted[["p_H", "p_D", "p_A"]].to_numpy(dtype=float), predicted["prediction"].to_numpy()
    )["confidence"].to_numpy()
    names = list(TIERS)
    # Tiers are ordered from the highest lower edge down: the first one reached wins
    tier = np.select([conf >= low for low in TIERS.values()], names, default=names[-1])
    table = predicted.assign(
        _conf=conf, _tier=tier, _hit=(predicted["prediction"] == predicted["FTR"]).to_numpy()
    )

    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for by in ([], ["season"], ["league"], ["season", "league"]):
        groups = table.groupby([*by, "model"]) if by else table.groupby("model")
        for key, rows in groups:
            values = key if isinstance(key, tuple) else (key,)
            labels = dict(zip([*by, "model"], values, strict=True))
            league_season = f"{labels.get('league', 'all')}|{labels.get('season', 'all')}"
            out.setdefault(league_season, {})[str(labels["model"])] = [
                _clean(
                    {
                        "tier": name,
                        "low": TIERS[name],
                        "matches": int((rows["_tier"] == name).sum()),
                        "share": float((rows["_tier"] == name).mean()),
                        "hit_rate": float(rows.loc[rows["_tier"] == name, "_hit"].mean())
                        if (rows["_tier"] == name).any()
                        else None,
                        "confidence": float(rows.loc[rows["_tier"] == name, "_conf"].mean())
                        if (rows["_tier"] == name).any()
                        else None,
                    }
                )
                for name in names
            ]
    return out


def _planned_results(backtest: pd.DataFrame, tickets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Hit rate and prizes of the v4 planned ticket, keyed by ``"<league>|<season>"``.

    v4 marks the planner model's sign in every match and, on the ticket, its
    doubles and triples: a match is a hit when its result is among the
    marked signs. Probability metrics do not apply (v4 has v3's probabilities).
    Prizes are only given for both leagues together, as for the models.
    """
    from quiniela.evaluation.backtest import summarize_tickets, ticket_picks
    from quiniela.prediction.planner import PLANNER_MODEL

    picks = ticket_picks(tickets)
    base = backtest[backtest["model"] == PLANNER_MODEL].dropna(subset=["prediction", "FTR"])
    if picks.empty or base.empty:
        return {}
    merged = base.merge(
        picks[["season", "league", "game", "picks"]], on=["season", "league", "game"], how="left"
    )
    signs = merged["picks"].fillna(merged["prediction"])
    table = merged.assign(
        _hit=[result in sign for sign, result in zip(signs, merged["FTR"], strict=True)]
    )

    out: dict[str, dict[str, Any]] = {}
    for by in ([], ["season"], ["league"], ["season", "league"]):
        groups = table.groupby(by) if by else [((), table)]
        for key, rows in groups:
            values = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(by, values, strict=True))
            league_season = f"{labels.get('league', 'all')}|{labels.get('season', 'all')}"
            out[league_season] = {"matches": len(rows), "hit_rate": float(rows["_hit"].mean())}

    planned = tickets[tickets["strategy"] == "v4"]
    for by in ([], ["season"]):
        for row in summarize_tickets(planned, by).to_dict(orient="records"):
            key = f"all|{row.get('season', 'all')}"
            if key in out:
                out[key].update(prizes=int(row["prizes"]), tickets=int(row["tickets"]))
    return {key: _clean(value) for key, value in out.items()}


def _ticket_tables(tickets: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Prizes of every ticket strategy, for every season and in total (``"all"``)."""
    from quiniela.evaluation.backtest import summarize_tickets

    if tickets.empty:
        return {}
    out = {"all": [_clean(r) for r in summarize_tickets(tickets).to_dict(orient="records")]}
    for row in summarize_tickets(tickets, ["season"]).to_dict(orient="records"):
        out.setdefault(str(row["season"]), []).append(_clean(row))
    return out


def _money_labels() -> dict[str, str]:
    """Labels of the money strategies."""
    from quiniela.evaluation.money import MONEY_STRATEGIES

    return dict(MONEY_STRATEGIES)


def _money_tables(results: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Euros of every money strategy on the real tickets, per season and in total."""
    from quiniela.evaluation.money import summarize_money

    if results.empty:
        return {}
    out = {"all": [_clean(r) for r in summarize_money(results).to_dict(orient="records")]}
    for row in summarize_money(results, ["season"]).to_dict(orient="records"):
        out.setdefault(str(row["season"]), []).append(_clean(row))
    return out


def evaluation_payload(
    backtest: pd.DataFrame | None = None,
    tickets: pd.DataFrame | None = None,
    money_results: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Collect the data shown in the evaluation (backtest) dashboard.

    Args:
        backtest: Backtest predictions with real results. Defaults to the saved backtest.
        tickets: Backtest tickets with doubles and triples. Defaults to the saved ones.
        money_results: Real tickets played in euros. Defaults to the saved ones.

    Returns:
        JSON-serialisable dictionary with the models, metric definitions,
        metrics for every league/season combination, calibration tables, hit
        rate by confidence tier and the prizes of each ticket strategy.
    """
    # Imported here: only this dashboard needs the evaluation package
    from quiniela.evaluation.backtest import (
        RANDOM_DRAWS,
        TICKET_STRATEGIES,
        load_backtest,
        load_tickets,
        summarize,
    )
    from quiniela.evaluation.metrics import calibration_table
    from quiniela.prediction import planner

    backtest = load_backtest() if backtest is None else backtest
    if backtest.empty:
        return {"models": [], "metrics": EVALUATION_METRICS, "seasons": [], "results": {}}
    tickets = load_tickets() if tickets is None else tickets
    if money_results is None:
        from quiniela.evaluation.money import load_money

        money_results = load_money()

    models = _model_specs(list(backtest["model"].unique()))
    seasons = sorted(backtest["season"].unique())

    # Metrics keyed by "<league>|<season>" ("all" for both leagues / every season)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for by in ([], ["season"], ["league"], ["season", "league"]):
        for row in summarize(backtest, by).to_dict(orient="records"):
            key = f"{row.get('league', 'all')}|{row.get('season', 'all')}"
            results.setdefault(key, {})[row["model"]] = _clean(row)

    # v4 planned ticket: only hit rate and prizes, right after the planner model
    planned = _planned_results(backtest, tickets)
    for key, row in planned.items():
        results.setdefault(key, {})[PLANNED_KEY] = row
    if planned:
        position = next(
            (i + 1 for i, m in enumerate(models) if m["key"] == planner.PLANNER_MODEL),
            len(models),
        )
        models.insert(position, PLANNED_SPEC)

    calibration: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for league in ["all", *sorted(backtest["league"].unique())]:
        subset = backtest if league == "all" else backtest[backtest["league"] == league]
        for model in subset["model"].unique():
            rows = subset[subset["model"] == model].dropna(subset=["p_H", "p_D", "p_A"])
            if rows.empty:
                continue
            table = calibration_table(rows[["p_H", "p_D", "p_A"]].to_numpy(), rows["FTR"])
            calibration.setdefault(league, {})[model] = [
                _clean(r) for r in table.to_dict(orient="records")
            ]

    return {
        "models": models,
        "metrics": EVALUATION_METRICS,
        "leagues": {"all": "Both leagues", **LEAGUE_NAMES},
        "seasons": seasons,
        "matches": int(backtest[backtest["model"] == backtest["model"].iloc[0]].shape[0]),
        "results": results,
        "calibration": calibration,
        "confidence": _confidence_tables(backtest),
        "tickets": _ticket_tables(tickets),
        "money": _money_tables(money_results),
        "money_strategies": _money_labels(),
        "ticket_strategies": {key: label for key, (label, _, _) in TICKET_STRATEGIES.items()},
        "ticket_budget": {
            "model": planner.PLANNER_MODEL,
            "doubles": planner.DEFAULT_DOUBLES,
            "triples": planner.DEFAULT_TRIPLES,
            "max_bets": planner.DEFAULT_MAX_BETS,
            "random_draws": RANDOM_DRAWS,
        },
    }


# %% RENDERING


def _template(name: str) -> str:
    """Read a file from the bundled templates folder."""
    return resources.files("quiniela.visualization").joinpath("templates", name).read_text("utf-8")


def render(template: str, payload: dict[str, Any], output: Path) -> Path:
    """Embed the shared stylesheet, helpers and the data into a template and write it.

    Args:
        template: Template file name inside ``visualization/templates``.
        payload: Data exposed to the page as ``DATA``.
        output: Destination HTML file.

    Returns:
        The path written.
    """
    # "</" would close the <script> tag if a value contained it
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (
        _template(template)
        .replace(STYLE_PLACEHOLDER, _template("base.css"))
        .replace(SCRIPT_PLACEHOLDER, _template("base.js"))
        # Data last, so placeholder-like text inside the data is never replaced
        .replace(DATA_PLACEHOLDER, data)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s (%.1f KB)", output, output.stat().st_size / 1024)
    return output


# Dashboards of earlier versions, deleted when the dashboards are rebuilt
RETIRED_DASHBOARDS = ["predictions.html"]


def build_dashboards(output_dir: Path = DASHBOARDS_DIR) -> list[Path]:
    """Build every dashboard from the current data folders.

    Args:
        output_dir: Folder where the HTML files are written.

    Returns:
        Paths of the dashboards written.
    """
    for name in RETIRED_DASHBOARDS:
        (output_dir / name).unlink(missing_ok=True)
    # The season payload (with the v5 simulations) is built once for both pages
    season = season_payload()
    return [
        render("current.html", current_payload(season=season), output_dir / "current.html"),
        render("season.html", season, output_dir / "season.html"),
        render("historical.html", historical_payload(), output_dir / "historical.html"),
        render("evaluation.html", evaluation_payload(), output_dir / "evaluation.html"),
    ]
