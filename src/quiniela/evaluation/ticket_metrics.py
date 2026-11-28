r"""Metrics and KPIs of the official tickets: one record per match and one per draw.

The tickets plan their signs on the v3 probabilities $p$ (v3+v3F: v3F on the Liga F
matches), and their prizes depend on what the public marks $s$ (the share of bettors
marking each sign, saved from the betting page since jornada 21 of 2026-27). This
module gathers, for every match of every official ticket, what was known before the
draw, and for every draw what the tickets made of it, so that a person can decide
and the metrics can be judged by the KPIs they should predict.

**Per match** (``match_metrics``), with $k^\ast = \arg\max_k p_k$ the v3 pick:

- ``fav_prob`` $= p_{k^\ast}$ and ``entropy`` $= -\sum_k p_k \log p_k$: how open the
  match is (0.80 / 0.80 for a clear favourite, about 1.10 / 0.36 for a toss-up).
- ``strength``: pi-ratings (v4) home-win minus away-win probability, from $-1$ to
  $1$: the strength gap of the teams by results alone.
- ``edge_market`` $= \log(p_{k^\ast} / m_{k^\ast})$: how much more v3 believes its
  pick than the market $m$.
- ``edge_public`` $= \log(p_{k^\ast} / s_{k^\ast})$ and ``value`` $= \max_k p_k / s_k$
  (sign ``value_sign``): how much the public underrates a sign. Above 0 (above 1)
  the sign is less marked than it is likely, which is what makes a prize larger.
- ``spread`` $= \frac{1}{3}\sum_k \log(\text{Max}_k / \text{Avg}_k)$: how much the
  bookmakers disagree (informative only: it did not improve v3, see
  ``DISCARDED.todo``).
- ``hit`` and ``log_loss`` of v3 once the result is known.

**Per draw** (``draw_metrics``): the means of the match metrics over the matches
with a v3 probability, the expected hits of the single ticket $\sum p_{k^\ast}$, and,
when the public's shares of the 14 matches are known, the **differentiation** of
the single ticket $D = \sum_i \log(p_{i,k^\ast} / s_{i,k^\ast})$ and the other bets
expected to share its 14 hits, $\hat W_{14} = B \prod_i s_{i,k^\ast}$. The KPIs come
from the money backtest (``quiniela.evaluation.money``) for the past seasons and from
the season dashboard for the current one: cost, payout, net and best hits of every
ticket, the planned chances its traffic lights rank and the verdict.

**Verdict** (``verdicts``). A first rule built on the traffic lights, judged by the
KPIs of the draws it would have played: *big* (play v3+v3F·B) when its chance of a
prize of 1,000 € or more is green, *small* (play the single or the v3+v3F·D ticket)
when only the chance of 12+ hits of v3+v3F·D is green, *skip* otherwise. In the past
draws the lights rank each draw only among the draws before it (no look-ahead).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quiniela.config import METRICS_DIR
from quiniela.data.prizes import BET_PRICE

OUTCOMES = ["H", "D", "A"]

# Ticket keys of the dashboards and their strategies in the money backtest
TICKETS = {"v3s": "single", "v3d2": "v3d2", "v3d": "v3d", "v3r": "v3r", "v3b": "v3b"}

# Traffic light thresholds (percentiles) and fewest earlier draws to rank against
GREEN_AT = 75
RED_AT = 25
MIN_EARLIER = 20

# Fewest La Liga / Segunda matches for a verdict (as the money backtest): with fewer the
# planned chances rest on the league shares of 1, X and 2, not on the models
MIN_MATCHED = 10

# Columns of the match records
MATCH_KEY = ["season", "jornada", "draw_id", "position"]
MATCH_COLUMNS = [
    *MATCH_KEY,
    "date",
    "league",
    "game",
    "home",
    "away",
    "category",
    "p_H",
    "p_D",
    "p_A",
    "m_H",
    "m_D",
    "m_A",
    "s_H",
    "s_D",
    "s_A",
    "pick",
    "fav_prob",
    "entropy",
    "strength",
    "edge_market",
    "public_pick",
    "edge_public",
    "value_sign",
    "value",
    "spread",
    "FTR",
    "hit",
    "value_hit",
    "log_loss",
]

# Name, label and meaning of every metric, shown by metrics.html
METRIC_INFO: dict[str, tuple[str, str]] = {
    "fav_prob": (
        "Favorito",
        "probabilidad del signo más probable (v3; v3F en la Liga F): más alta, partido más claro",
    ),
    "entropy": (
        "Apertura",
        "entropía de las probabilidades (v3; v3F en la Liga F): más alta, más abierto",
    ),
    "strength": (
        "Diferencia de fuerza",
        "probabilidad de que gane el local menos la de que gane el visitante según los pi-ratings (v4)",
    ),
    "edge_market": (
        "v3 frente al mercado",
        "logaritmo de la probabilidad de v3 de su signo dividida por la del mercado",
    ),
    "edge_public": (
        "v3 frente al público",
        "logaritmo de la probabilidad de v3 de su signo dividida por el porcentaje del público "
        "(por encima de 0: infravalorado)",
    ),
    "value": (
        "Mejor valor",
        "mayor probabilidad de v3 / porcentaje del público entre los tres signos",
    ),
    "spread": ("Dispersión de cuotas", "media del logaritmo de la cuota más alta sobre la media"),
    "expected_hits": (
        "Aciertos esperados",
        "aciertos que espera el boleto sencillo (suma de sus probabilidades)",
    ),
    "differentiation": (
        "Diferenciación",
        "suma de las ventajas de v3 frente al público en los signos del boleto sencillo",
    ),
    "co_winners_14": (
        "Coacertantes si hay 14",
        "apuestas que se espera que compartan un 14 del boleto sencillo",
    ),
    "p_big_v3b": ("v3+v3F·B 1.000 €+", "probabilidad planificada de un premio de 1.000 € o más"),
    "p_hits_v3d": ("v3+v3F·D 12+", "probabilidad planificada de 12 o más aciertos"),
    "p_hits_v3d2": (
        "v3+v3F·D2 12+",
        "probabilidad planificada de 12 o más aciertos del boleto de 2 apuestas",
    ),
    "p_hits_v3s": (
        "Sencillo 12+",
        "probabilidad planificada de 12 o más aciertos del boleto sencillo",
    ),
    "expected_return": (
        "Retorno esperado",
        "premio esperado del v3+v3F·E dividido por su coste (prudente)",
    ),
    "jackpot": ("Bote", "bote anunciado para el Pleno al 15 (€)"),
}


def _probs(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """Probabilities ``<prefix>_H``, ``<prefix>_D``, ``<prefix>_A`` (``NaN`` when missing)."""
    columns = [f"{prefix}_{o}" for o in OUTCOMES]
    if not set(columns) <= set(frame.columns):
        return np.full((len(frame), 3), np.nan)
    values: np.ndarray = frame[columns].to_numpy(dtype=float)
    return values


def _odds(frame: pd.DataFrame, prefixes: tuple[str, str]) -> np.ndarray:
    """1X2 odds of the first prefix available in each row (``NaN`` without any)."""
    out = np.full((len(frame), 3), np.nan)
    for prefix in prefixes:
        columns = [f"{prefix}{o}" for o in OUTCOMES]
        if not set(columns) <= set(frame.columns):
            continue
        odds = frame[columns].to_numpy(dtype=float)
        usable = np.isnan(out).any(axis=1) & np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
        out[usable] = odds[usable]
    return out


def _signs(index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Signs ``H``, ``D``, ``A`` of outcome indices, ``None`` where ``mask`` is false."""
    signs: np.ndarray = np.array(OUTCOMES, dtype=object)[index]
    signs[~mask] = None
    return signs


def match_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the metrics of every match.

    Args:
        frame: One row per ticket match with the v3 probabilities ``p_H``, ``p_D``,
            ``p_A`` and, when known, the market's (``m_*``), the public's shares
            (``s_*``, adding up to 1), the pi-ratings (``v4_H``, ``v4_A``), the
            average and highest odds (``AvgH``... / ``BbAvH``..., ``MaxH``... /
            ``BbMxH``...) and the result ``FTR``.

    Returns:
        ``frame`` with ``pick``, ``fav_prob``, ``entropy``, ``strength``,
        ``edge_market``, ``public_pick``, ``edge_public``, ``value_sign``,
        ``value``, ``spread``, ``hit``, ``value_hit`` (the best-value sign came up)
        and ``log_loss`` (``NaN`` where an input is missing).
    """
    out = frame.copy()
    p, m, s = _probs(out, "p"), _probs(out, "m"), _probs(out, "s")
    known = ~np.isnan(p).any(axis=1)
    rows = np.arange(len(out))
    pick = np.where(known, np.nan_to_num(p, nan=-1.0).argmax(axis=1), 0)
    out["pick"] = _signs(pick, known)
    fav = np.where(known, p[rows, pick], np.nan)
    out["fav_prob"] = fav
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.clip(p, 1e-12, 1.0))
        out["entropy"] = np.where(known, -(p * logs).sum(axis=1), np.nan)
        v4_home = out["v4_H"].to_numpy(dtype=float) if "v4_H" in out else np.full(len(out), np.nan)
        v4_away = out["v4_A"].to_numpy(dtype=float) if "v4_A" in out else np.full(len(out), np.nan)
        out["strength"] = v4_home - v4_away
        out["edge_market"] = np.where(known, np.log(fav / m[rows, pick]), np.nan)
        public = ~np.isnan(s).any(axis=1)
        public_pick = np.nan_to_num(s, nan=-1.0).argmax(axis=1)
        out["public_pick"] = _signs(public_pick, public)
        both = known & public
        out["edge_public"] = np.where(both, np.log(fav / s[rows, pick]), np.nan)
        ratio = p / np.clip(s, 1e-6, None)
        best = np.nan_to_num(ratio, nan=-1.0).argmax(axis=1)
        out["value_sign"] = _signs(best, both)
        out["value"] = np.where(both, ratio[rows, best], np.nan)
        average, highest = _odds(out, ("Avg", "BbAv")), _odds(out, ("Max", "BbMx"))
        out["spread"] = np.log(highest / average).mean(axis=1)
    result = out["FTR"] if "FTR" in out else pd.Series(None, index=out.index, dtype=object)
    played = known & result.isin(OUTCOMES).to_numpy()
    index = np.array([OUTCOMES.index(r) if r in OUTCOMES else 0 for r in result])
    out["hit"] = np.where(played, out["pick"].to_numpy() == result.to_numpy(), np.nan)
    # Whether the best-value sign (the most underrated by the public) came up
    valued = played & out["value_sign"].notna().to_numpy()
    out["value_hit"] = np.where(valued, out["value_sign"].to_numpy() == result.to_numpy(), np.nan)
    with np.errstate(divide="ignore"):
        out["log_loss"] = np.where(played, -np.log(np.clip(p[rows, index], 1e-12, 1.0)), np.nan)
    return out


def _join_probs(
    rows: pd.DataFrame,
    frame: pd.DataFrame | None,
    prefix: str,
    columns: tuple[str, ...] = ("H", "D", "A"),
    on: tuple[str, ...] = ("league", "game"),
) -> pd.DataFrame:
    """Join one model's probabilities (``p_H``...) by league and game, renamed ``<prefix>_*``."""
    names = {f"p_{o}": f"{prefix}_{o}" for o in columns}
    if frame is None or frame.empty:
        return rows.assign(**{name: np.nan for name in names.values()})
    keys = list(on)
    probs = frame[[*keys, *names]].drop_duplicates(keys, keep="last")
    return rows.merge(probs.rename(columns=names), on=keys, how="left")


def ticket_matches(
    matched: pd.DataFrame,
    models: Mapping[str, pd.DataFrame | None],
    odds: pd.DataFrame,
    shares: Mapping[tuple[str, int], np.ndarray] | None = None,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """Match records of the official tickets (positions 1 to 14).

    Args:
        matched: Ticket matches linked to football-data
            (``quiniela.data.prizes.match_tickets``).
        models: Probabilities (``league``, ``game``, ``p_H``, ``p_D``, ``p_A``) of
            ``"p"`` (v3), ``"m"`` (the market) and ``"v4"`` (pi-ratings).
        odds: football-data matches with ``league``, ``game`` and the average and
            highest odds (``quiniela.data.loading``).
        shares: Public shares of the 14 matches by ``(season, jornada)``
            (``quiniela.data.women.load_public_shares``).
        seasons: Seasons to keep (all by default).

    Returns:
        ``MATCH_COLUMNS``, one row per ticket match.
    """
    from quiniela.data.prizes import match_category

    rows = matched[matched["position"] <= 14].copy()
    if seasons is not None:
        rows = rows[rows["season"].isin(seasons)]
    rows = rows.rename(columns={"sign": "FTR", "lae_home": "home", "lae_away": "away"})
    rows["date"] = pd.to_datetime(rows["kickoff"]).fillna(rows["draw_date"])
    rows["category"] = [
        match_category(h, a, lg)
        for h, a, lg in zip(rows["home"], rows["away"], rows["league"], strict=True)
    ]
    rows = _join_probs(rows, models.get("p"), "p")
    rows = _join_probs(rows, models.get("m"), "m")
    rows = _join_probs(rows, models.get("v4"), "v4", ("H", "A"))
    odd_columns = [
        c for c in odds.columns if c[:-1] in ("Avg", "BbAv", "Max", "BbMx") and c[-1] in "HDA"
    ]
    rows = rows.merge(
        odds[["league", "game", *odd_columns]].drop_duplicates(["league", "game"]),
        on=["league", "game"],
        how="left",
    )
    for outcome in OUTCOMES:
        rows[f"s_{outcome}"] = np.nan
    for (season, jornada), values in (shares or {}).items():
        mask = (rows["season"] == season) & (rows["jornada"] == jornada)
        positions = rows.loc[mask, "position"].astype(int).to_numpy() - 1
        rows.loc[mask, ["s_H", "s_D", "s_A"]] = values[positions]
    # Matches no model predicts (Liga F, other competitions): the public's picks, as the
    # tickets' fallback, so that a ticket has a pick in all its 14 matches
    missing = rows[["p_H", "p_D", "p_A"]].isna().any(axis=1) & rows[
        ["s_H", "s_D", "s_A"]
    ].notna().all(axis=1)
    rows.loc[missing, ["p_H", "p_D", "p_A"]] = rows.loc[missing, ["s_H", "s_D", "s_A"]].to_numpy()
    return match_metrics(rows).reindex(columns=MATCH_COLUMNS)


def upcoming_matches(
    matches: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    planner_key: str,
    season: str,
    jornada: int,
) -> pd.DataFrame:
    """Match records of the next official ticket (no result yet).

    Args:
        matches: Ticket matches of ``quiniela.prediction.pipeline.predict_upcoming``
            (``official_position``, ``league``, ``game``, teams, odds and the
            public's shares ``_public_*`` in percent).
        predictions: Its predictions per model key.
        planner_key: Model whose probabilities the tickets plan on (v3, with v3F
            and the public's picks filled in on the other matches).
        season: Season of the ticket.
        jornada: Its jornada.

    Returns:
        ``MATCH_COLUMNS`` of the 14 matches (``draw_id`` empty).
    """
    from quiniela.prediction.pipeline import OTHER_LEAGUE, WOMEN_LEAGUE

    rows = matches[matches["official_position"].between(1, 14)].copy()
    rows["position"] = rows["official_position"].astype(int)
    rows = rows.rename(columns={"home_team": "home", "away_team": "away"})
    rows = rows.assign(season=season, jornada=jornada, draw_id=None)
    rows["category"] = rows["league"].map({WOMEN_LEAGUE: "F", OTHER_LEAGUE: "O"}).fillna("M")
    rows = _join_probs(rows, predictions.get(planner_key), "p", on=("game",))
    rows = _join_probs(rows, predictions.get("market"), "m", on=("game",))
    rows = _join_probs(rows, predictions.get("pi_ratings"), "v4", ("H", "A"), on=("game",))
    public = [f"_public_{o}" for o in OUTCOMES]
    if set(public) <= set(rows.columns):
        shares = rows[public].to_numpy(dtype=float)
        totals = shares.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows[["s_H", "s_D", "s_A"]] = np.where(totals > 0, shares / totals, np.nan)
    rows = rows.sort_values("position", ignore_index=True)
    return match_metrics(rows).reindex(columns=MATCH_COLUMNS)


def add_kpis(draws: pd.DataFrame) -> pd.DataFrame:
    """Add the KPIs derived from the cost, payout and hits of every ticket.

    Args:
        draws: Draws with ``cost_*``, ``payout_*`` and ``hits_*`` of the ticket keys.

    Returns:
        ``draws`` with ``return_*`` (payout over cost), ``hit12_*`` and
        ``hit13_*`` (best column with 12 or 13+ hits) and ``big_v3b``
        (v3+v3F·B paid 1,000 € or more); ``NaN`` while a draw is not complete.
    """
    out = draws.copy()
    for key in TICKETS:
        cost = pd.to_numeric(out.get(f"cost_{key}"), errors="coerce")
        payout = pd.to_numeric(out.get(f"payout_{key}"), errors="coerce")
        hits = pd.to_numeric(out.get(f"hits_{key}"), errors="coerce")
        done = payout.notna()
        out[f"return_{key}"] = payout / cost
        out[f"hit12_{key}"] = (hits >= 12).astype(float).where(done)
        out[f"hit13_{key}"] = (hits >= 13).astype(float).where(done)
    payout_b = pd.to_numeric(out.get("payout_v3b"), errors="coerce")
    out["big_v3b"] = (payout_b >= 1_000).astype(float).where(payout_b.notna())
    return out


def draw_metrics(
    matches: pd.DataFrame, expected_bets: Mapping[str, float] | None = None
) -> pd.DataFrame:
    """Metrics of every draw from its match records.

    Args:
        matches: Output of :func:`ticket_matches` (or :func:`match_metrics` with ``MATCH_KEY``).
        expected_bets: Bets expected before each draw, by ``draw_id`` (for the
            co-winners of a 14).

    Returns:
        One row per draw: ``matched`` (La Liga / Segunda matches with a v3
        probability), the mean ``fav_prob``, ``entropy``, ``spread``,
        ``edge_market`` and ``edge_public``, ``expected_hits`` (``NaN`` without
        any probability), ``public`` (matches with the public's shares),
        ``differentiation`` and ``co_winners_14``.
    """
    rows = []
    for key, group in matches.groupby(["season", "jornada", "draw_id"], sort=False, dropna=False):
        season, jornada, draw_id = key
        known = group["fav_prob"].notna()
        men = known & (group["category"] == "M")
        public = group["edge_public"].notna()
        full = len(group) == 14 and bool(public.all())
        picked = []
        if full:
            for _, row in group.iterrows():
                picked.append(float(row[f"s_{row['pick']}"]))
        bets = (expected_bets or {}).get(str(draw_id))
        rows.append(
            {
                "season": season,
                "jornada": jornada,
                "draw_id": draw_id,
                "date": group["date"].min(),
                "matched": int(men.sum()),
                "fav_prob": group["fav_prob"].mean(),
                "entropy": group["entropy"].mean(),
                "spread": group["spread"].mean(),
                "edge_market": group["edge_market"].mean(),
                "edge_public": group["edge_public"].mean(),
                "expected_hits": float(group.loc[known, "fav_prob"].sum())
                if known.any()
                else np.nan,
                "public": int(public.sum()),
                "differentiation": float(group["edge_public"].sum()) if full else np.nan,
                "co_winners_14": float(bets * np.prod(picked)) if full and bets else np.nan,
            }
        )
    return pd.DataFrame(rows)


def money_kpis(money: pd.DataFrame) -> pd.DataFrame:
    """KPIs of every ticket in every draw of the money backtest, one row per draw.

    Args:
        money: Output of ``quiniela.evaluation.money.load_money``.

    Returns:
        ``draw_id``, ``jackpot`` and, for every ticket key ``v3s``, ``v3d2``,
        ``v3d``, ``v3r`` and ``v3b``: ``cost_*``, ``payout_*``, ``net_*``, ``hits_*``
        (best column) and the planned chance its light ranks (``p_hits_*`` of 12+
        hits, ``p_big_v3b`` of 1,000 € or more); ``expected_return`` of v3+v3F·E.
    """
    if money.empty:
        return pd.DataFrame(columns=["draw_id"])
    out = money.drop_duplicates("draw_id")[["draw_id", "jackpot"]].set_index("draw_id")
    for key, strategy in TICKETS.items():
        part = money[money["strategy"] == strategy].set_index("draw_id")
        out[f"cost_{key}"] = part["cost"]
        out[f"payout_{key}"] = part["payout"]
        out[f"net_{key}"] = part["net"]
        out[f"hits_{key}"] = part["best_hits"]
        if key == "v3b":
            out["p_big_v3b"] = part["p_big"]
        else:
            out[f"p_hits_{key}"] = part["p_hits"]
    prudent = money[money["strategy"] == "v3e"].set_index("draw_id")
    # Price of the ticket it planned (its cost is 0 in the draws it does not play)
    price = BET_PRICE * prudent["bets"].where(prudent["bets"] > 0)
    out["expected_return"] = (prudent["expected_net"] + price) / price
    return out.reset_index()


def _percentile_light(value: float, earlier: np.ndarray) -> str | None:
    """Traffic light of a value among earlier values (None with too few)."""
    earlier = earlier[np.isfinite(earlier)]
    if not np.isfinite(value) or len(earlier) < MIN_EARLIER:
        return None
    percentile = float((earlier < value).mean() * 100)
    return "green" if percentile >= GREEN_AT else "red" if percentile <= RED_AT else "amber"


def walk_forward_lights(draws: pd.DataFrame, column: str) -> list[str | None]:
    """Light of every draw by its ``column``, ranked only among the draws before it.

    Args:
        draws: Draws with ``date`` and ``column``.
        column: Planned chance to rank.

    Returns:
        One light per row of ``draws`` (in its order).
    """
    order = np.argsort(draws["date"].to_numpy(), kind="stable")
    values = draws[column].to_numpy(dtype=float)
    lights: list[str | None] = [None] * len(draws)
    for rank, i in enumerate(order):
        lights[i] = _percentile_light(values[i], values[order[:rank]])
    return lights


def verdicts(
    light_big: list[str | None], light_hits: list[str | None], matched: list[int] | None = None
) -> list[str | None]:
    """Verdict of every draw from the v3+v3F·B and v3+v3F·D lights.

    Args:
        light_big: Light of the chance of 1,000 € or more of v3+v3F·B.
        light_hits: Light of the chance of 12+ hits of v3+v3F·D.
        matched: La Liga / Segunda matches of every draw (no verdict below
            ``MIN_MATCHED``); all draws count by default.

    Returns:
        ``"big"`` (play v3+v3F·B), ``"small"`` (single or v3+v3F·D ticket),
        ``"skip"``, or ``None`` without lights or with too few league matches.
    """
    counts = matched if matched is not None else [MIN_MATCHED] * len(light_big)
    out: list[str | None] = []
    for big, hits, n in zip(light_big, light_hits, counts, strict=True):
        if (big is None and hits is None) or n < MIN_MATCHED:
            out.append(None)
        elif big == "green":
            out.append("big")
        elif hits == "green":
            out.append("small")
        else:
            out.append("skip")
    return out


def past_draws(draw_rows: pd.DataFrame, money: pd.DataFrame) -> pd.DataFrame:
    """Draws of the money backtest with their metrics, KPIs, lights and verdict.

    Args:
        draw_rows: Output of :func:`draw_metrics`.
        money: Output of ``quiniela.evaluation.money.load_money``.

    Returns:
        One row per draw played in the money backtest, oldest first.
    """
    kpis = money_kpis(money)
    if kpis.empty:
        return kpis
    out = draw_rows.merge(kpis, on="draw_id", how="inner").sort_values("date", ignore_index=True)
    out["light_v3b"] = walk_forward_lights(out, "p_big_v3b")
    out["light_v3d"] = walk_forward_lights(out, "p_hits_v3d")
    out["light_v3d2"] = walk_forward_lights(out, "p_hits_v3d2")
    out["verdict"] = verdicts(list(out["light_v3b"]), list(out["light_v3d"]))
    return add_kpis(out)


def season_draws(tickets: list[dict[str, Any]], draw_rows: pd.DataFrame) -> pd.DataFrame:
    """Draws of the current season from the season dashboard's official tickets.

    Args:
        tickets: ``tickets`` of the season payload, with the ``outcomes`` (cost,
            payout, hits, planned ``value`` and ``signal``) of every ticket key.
        draw_rows: Output of :func:`draw_metrics` for the same draws.

    Returns:
        One row per ticket with the columns of :func:`past_draws` (payouts are
        ``NaN`` until the draw is complete).
    """
    rows = []
    for ticket in tickets:
        row: dict[str, Any] = {"draw_id": str(ticket["draw_id"])}
        for key in TICKETS:
            outcome = (ticket.get("outcomes") or {}).get(key) or {}
            row[f"cost_{key}"] = outcome.get("cost")
            row[f"payout_{key}"] = outcome.get("payout")
            payout, cost = outcome.get("payout"), outcome.get("cost")
            row[f"net_{key}"] = payout - cost if payout is not None and cost is not None else None
            row[f"hits_{key}"] = outcome.get("hits")
            value_column = "p_big_v3b" if key == "v3b" else f"p_hits_{key}"
            row[value_column] = outcome.get("value")
            if key in ("v3b", "v3d", "v3d2"):
                row[f"light_{key}"] = outcome.get("signal")
        row["winners"] = ticket.get("winners")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = draw_rows.merge(pd.DataFrame(rows), on="draw_id", how="right")
    matched = [int(m) if pd.notna(m) else 0 for m in out["matched"]]
    out["verdict"] = verdicts(list(out["light_v3b"]), list(out["light_v3d"]), matched)
    return add_kpis(out.sort_values("date", ignore_index=True))


def quintiles(
    frame: pd.DataFrame, metric: str, kpis: Mapping[str, str], bins: int = 5
) -> list[dict[str, Any]]:
    """Mean of every KPI by quintile of a metric.

    Args:
        frame: Records with the metric and the KPIs.
        metric: Column to split by.
        kpis: Output name -> column averaged in every quintile.
        bins: Number of groups (5: quintiles).

    Returns:
        One dict per group, lowest first: ``group``, ``n``, ``low``, ``high``,
        ``mean`` of the metric and the mean of every KPI.
    """
    data = frame.dropna(subset=[metric])
    if len(data) < bins * 5:
        return []
    groups = pd.qcut(data[metric].rank(method="first"), bins, labels=False)
    out = []
    for group, part in data.groupby(groups):
        row: dict[str, Any] = {
            "group": int(group) + 1,
            "n": len(part),
            "low": float(part[metric].min()),
            "high": float(part[metric].max()),
            "mean": float(part[metric].mean()),
        }
        for name, column in kpis.items():
            values = pd.to_numeric(part[column], errors="coerce")
            row[name] = float(values.mean()) if values.notna().any() else None
        out.append(row)
    return out


def save_metrics(
    matches: pd.DataFrame, draws: pd.DataFrame, metrics_dir: Path = METRICS_DIR
) -> list[Path]:
    """Write the match and draw records (``matches.parquet``, ``draws.parquet``).

    Args:
        matches: Match records.
        draws: Draw records.
        metrics_dir: Destination folder.

    Returns:
        The files written.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in (("matches", matches), ("draws", draws)):
        path = metrics_dir / f"{name}.parquet"
        clean = frame.copy()
        for column in clean.columns:
            if clean[column].dtype == object:
                clean[column] = clean[column].astype("string")
        clean.to_parquet(path, index=False)
        paths.append(path)
    return paths
