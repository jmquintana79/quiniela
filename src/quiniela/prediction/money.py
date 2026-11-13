r"""Money planner (v5): the ticket with the highest expected prize in EUR, when it pays.

v4 (``quiniela.prediction.planner``) maximises the chance of a prize. But a
prize is shared among its winners, and the quiniela gives back only 55% of the
takings, so on average every bet loses money. v5 looks at euros instead: it
plays only when the expected prize is larger than the cost of the ticket, which
happens with a jackpot or when results the public does not expect are likely.

**Prize of a category.** With $B$ bets in the draw (takings $0.75\,B$), a
share $s_k$ of the takings for $k$ hits (16% for 14, 7.5% for 13 to 11, 9% for
10) and $W_k$ winners, each winning column gets

$$
\text{prize}_k = \frac{s_k \cdot 0.75\,B}{W_k + 1}
$$

(the $+1$ is our own ticket); the 10-hit prize is not paid below 1.50 EUR. The
Pleno al 15 (14 hits and the 15th match) gets 7.5% plus the announced jackpot.

**How many winners: the surprise of the results.** The public mostly marks
favourites, so when favourites win there are many winners and small prizes,
and after surprises few winners share large prizes. With the market
probabilities $m_i$ of the real results $r_i$ of the 14 matches, the surprise
of a draw is

$$
S = -\sum_{i=1}^{14} \log m_{i, r_i}
$$

and the share of winning bets of every category follows it closely (fitted on
past draws, ``PrizeModel``):

$$
\log \frac{W_k + 0.5}{B} = a_k + b_k\, S + \varepsilon_k, \qquad \varepsilon_k \sim N(0, \sigma_k^2)
$$

The Pleno al 15 winners are a fixed share of the 14-hit winners.

**Expected prize of a ticket.** Results are simulated from the model
probabilities (v3); each simulation gives the hits of every column of the
ticket, its surprise $S$ (from the market probabilities, the stand-in for the
public) and so the prizes. A ticket with $d$ doubles and $t$ triples has
$2^d 3^t$ columns: if $h$ singles and $d_h$ doubles are right, the number of
columns with $h + j$ hits is the coefficient of $x^j$ in
$2^{d - d_h} (x + 1)^{d_h} (x + 2)^t$, so lower categories also pay. The
expected net prize is the average payout minus $0.75 \cdot 2^d 3^t$.

**Plan.** For every mix of doubles and triples within the budget (at least two
bets, the minimum ticket), the placements with the highest chance of 11+
hits are evaluated by simulation, and the best expected net prize is kept.
Picking the best of many tickets on the same simulations would overestimate
its prize (rare 14-hit prizes make the estimates noisy), so the ticket is
chosen on one set of simulations and valued on another. v5 recommends playing
only when that value is above the cost.

**Big-win mode** (``plan_big_win``). Instead of the expected prize, the
chance of a large prize (10,000 EUR or more by default) within a weekly budget
(50 EUR): many hits and few other winners at once. It accepts losing on
average for that chance, and shows the whole payout distribution.

**Prudent verdict.** Most of the expected prize of a ticket comes from 14 hits
and the Pleno al 15 (a few times per thousand draws): a lottery the backtest
cannot check. By default (``top_weight = 0``) only the 10 to 13-hit prizes
count when choosing the ticket and deciding to play it; ``top_weight = 1``
counts everything (v5 until it was made prudent).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quiniela.data.prizes import (
    BET_PRICE,
    MIN_10_HIT_PRIZE,
    PRIZE_SHARES,
    SPECIAL_SHARE,
)
from quiniela.prediction.planner import (
    TicketPlan,
    _build,
    _placements,
    _sorted_signs,
    compositions,
    hit_distribution,
)

# Prize categories (hits), best first
CATEGORY_HITS = [14, 13, 12, 11, 10]

# Smallest ticket that can be played (two columns, 1.50 EUR)
MIN_BETS = 2

# Default budget, simulations and candidates per mix
DEFAULT_MAX_BETS = 48
DEFAULT_SIMULATIONS = 4000
DEFAULT_CANDIDATES = 60

# Chance of hitting the Pleno al 15 with the Dixon-Coles pick (backtest hit rate)
PLENO_HIT_RATE = 0.145

# Weight of the 14-hit and Pleno al 15 prizes in the decision (prudent: none). They
# are a few-per-thousand lottery whose value the backtest cannot check, and the
# 10 to 13-hit prizes alone never paid the cost of a ticket in 11 seasons
PRUDENT_TOP_WEIGHT = 0.0

# Big-win mode: at most 50 EUR a matchday (66 bets) and a prize of 10,000 EUR or more
BIG_WIN_MAX_COST = 50.0
BIG_WIN_THRESHOLD = 10_000.0
BIG_WIN_SIMULATIONS = 8000

# Payout ranges of the big-win distribution (EUR); the first two depend on the cost
PAYOUT_EDGES = [1_000.0, 10_000.0, 100_000.0]

# Smallest probability inside the logarithm of the surprise
PROB_FLOOR = 1e-6


def surprise(crowd: np.ndarray, results: np.ndarray) -> np.ndarray:
    """Surprise of results: minus the summed log market probability of each result.

    Args:
        crowd: Market (public) probabilities, shape ``(n_matches, 3)``.
        results: Outcome index (0 ``H``, 1 ``D``, 2 ``A``) of every match,
            shape ``(n_matches,)`` or ``(n_simulations, n_matches)``.

    Returns:
        The surprise of each set of results.
    """
    results = np.atleast_2d(results)
    picked = np.take_along_axis(np.broadcast_to(crowd, (*results.shape, 3)), results[..., None], 2)
    value: np.ndarray = -np.log(np.clip(picked[..., 0], PROB_FLOOR, 1.0)).sum(axis=1)
    return value


@dataclass(frozen=True)
class PrizeModel:
    """Share of winning bets of every category as a function of the surprise.

    Attributes:
        intercept: $a_k$ by hits.
        slope: $b_k$ by hits.
        sd: Residual standard deviation $\\sigma_k$ by hits.
        special_share: Pleno al 15 winners per 14-hit winner.
        draws: Draws the model was fitted on.
    """

    intercept: dict[int, float]
    slope: dict[int, float]
    sd: dict[int, float]
    special_share: float
    draws: int

    @classmethod
    def fit(cls, draws: pd.DataFrame) -> "PrizeModel":
        """Fit the winners of every category against the surprise of the draws.

        Args:
            draws: Past draws with ``surprise``, ``bets``, ``winners_14`` ...
                ``winners_10`` and ``winners_special``.

        Returns:
            The fitted model.

        Raises:
            ValueError: With fewer than 20 usable draws.
        """
        usable = draws[(draws["bets"] > 0) & np.isfinite(draws["surprise"])]
        usable = usable.dropna(subset=[f"winners_{k}" for k in CATEGORY_HITS])
        if len(usable) < 20:
            raise ValueError("At least 20 past draws are needed to fit the prize model")
        intercept, slope, sd = {}, {}, {}
        for k in CATEGORY_HITS:
            y = np.log((usable[f"winners_{k}"] + 0.5) / usable["bets"]).to_numpy()
            b, a = np.polyfit(usable["surprise"].to_numpy(), y, 1)
            intercept[k], slope[k] = float(a), float(b)
            sd[k] = float(np.std(y - (a + b * usable["surprise"].to_numpy())))
        special = usable["winners_special"].fillna(0).sum() / max(usable["winners_14"].sum(), 1)
        return cls(intercept, slope, sd, float(special), len(usable))

    def winners(
        self, surprise: np.ndarray, bets: float, rng: np.random.Generator | None = None
    ) -> dict[int, np.ndarray]:
        """Winners of every category for each surprise (with noise if ``rng`` is given)."""
        out = {}
        for k in CATEGORY_HITS:
            log_share = self.intercept[k] + self.slope[k] * surprise
            if rng is not None:
                log_share = log_share + rng.normal(0, self.sd[k], size=np.shape(surprise))
            out[k] = np.maximum(bets * np.exp(log_share) - 0.5, 0.0)
        return out

    def prizes(
        self,
        surprise: np.ndarray,
        bets: float,
        jackpot: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prize per winning column of every number of hits, and of the Pleno al 15.

        Args:
            surprise: Surprise of each simulated draw.
            bets: Bets of the draw (all players).
            jackpot: Jackpot announced for the Pleno al 15.
            rng: Random generator for the winners noise (none: the expected share).

        Returns:
            ``(prizes, special)``: shape ``(n, 15)`` prize for 0..14 hits
            (0 below 10) and shape ``(n,)`` prize of the Pleno al 15.
        """
        winners = self.winners(np.asarray(surprise, dtype=float), bets, rng)
        takings = BET_PRICE * bets
        prizes = np.zeros((np.size(surprise), 15))
        for k in CATEGORY_HITS:
            prizes[:, k] = PRIZE_SHARES[k] * takings / (winners[k] + 1)
        prizes[prizes[:, 10] < MIN_10_HIT_PRIZE, 10] = 0.0
        special = (SPECIAL_SHARE * takings + jackpot) / (winners[14] * self.special_share + 1)
        return prizes, special


def _column_coefficients(doubles_hit: int, triples: int) -> np.ndarray:
    """Coefficients of $(x + 1)^{d_h} (x + 2)^t$, lowest power first."""
    poly = np.array([1.0])
    for _ in range(doubles_hit):
        poly = np.convolve(poly, [1.0, 1.0])
    for _ in range(triples):
        poly = np.convolve(poly, [2.0, 1.0])
    return poly


def column_hits(levels: np.ndarray, rank_hit: np.ndarray) -> np.ndarray:
    """Number of columns of a ticket with every number of hits.

    Args:
        levels: Coverage level of each match (0 single, 1 double, 2 triple),
            shape ``(n_tickets, n)`` or ``(n,)``.
        rank_hit: Rank of the real result among each match's signs by model
            probability (0 the most likely), shape ``(n_draws, n)`` or ``(n,)``.

    Returns:
        Shape ``(n_tickets, n_draws, n + 1)``: columns with 0..n hits.
    """
    levels = np.atleast_2d(levels)
    rank_hit = np.atleast_2d(rank_hit)
    n = levels.shape[1]
    hit = rank_hit[None, :, :] <= levels[:, None, :]
    singles = ((levels == 0)[:, None, :] & hit).sum(axis=2)
    doubles_hit = ((levels == 1)[:, None, :] & hit).sum(axis=2)
    doubles = (levels == 1).sum(axis=1)[:, None]
    triples = (levels == 2).sum(axis=1)[:, None]
    out = np.zeros((*singles.shape, n + 1))
    for t in np.unique(triples):
        for dh in np.unique(doubles_hit):
            mask = (triples == t) & (doubles_hit == dh)
            if not mask.any():
                continue
            coef = _column_coefficients(int(dh), int(t))
            scale = 2.0 ** (np.broadcast_to(doubles, mask.shape)[mask] - dh)
            base = singles[mask]
            for j, c in enumerate(coef):
                np.add.at(out, (*np.nonzero(mask), base + j), c * scale)
    return out


@dataclass(frozen=True)
class MoneyPlan:
    """The ticket with the best expected net prize.

    Attributes:
        plan: The ticket (signs, doubles and triples).
        cost: Price of the ticket in EUR.
        expected_payout: Average prize in EUR.
        expected_net: ``expected_payout - cost``.
        prize_chance: Chance of any prize (some column with 10+ hits).
        worth: True when it is worth playing: the expected prize, with the
            14-hit and Pleno al 15 part weighted by ``top_weight``, is above the cost.
        expected_surprise: Average surprise of the simulated results.
        expected_top: Part of ``expected_payout`` from 14 hits and the Pleno
            al 15: rare (a few times per thousand draws) but large, so most of
            the time the ticket wins only ``expected_payout - expected_top``.
    """

    plan: TicketPlan
    cost: float
    expected_payout: float
    expected_net: float
    prize_chance: float
    worth: bool
    expected_surprise: float
    expected_top: float = 0.0
    top_weight: float = PRUDENT_TOP_WEIGHT

    @property
    def expected_common(self) -> float:
        """Expected prize from 10 to 13 hits (the prizes that do come up)."""
        return self.expected_payout - self.expected_top

    @property
    def roi(self) -> float:
        """Expected net prize per euro spent."""
        return self.expected_net / self.cost


def _candidates(
    probs: np.ndarray, max_bets: int, per_mix: int, min_hits: int = 11
) -> list[tuple[int, int, np.ndarray]]:
    """Placements worth simulating: the most likely to reach ``min_hits``, for every mix."""
    _, cover = _sorted_signs(probs)
    n = len(probs)
    rows = np.arange(n)
    out = []
    for d, t in compositions(max_bets, n):
        if 2**d * 3**t < MIN_BETS:
            continue
        placements = _placements(n, d, t)
        chance = hit_distribution(cover[rows, placements])[:, min_hits:].sum(axis=1)
        best = np.argsort(-chance, kind="stable")[:per_mix]
        out.append((d, t, placements[best]))
    return out


def plan_money(
    probs: np.ndarray,
    crowd: np.ndarray,
    model: PrizeModel,
    bets: float,
    jackpot: float = 0.0,
    max_bets: int = DEFAULT_MAX_BETS,
    simulations: int = DEFAULT_SIMULATIONS,
    per_mix: int = DEFAULT_CANDIDATES,
    pleno_rate: float = PLENO_HIT_RATE,
    top_weight: float = PRUDENT_TOP_WEIGHT,
    rng: np.random.Generator | None = None,
) -> MoneyPlan:
    """Ticket with the highest expected net prize within a budget.

    Args:
        probs: Model probabilities of the 14 ticket matches, shape ``(14, 3)``.
        crowd: Market probabilities of the same matches (the public).
        model: Fitted prize model.
        bets: Expected bets of the draw (all players).
        jackpot: Jackpot announced for the Pleno al 15.
        max_bets: Largest ticket, in bets.
        simulations: Simulated draws.
        per_mix: Placements evaluated per mix of doubles and triples.
        pleno_rate: Chance of also hitting the Pleno al 15.
        top_weight: Weight of the 14-hit and Pleno al 15 prizes when choosing
            the ticket and deciding whether to play it: 0 (prudent, default)
            counts only 10 to 13 hits, 1 the whole expected prize.
        rng: Random generator (seeded for reproducible results).

    Returns:
        The best ticket and its expected prize.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    probs = np.asarray(probs, dtype=float)
    order, _ = _sorted_signs(probs)
    rank = np.argsort(order, axis=1)  # rank[i, k]: position of sign k among match i's signs

    def simulate(noise: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ranks of the simulated results, their surprise and prizes."""
        n = len(probs)
        cumulative = np.cumsum(probs / probs.sum(axis=1, keepdims=True), axis=1)
        outcomes = np.minimum((rng.random((simulations, n, 1)) > cumulative[None]).sum(axis=2), 2)
        s = surprise(crowd, outcomes)
        prizes, special = model.prizes(s, bets, jackpot, rng if noise else None)
        return rank[np.arange(n)[None, :], outcomes], s, prizes, special

    def payouts(levels: np.ndarray, draws: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Payout of every ticket in every simulation, and its columns by hits."""
        rank_hit, _, prizes, special = draws
        columns = column_hits(levels, rank_hit)
        payout = (columns * prizes[None, :, :]).sum(axis=2)
        return payout + columns[:, :, 14] * pleno_rate * special[None, :], columns

    def top(columns: np.ndarray, draws: tuple[np.ndarray, ...]) -> float:
        """Expected payout from 14 hits and the Pleno al 15."""
        _, _, prizes, special = draws
        values = columns[0, :, 14] * (prizes[:, 14] + pleno_rate * special)
        return float(values.mean())

    # Choosing the best of many tickets on the same simulations overestimates its
    # prize (the winner's curse, strong with rare 14-hit prizes): the ticket is
    # chosen with typical winners (no noise) and valued on independent simulations
    choice = simulate(noise=False)
    best: tuple[float, int, int, np.ndarray] | None = None
    for d, t, placements in _candidates(probs, max_bets, per_mix):
        payout, columns = payouts(placements, choice)
        # Value used to choose: the 14-hit and Pleno al 15 part counts top_weight
        lottery = columns[:, :, 14] * (choice[2][None, :, 14] + pleno_rate * choice[3][None, :])
        value = payout - (1 - top_weight) * lottery
        net = value.mean(axis=1) - BET_PRICE * 2**d * 3**t
        i = int(np.argmax(net))
        if best is None or net[i] > best[0]:
            best = (float(net[i]), d, t, placements[i])
    assert best is not None
    _, d, t, levels = best
    # Typical winners (no noise) also for the valuation: with log-normal noise the
    # average prize of 11 to 13 hits came out 20-50% above the real one
    valuation = simulate(noise=False)
    payout, columns = payouts(levels[None, :], valuation)
    plan = _build(probs, levels, 11, d, t)
    cost = BET_PRICE * plan.bets
    expected = float(payout[0].mean())
    rare = top(columns, valuation)
    return MoneyPlan(
        plan=plan,
        cost=cost,
        expected_payout=expected,
        expected_net=expected - cost,
        prize_chance=float((columns[0, :, 10:].sum(axis=1) > 0).mean()),
        worth=expected - (1 - top_weight) * rare > cost,
        expected_surprise=float(valuation[1].mean()),
        expected_top=rare,
        top_weight=top_weight,
    )


@dataclass(frozen=True)
class BigWinPlan:
    """The ticket with the best chance of a large prize within a budget.

    Attributes:
        plan: The ticket (signs, doubles and triples).
        cost: Price of the ticket in EUR.
        threshold: Prize sought (EUR).
        chance: Chance of winning ``threshold`` or more in the matchday.
        chance_13: Chance that some column gets 13 or more hits.
        chance_14: Chance of 14 hits.
        expected_payout: Average prize in EUR (for reference: it is below the cost).
        outcomes: Chance of each payout range, as ``(label, low, high, chance)``
            from nothing back to 100,000 EUR or more.
    """

    plan: TicketPlan
    cost: float
    threshold: float
    chance: float
    chance_13: float
    chance_14: float
    expected_payout: float
    outcomes: list[tuple[str, float, float, float]]

    @property
    def matchdays_per_win(self) -> float:
        """Average matchdays between two prizes of ``threshold`` or more."""
        return 1 / self.chance if self.chance > 0 else float("inf")


def _payout_outcomes(payout: np.ndarray, cost: float) -> list[tuple[str, float, float, float]]:
    """Chance of each payout range of a ticket (from its simulated payouts)."""
    edges = [0.0, cost, *[e for e in PAYOUT_EDGES if e > cost], float("inf")]
    out = [("Nothing back", 0.0, 0.0, float((payout <= 0).mean()))]
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if low == 0.0:
            label = "Less than the cost"
            chance = float(((payout > 0) & (payout < high)).mean())
        else:
            label = f"{low:,.0f} € or more" if np.isinf(high) else f"{low:,.0f} to {high:,.0f} €"
            chance = float(((payout >= low) & (payout < high)).mean())
        out.append((label, low, high, chance))
    return out


def plan_big_win(
    probs: np.ndarray,
    crowd: np.ndarray,
    model: PrizeModel,
    bets: float,
    jackpot: float = 0.0,
    max_cost: float = BIG_WIN_MAX_COST,
    threshold: float = BIG_WIN_THRESHOLD,
    simulations: int = BIG_WIN_SIMULATIONS,
    per_mix: int = DEFAULT_CANDIDATES,
    pleno_rate: float = PLENO_HIT_RATE,
    rng: np.random.Generator | None = None,
) -> BigWinPlan:
    r"""Ticket with the highest chance of a large prize (big-win mode of v5).

    A large prize needs two things at once: many hits (13 or 14) and few other
    winners (results the public did not expect). Both are in one number, the
    chance that the ticket pays ``threshold`` or more:

    $$
    \max_{\text{ticket}} \; P(\text{payout} \ge \text{threshold}), \qquad 0.75 \cdot 2^d 3^t \le \text{max\_cost}
    $$

    The expected prize stays below the cost (the quiniela keeps 45%): this
    mode accepts losing on average for a chance of a large prize.

    Args:
        probs: Model probabilities of the 14 ticket matches, shape ``(14, 3)``.
        crowd: Market probabilities of the same matches (the public).
        model: Fitted prize model.
        bets: Expected bets of the draw (all players).
        jackpot: Jackpot announced for the Pleno al 15.
        max_cost: Budget in EUR (at most ``max_cost / 0.75`` bets).
        threshold: Prize sought in EUR.
        simulations: Simulated draws (large prizes are rare: many are needed).
        per_mix: Placements evaluated per mix of doubles and triples (the most
            likely to reach 13+ hits).
        pleno_rate: Chance of also hitting the Pleno al 15.
        rng: Random generator (seeded for reproducible results).

    Returns:
        The ticket, its chance of a large prize and its payout distribution.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    probs = np.asarray(probs, dtype=float)
    order, _ = _sorted_signs(probs)
    rank = np.argsort(order, axis=1)
    n = len(probs)
    cumulative = np.cumsum(probs / probs.sum(axis=1, keepdims=True), axis=1)

    def simulate() -> tuple[np.ndarray, np.ndarray]:
        """Ranks of simulated results and their surprise."""
        outcomes = np.minimum((rng.random((simulations, n, 1)) > cumulative[None]).sum(axis=2), 2)
        return rank[np.arange(n)[None, :], outcomes], surprise(crowd, outcomes)

    max_bets = int(max_cost // BET_PRICE)
    # Chosen on one set of simulations, valued on another (see plan_money)
    rank_hit, s = simulate()
    best: tuple[float, int, int, np.ndarray] | None = None
    for d, t, placements in _candidates(probs, max_bets, per_mix, min_hits=13):
        columns = column_hits(placements, rank_hit)
        chance = big_win_chance(columns, s, model, bets, jackpot, threshold, pleno_rate).mean(
            axis=1
        )
        fourteen = _fourteen_chance(
            placements, probs, crowd, model, bets, jackpot, threshold, pleno_rate, rng
        )
        chance = 1 - (1 - chance) * (1 - fourteen)
        i = int(np.argmax(chance))
        if best is None or chance[i] > best[0]:
            best = (float(chance[i]), d, t, placements[i])
    assert best is not None
    _, d, t, levels = best
    rank_hit, s = simulate()
    columns = column_hits(levels[None, :], rank_hit)
    common = float(
        big_win_chance(columns, s, model, bets, jackpot, threshold, pleno_rate)[0].mean()
    )
    rare = float(
        _fourteen_chance(
            levels[None, :], probs, crowd, model, bets, jackpot, threshold, pleno_rate, rng
        )[0]
    )
    # The payout distribution draws the winners with their real spread too; the
    # average uses the typical winners (the spread inflates averages, see plan_money)
    prizes, special = model.prizes(s, bets, jackpot, rng)
    payout = (columns[0] * prizes).sum(axis=1) + columns[0, :, 14] * pleno_rate * special
    typical, typical_special = model.prizes(s, bets, jackpot)
    mean_payout = (columns[0] * typical).sum(axis=1) + columns[
        0, :, 14
    ] * pleno_rate * typical_special
    plan = _build(probs, levels, 13, d, t)
    cost = BET_PRICE * plan.bets
    return BigWinPlan(
        plan=plan,
        cost=cost,
        threshold=threshold,
        chance=1 - (1 - common) * (1 - rare),
        chance_13=float((columns[0, :, 13:].sum(axis=1) > 0).mean()),
        chance_14=float(np.prod(_sorted_signs(probs)[1][np.arange(len(probs)), levels])),
        expected_payout=float(mean_payout.mean()),
        outcomes=_payout_outcomes(payout, cost),
    )


def big_win_chance(
    columns: np.ndarray,
    s: np.ndarray,
    model: PrizeModel,
    bets: float,
    jackpot: float,
    threshold: float,
    pleno_rate: float = PLENO_HIT_RATE,
) -> np.ndarray:
    r"""Chance that a ticket pays ``threshold`` or more in each simulated draw.

    In a draw with surprise $S$, the $c_k$ winning columns with $k$ hits pay
    $c_k \, s_k \, 0.75 B / W_k$ with $\log(W_k / B) = a_k + b_k S + \varepsilon_k$,
    $\varepsilon_k \sim N(0, \sigma_k^2)$, so they reach the threshold when
    $\varepsilon_k$ is small enough:

    $$
    P_k = \Phi\left(\frac{\log(c_k s_k \, 0.75 B / \text{threshold}) - \log B - a_k - b_k S}{\sigma_k}\right)
    $$

    This covers 12 and 13 hits (combined as independent); 14 hits and the
    Pleno al 15 are too rare for plain simulations and have their own exact
    term (``_fourteen_chance``). Using the spread of the winners matters: large
    prizes are the draws with fewer winners than usual.

    Args:
        columns: Columns with 0..14 hits, shape ``(n_tickets, n_draws, 15)``.
        s: Surprise of each simulated draw, shape ``(n_draws,)``.
        model: Fitted prize model.
        bets: Expected bets of the draw.
        jackpot: Jackpot announced for the Pleno al 15.
        threshold: Prize sought in EUR.
        pleno_rate: Chance of also hitting the Pleno al 15.

    Returns:
        Shape ``(n_tickets, n_draws)``.
    """
    from scipy.stats import norm

    takings = BET_PRICE * bets
    miss = np.ones(columns.shape[:2])
    for k in (12, 13):
        ours = columns[:, :, k]
        mu = model.intercept[k] + model.slope[k] * s
        with np.errstate(divide="ignore"):
            z = (
                np.log(ours * PRIZE_SHARES[k] * takings / threshold) - np.log(bets) - mu
            ) / model.sd[k]
        miss *= 1 - np.where(ours > 0, norm.cdf(z), 0.0)
    result: np.ndarray = 1 - miss
    return result


def _fourteen_chance(
    placements: np.ndarray,
    probs: np.ndarray,
    crowd: np.ndarray,
    model: PrizeModel,
    bets: float,
    jackpot: float,
    threshold: float,
    pleno_rate: float,
    rng: np.random.Generator,
    samples: int = 600,
) -> np.ndarray:
    r"""Chance of reaching ``threshold`` with 14 hits (and the Pleno al 15), per ticket.

    14 hits happen with probability $\prod_i q_i$ (the coverage of every match),
    a few times per thousand draws, too rarely to count in plain simulations.
    So that probability is exact, and only the surprise given 14 hits is
    sampled: each match's result drawn among the signs the ticket marks,
    with the model probabilities renormalised to them.

    Returns:
        Shape ``(n_tickets,)``.
    """
    from scipy.stats import norm

    order, cover = _sorted_signs(probs)
    n = len(probs)
    levels = np.atleast_2d(placements)
    rows = np.arange(n)
    p14 = np.prod(cover[rows, levels], axis=1)
    # Probabilities of the signs by rank, truncated to the marked ones and renormalised
    by_rank = np.take_along_axis(probs, order, axis=1)
    marked = np.arange(3)[None, None, :] <= levels[:, :, None]
    truncated = np.where(marked, by_rank[None, :, :], 0.0)
    cumulative = np.cumsum(truncated, axis=2) / truncated.sum(axis=2, keepdims=True)
    u = rng.random((samples, n))
    chosen = (u[None, :, :, None] > cumulative[:, None, :, :]).sum(axis=3)
    chosen = np.minimum(chosen, levels[:, None, :])
    signs = order[rows[None, None, :], chosen]
    picked = crowd[rows[None, None, :], signs]
    s = -np.log(np.clip(picked, PROB_FLOOR, 1.0)).sum(axis=2)

    takings = BET_PRICE * bets
    mu = model.intercept[14] + model.slope[14] * s
    z = (np.log(PRIZE_SHARES[14] * takings / threshold) - np.log(bets) - mu) / model.sd[14]
    pool = SPECIAL_SHARE * takings + jackpot
    share = np.log(max(model.special_share, PROB_FLOOR))
    z_special = (np.log(pool / threshold) - np.log(bets) - mu - share) / model.sd[14]
    reach = 1 - (1 - norm.cdf(z)) * (1 - pleno_rate * norm.cdf(z_special))
    result: np.ndarray = p14 * reach.mean(axis=1)
    return result


def ticket_payout(
    levels: np.ndarray,
    probs: np.ndarray,
    results: np.ndarray,
    winners: dict[int, float],
    takings: float,
    jackpot: float = 0.0,
    winners_special: float = 0.0,
    pleno_hit: bool = False,
) -> tuple[float, int, float]:
    """Real prize of a ticket in a past draw, as one more player.

    The pool of every category is shared among its real winners plus our
    winning columns (when nobody won it, we would take the whole pool).

    Args:
        levels: Coverage level of each match (0 single, 1 double, 2 triple).
        probs: Model probabilities the signs were chosen from, shape ``(14, 3)``.
        results: Real outcome index of every match (0 ``H``, 1 ``D``, 2 ``A``).
        winners: Real winners of every category (hits -> winners).
        takings: Takings of the draw in EUR.
        jackpot: Jackpot announced for the Pleno al 15.
        winners_special: Real winners of the Pleno al 15.
        pleno_hit: Whether our Pleno al 15 pick was right.

    Returns:
        ``(payout, best_hits, top)``: prize in EUR, hits of the best column and
        the part of the prize from 14 hits and the Pleno al 15.
    """
    order, _ = _sorted_signs(probs)
    rank = np.argsort(order, axis=1)
    rank_hit = rank[np.arange(len(probs)), np.asarray(results, dtype=int)]
    columns = column_hits(np.asarray(levels), rank_hit)[0, 0]
    payout, top = prize_payout(columns, winners, takings, jackpot, winners_special, pleno_hit)
    best_hits = int(np.max(np.nonzero(columns)[0])) if columns.any() else 0
    return payout, best_hits, top


def picks_columns(picks: list[str], results: list[str]) -> np.ndarray:
    """Columns of a ticket with every number of hits, from its marked signs.

    Args:
        picks: Signs marked in each match (``"H"``, ``"HD"``, ``"HDA"``...).
        results: Real results (``H``, ``D``, ``A``).

    Returns:
        Shape ``(n + 1,)``: columns with 0..n hits.
    """
    poly = np.array([1.0])
    for marked, result in zip(picks, results, strict=True):
        right = 1.0 if result in marked else 0.0
        # Each match multiplies the columns: (wrong signs) + (right sign) * x
        poly = np.convolve(poly, [len(marked) - right, right])
    return poly


def prize_payout(
    columns: np.ndarray,
    winners: dict[int, float],
    takings: float,
    jackpot: float = 0.0,
    winners_special: float = 0.0,
    pleno_hit: bool = False,
) -> tuple[float, float]:
    """Prize of a ticket's winning columns in a draw, as one more player.

    Args:
        columns: Columns with 0..14 hits.
        winners: Real (or expected) winners of every category, other players.
        takings: Takings of the draw in EUR.
        jackpot: Jackpot announced for the Pleno al 15.
        winners_special: Other winners of the Pleno al 15.
        pleno_hit: Whether our Pleno al 15 pick was right.

    Returns:
        ``(payout, top)``: prize in EUR and the part from 14 hits and the Pleno al 15.
    """
    payout, top = 0.0, 0.0
    for k in CATEGORY_HITS:
        ours = float(columns[k])
        if ours == 0:
            continue
        prize = PRIZE_SHARES[k] * takings / (winners.get(k, 0) + ours)
        if k == 10 and prize < MIN_10_HIT_PRIZE:
            continue
        payout += ours * prize
        top += ours * prize if k == 14 else 0.0
    if pleno_hit and columns[14] > 0:
        special = (SPECIAL_SHARE * takings + jackpot) / (winners_special + 1)
        payout += special
        top += special
    return payout, top
