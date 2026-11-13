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
    probs: np.ndarray, max_bets: int, per_mix: int
) -> list[tuple[int, int, np.ndarray]]:
    """Placements worth simulating: the most likely to reach 11+ hits, for every mix."""
    _, cover = _sorted_signs(probs)
    n = len(probs)
    rows = np.arange(n)
    out = []
    for d, t in compositions(max_bets, n):
        if 2**d * 3**t < MIN_BETS:
            continue
        placements = _placements(n, d, t)
        chance = hit_distribution(cover[rows, placements])[:, 11:].sum(axis=1)
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
