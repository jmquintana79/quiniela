r"""Ticket planner (v4): where to spend the doubles and triples of a quiniela ticket.

A quiniela bet can cover more than one sign per match: a **double** (two
signs) multiplies the number of bets by 2 and a **triple** (1, X and 2) by 3,
so $d$ doubles and $t$ triples cost $2^d \, 3^t$ bets. The planner decides
which matches get them, from the 1X2 probabilities of a model (v3 by default).

**Coverage of a match.** With the probabilities of match $i$ sorted as
$p_{i(1)} \ge p_{i(2)} \ge p_{i(3)}$, the probability of hitting it is

$$
q_i = \begin{cases}
p_{i(1)} & \text{single sign} \\\\
p_{i(1)} + p_{i(2)} & \text{double (the two most likely signs)} \\\\
1 & \text{triple}
\end{cases}
$$

so a double gains $p_{i(2)}$ and a triple $1 - p_{i(1)}$.

**Probability of each prize.** Treating the 14 matches as independent, the
number of hits $H$ follows a Poisson-binomial distribution, computed exactly
by adding one match at a time:

$$
P_i(h) = P_{i-1}(h)\,(1 - q_i) + P_{i-1}(h - 1)\, q_i
$$

which gives the chance of each category: $P(H = 14)$ for the 1st,
$P(H \ge 10)$ for any prize.

**Optimal ticket.** Given the doubles and triples (or a maximum number of
bets, trying every mix that fits), every way of placing them on the 14
matches is evaluated and the one with the highest probability of the target,
$P(H \ge k)$, is kept ($k = 11$ by default, 10 for any prize, $k = 14$ for the top
one); ties are broken by the chance of 14 hits and then by the expected hits.
With up to a few hundred thousand placements per mix this exhaustive search
takes milliseconds.

Two simple strategies are kept to measure what the optimisation adds: random
matches (``random_plan``) and "by eye" (``heuristic_plan``: triples on the
least predictable matches, doubles on the next ones).
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from quiniela.prediction.ticket import TICKET_SIZE

OUTCOMES = ["H", "D", "A"]

# Model whose probabilities plan the ticket by default (v3)
PLANNER_MODEL = "market_blend"

# Default budget: 48 bets, e.g. 4 doubles and 1 triple (2^4 * 3 = 48)
DEFAULT_MAX_BETS = 48
DEFAULT_DOUBLES = 4
DEFAULT_TRIPLES = 1

# Default target: at least 11 hits (4th category or better). The 10-hit prize is
# often not paid at all (below 1.50 EUR), so securing it is a poor target
DEFAULT_MIN_HITS = 11

# Placements evaluated at once (memory bound) and in total (time bound)
CHUNK_SIZE = 50_000
MAX_PLACEMENTS = 2_000_000

# Coverage level of a match: 0 single sign, 1 double, 2 triple
SINGLE, DOUBLE, TRIPLE = 0, 1, 2


@dataclass(frozen=True)
class TicketPlan:
    """A ticket with its doubles and triples.

    Attributes:
        picks: Signs covered in each match, most likely first (``"H"``,
            ``"HD"``, ``"HDA"``...).
        levels: Coverage level of each match (0 single, 1 double, 2 triple).
        coverage: Probability of hitting each match with its picks.
        doubles: Number of doubles.
        triples: Number of triples.
        bets: Cost of the ticket in bets, $2^d 3^t$.
        min_hits: Hits the plan maximises the probability of.
        distribution: Probability of every number of hits, from 0 to the
            number of matches.
    """

    picks: list[str]
    levels: np.ndarray
    coverage: np.ndarray
    doubles: int
    triples: int
    bets: int
    min_hits: int
    distribution: np.ndarray

    @property
    def probability(self) -> float:
        """Probability of reaching ``min_hits`` hits."""
        return float(self.distribution[self.min_hits :].sum())

    def at_least(self, hits: int) -> float:
        """Probability of ``hits`` or more hits."""
        return float(self.distribution[hits:].sum())

    def hits(self, results: list[str] | np.ndarray) -> int:
        """Number of matches whose real result is among the picks.

        Args:
            results: Real results (``H``, ``D``, ``A``) in ticket order.

        Returns:
            Hits of the ticket.
        """
        return sum(result in picks for picks, result in zip(self.picks, results, strict=True))


def hit_distribution(coverage: np.ndarray) -> np.ndarray:
    """Probability of every number of hits (Poisson-binomial).

    Args:
        coverage: Probability of hitting each match, shape ``(n_matches,)`` or
            ``(n_tickets, n_matches)`` for several tickets at once.

    Returns:
        Shape ``(n_matches + 1,)`` (or ``(n_tickets, n_matches + 1)``): the
        probability of 0, 1, ..., ``n_matches`` hits.
    """
    q = np.atleast_2d(np.asarray(coverage, dtype=float))
    dist = np.zeros((q.shape[0], q.shape[1] + 1))
    dist[:, 0] = 1.0
    for i in range(q.shape[1]):
        hit = q[:, i : i + 1]
        dist[:, 1 : i + 2] = dist[:, 1 : i + 2] * (1 - hit) + dist[:, : i + 1] * hit
        dist[:, 0] *= 1 - hit[:, 0]
    result: np.ndarray = dist[0] if np.ndim(coverage) == 1 else dist
    return result


def category_probabilities(coverage: np.ndarray, size: int = TICKET_SIZE) -> dict[int, float]:
    """Probability of reaching each prize category (10 to 14 hits).

    Args:
        coverage: Probability of hitting each match of the ticket.
        size: Ticket size (the categories are 10 to ``size`` hits).

    Returns:
        ``{hits: P(H >= hits)}`` for 10 to ``size`` hits (``P(H = 14)`` for 14).
    """
    dist = hit_distribution(coverage)
    return {hits: float(dist[hits:].sum()) for hits in range(size - 4, size + 1)}


def _sorted_signs(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signs of every match from most to least likely, and the coverage levels.

    Returns:
        ``order`` (indices into ``OUTCOMES``, shape ``(n, 3)``) and ``levels``
        (probability of hitting with 1, 2 or 3 signs, shape ``(n, 3)``).
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(OUTCOMES):
        raise ValueError("probs must have shape (n_matches, 3)")
    if np.isnan(probs).any():
        raise ValueError("Every match of the ticket needs probabilities")
    order = np.argsort(-probs, axis=1, kind="stable")
    levels = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    levels[:, -1] = 1.0
    return order, levels


@lru_cache(maxsize=64)
def _placements(n: int, doubles: int, triples: int) -> np.ndarray:
    """Every way of placing the doubles and triples on ``n`` matches.

    Depends only on the counts, so it is cached across matchdays.

    Returns:
        Array of shape ``(n_placements, n)`` with the level of each match.
    """
    rows = []
    for triple in combinations(range(n), triples):
        rest = [i for i in range(n) if i not in triple]
        for double in combinations(rest, doubles):
            row = np.zeros(n, dtype=np.int8)
            row[list(triple)] = TRIPLE
            row[list(double)] = DOUBLE
            rows.append(row)
    placements = np.array(rows, dtype=np.int8).reshape(-1, n)
    placements.setflags(write=False)
    return placements


def _count_placements(n: int, doubles: int, triples: int) -> int:
    """Number of ways of placing the doubles and triples on ``n`` matches."""
    from math import comb

    return comb(n, triples) * comb(n - triples, doubles)


def compositions(max_bets: int, n: int = 14) -> list[tuple[int, int]]:
    r"""Every mix of doubles and triples that costs at most ``max_bets`` bets.

    Args:
        max_bets: Maximum number of bets ($2^d 3^t \le$ ``max_bets``).
        n: Matches in the ticket.

    Returns:
        ``(doubles, triples)`` pairs, cheapest first.
    """
    if max_bets < 1:
        raise ValueError("max_bets must be at least 1")
    mixes = [(d, t) for t in range(n + 1) for d in range(n + 1 - t) if 2**d * 3**t <= max_bets]
    return sorted(mixes, key=lambda m: (2 ** m[0] * 3 ** m[1], m))


def _build(
    probs: np.ndarray, levels: np.ndarray, min_hits: int, doubles: int, triples: int
) -> TicketPlan:
    """Ticket plan from the coverage level of each match."""
    order, cover = _sorted_signs(probs)
    coverage = cover[np.arange(len(levels)), levels]
    picks = ["".join(OUTCOMES[k] for k in order[i, : level + 1]) for i, level in enumerate(levels)]
    return TicketPlan(
        picks=picks,
        levels=np.asarray(levels, dtype=int),
        coverage=coverage,
        doubles=doubles,
        triples=triples,
        bets=2**doubles * 3**triples,
        min_hits=min_hits,
        distribution=hit_distribution(coverage),
    )


def plan_ticket(
    probs: np.ndarray,
    doubles: int | None = None,
    triples: int | None = None,
    max_bets: int | None = None,
    min_hits: int = DEFAULT_MIN_HITS,
) -> TicketPlan:
    """Optimal placement of doubles and triples on a ticket.

    Either give the exact ``doubles`` and ``triples`` (missing ones count as
    0), or a ``max_bets`` budget to also choose the best mix of them.

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        doubles: Number of doubles.
        triples: Number of triples.
        max_bets: Maximum cost in bets; every mix within it is tried.
            Defaults to ``DEFAULT_MAX_BETS`` when neither ``doubles`` nor
            ``triples`` is given.
        min_hits: Target: maximise the probability of at least this many
            hits (10 secures any prize, 14 aims at the top one).

    Returns:
        The best ticket.

    Raises:
        ValueError: With missing probabilities, an impossible target or a
            budget with too many placements to try.
    """
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    _, cover = _sorted_signs(probs)
    if not 0 <= min_hits <= n:
        raise ValueError(f"min_hits must be between 0 and {n}")
    if doubles is None and triples is None:
        mixes = compositions(DEFAULT_MAX_BETS if max_bets is None else max_bets, n)
    else:
        mixes = [(doubles or 0, triples or 0)]
    if any(d < 0 or t < 0 or d + t > n for d, t in mixes):
        raise ValueError(f"Doubles and triples must be non-negative and fit in {n} matches")
    total = sum(_count_placements(n, d, t) for d, t in mixes)
    if total > MAX_PLACEMENTS:
        raise ValueError(f"{total:,} placements to try: use fewer doubles and triples")

    best: tuple[tuple[float, float, float, int], np.ndarray, int, int] | None = None
    rows = np.arange(n)
    for d, t in mixes:
        placements = _placements(n, d, t)
        for start in range(0, len(placements), CHUNK_SIZE):
            chunk = placements[start : start + CHUNK_SIZE]
            coverage = cover[rows, chunk]
            dist = hit_distribution(coverage)
            target = dist[:, min_hits:].sum(axis=1)
            top = dist[:, -1]
            expected = coverage.sum(axis=1)
            # Highest target probability, then chance of 14, then expected hits
            i = int(np.lexsort((expected, top, target))[-1])
            # Cheaper mixes win ties (negative cost)
            score = (float(target[i]), float(top[i]), float(expected[i]), -(2**d * 3**t))
            if best is None or score > best[0]:
                best = (score, chunk[i].copy(), d, t)
    assert best is not None  # mixes always holds at least (0, 0)
    return _build(probs, best[1], min_hits, best[2], best[3])


def heuristic_plan(
    probs: np.ndarray,
    doubles: int = DEFAULT_DOUBLES,
    triples: int = DEFAULT_TRIPLES,
    min_hits: int = DEFAULT_MIN_HITS,
) -> TicketPlan:
    """Doubles and triples "by eye": on the least predictable matches.

    Triples go to the matches with the lowest confidence ($p_{(1)}$) and
    doubles to the next ones.

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        doubles: Number of doubles.
        triples: Number of triples.
        min_hits: Hits whose probability the plan reports.

    Returns:
        The ticket.
    """
    _, cover = _sorted_signs(probs)
    by_confidence = np.argsort(cover[:, 0], kind="stable")
    levels = np.zeros(len(cover), dtype=int)
    levels[by_confidence[:triples]] = TRIPLE
    levels[by_confidence[triples : triples + doubles]] = DOUBLE
    return _build(probs, levels, min_hits, doubles, triples)


def random_plan(
    probs: np.ndarray,
    rng: np.random.Generator,
    doubles: int = DEFAULT_DOUBLES,
    triples: int = DEFAULT_TRIPLES,
    min_hits: int = DEFAULT_MIN_HITS,
) -> TicketPlan:
    """Doubles and triples on random matches (doubles cover the two most likely signs).

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        rng: Random generator (seeded for reproducible backtests).
        doubles: Number of doubles.
        triples: Number of triples.
        min_hits: Hits whose probability the plan reports.

    Returns:
        The ticket.
    """
    n = len(probs)
    chosen = rng.permutation(n)
    levels = np.zeros(n, dtype=int)
    levels[chosen[:triples]] = TRIPLE
    levels[chosen[triples : triples + doubles]] = DOUBLE
    return _build(probs, levels, min_hits, doubles, triples)


def single_plan(probs: np.ndarray, min_hits: int = DEFAULT_MIN_HITS) -> TicketPlan:
    """The plain ticket: one sign per match (the most likely), one bet.

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        min_hits: Hits whose probability the plan reports.

    Returns:
        The ticket.
    """
    return _build(probs, np.zeros(len(probs), dtype=int), min_hits, 0, 0)
