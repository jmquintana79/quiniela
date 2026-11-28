r"""Reduced tickets (v3+v3F·R): the official reductions of the quiniela in the ticket planner.

A **direct** ticket plays every combination of its doubles and triples
($2^d \, 3^t$ bets). A **reduced** ticket covers the same signs with only some
of those combinations, chosen so that every possible result is close to at
least one column played: if every covered sign is right, the best column has
at most one miss (two for the 5th and 6th reductions). The official ticket
offers six reductions:

| Reduction | Doubles | Triples | Direct bets | Bets | Guaranteed hits |
|---|---|---|---|---|---|
| 1st | 0 | 4 | 81 | 9 | 13 |
| 2nd | 7 | 0 | 128 | 16 | 13 |
| 3rd | 3 | 3 | 216 | 24 | 13 |
| 4th | 6 | 2 | 576 | 64 | 13 |
| 5th | 0 | 8 | 6,561 | 81 | 12 |
| 6th | 11 | 0 | 2,048 | 132 | 12 |

(the guarantee assumes the other matches of the ticket are right too). The
remaining matches can be played direct, with more doubles and triples that
multiply the bets as usual.

**Columns.** The official column tables are not published online, so each
reduction is a covering code with the official size and guarantee: the
ternary and binary Hamming codes for the 1st and 2nd (perfect codes), their
product for the 5th and codes found by local search for the others, all
translated to play the column of the most likely signs. Equivalent codes
give the same guarantee; the chance of 14 hits can differ slightly from the
official columns.

**Hit distribution.** With the probabilities of each match sorted from most
to least likely, the columns name each sign by its rank (0 the most likely;
in a double, rank 2 is the sign left out). For the $R$ matches of the
reduction every result $o \in \lbrace 0, 1, 2 \rbrace^R$ is enumerated (at most
$3^8$), with probability $\prod_i p_{i(o_i)}$ and the hits of the best column,

$$
h(o) = \max_{c \in C} \#\lbrace i : c_i = o_i \rbrace,
$$

which gives the exact distribution of the hits of the reduced block. The
other matches follow the Poisson-binomial of the direct planner, and the two
blocks are independent, so the hits of the ticket are their convolution.

**Search.** For each reduction within the budget, every way of putting it on
the least predictable matches (the ``CANDIDATES`` with the lowest
confidence) and every placement of the extra direct doubles and triples that
still fit is evaluated, and the ticket with the highest $P(H \ge k)$ wins
(ties: the chance of 14, the expected hits, then the cheaper ticket). v3+v3F·R
compares it with the best direct ticket (v3+v3F·D) and keeps the better one, so
its planned chance is never below v3+v3F·D's.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from quiniela.prediction.planner import (
    DEFAULT_MAX_BETS,
    DEFAULT_MIN_HITS,
    TicketPlan,
    _placements,
    _sorted_signs,
    compositions,
    hit_distribution,
    plan_ticket,
)

# Least predictable matches the reduction may go to (keeps the search fast)
CANDIDATES = 10


@dataclass(frozen=True)
class Reduction:
    """One official reduction of the quiniela ticket.

    Attributes:
        number: Number of the reduction on the official ticket (1 to 6).
        triples: Triples of the reduction (the first coordinates of the code).
        doubles: Doubles of the reduction (the last coordinates).
        code: Columns played, shape ``(bets, triples + doubles)``: the rank of
            the sign marked in each match (0 the most likely).
        guarantee: Misses the best column can have when every covered sign is
            right (1: guaranteed 13, 2: guaranteed 12).
    """

    number: int
    triples: int
    doubles: int
    code: np.ndarray
    guarantee: int

    @property
    def bets(self) -> int:
        """Bets of the reduction."""
        return len(self.code)

    @property
    def size(self) -> int:
        """Matches of the reduction."""
        return self.triples + self.doubles

    @property
    def label(self) -> str:
        """Short description, e.g. ``"3 doubles + 3 triples"``."""
        parts = []
        if self.doubles:
            parts.append(f"{self.doubles} double{'s' if self.doubles > 1 else ''}")
        if self.triples:
            parts.append(f"{self.triples} triple{'s' if self.triples > 1 else ''}")
        return " + ".join(parts)


# Columns of each reduction (one word per column, triples first), see the module docstring
_CODES: dict[int, str] = {
    1: ("0000 0112 0221 1011 1120 1202 2022 2101 2210 "),
    2: (
        "0000000 0001111 0010110 0011001 0100101 0101010 0110011 0111100 1000011 1001100 "
        "1010101 1011010 1100110 1101001 1110000 1111111 "
    ),
    3: (
        "000000 000111 011001 011100 011110 022010 022011 022101 101010 101011 101101 112000 "
        "112111 120001 120100 120110 202001 202100 202110 210010 210011 210101 221000 221111 "
    ),
    4: (
        "00000000 00000010 00000111 00111000 00111010 00111111 01001011 01001100 01010011 "
        "01010100 01101011 01101100 01110011 01110100 02011001 02011101 02011110 02100001 "
        "02100101 02100110 10001001 10001101 10001110 10010001 10010101 10010110 10101001 "
        "10101101 10101110 10110001 10110101 10110110 11011000 11011010 11011111 11100000 "
        "11100010 11100111 12000011 12000100 12111011 12111100 20011011 20011100 20100011 "
        "20100100 21000001 21000101 21000110 21111001 21111101 21111110 22001000 22001010 "
        "22001111 22010000 22010010 22010111 22101000 22101010 22101111 22110000 22110010 "
        "22110111 "
    ),
    5: (
        "00000000 00000112 00000221 00001011 00001120 00001202 00002022 00002101 00002210 "
        "01120000 01120112 01120221 01121011 01121120 01121202 01122022 01122101 01122210 "
        "02210000 02210112 02210221 02211011 02211120 02211202 02212022 02212101 02212210 "
        "10110000 10110112 10110221 10111011 10111120 10111202 10112022 10112101 10112210 "
        "11200000 11200112 11200221 11201011 11201120 11201202 11202022 11202101 11202210 "
        "12020000 12020112 12020221 12021011 12021120 12021202 12022022 12022101 12022210 "
        "20220000 20220112 20220221 20221011 20221120 20221202 20222022 20222101 20222210 "
        "21010000 21010112 21010221 21011011 21011120 21011202 21012022 21012101 21012210 "
        "22100000 22100112 22100221 22101011 22101120 22101202 22102022 22102101 22102210 "
    ),
    6: (
        "00000000000 00000010101 00000011010 00000100001 00000111101 00001000100 00001001011 "
        "00001100011 00001110111 00010010111 00010011010 00010011011 00010100011 00010100100 "
        "00010101010 00010110011 00011111001 00100000001 00100111010 00101011011 00101011100 "
        "00101100011 00110011110 00111000010 00111011111 00111101001 00111110011 00111110110 "
        "01000001111 01000010011 01000011000 01001000011 01001000101 01001100010 01001101001 "
        "01001101101 01001110100 01010001001 01010011000 01010100010 01010101111 01011000000 "
        "01011110111 01011111001 01011111101 01100001000 01100011011 01100011111 01100100010 "
        "01100101100 01100101101 01100110000 01100110010 01100110100 01101000011 01110000001 "
        "01110000110 01110011000 01110100110 01111000001 01111001100 01111011110 01111111000 "
        "10000001101 10000010001 10000011001 10000101010 10000101110 10000110001 10000111111 "
        "10001110001 10001111111 10010000000 10010101000 10010101111 10010110011 10011001100 "
        "10011001111 10011101001 10011101111 10011110011 10011111101 10100001111 10100011110 "
        "10100101001 10100110101 10101000011 10101000101 10101011011 10101101010 10101111011 "
        "10110000001 10110001011 10110010110 10110011101 10111000101 10111010100 10111110000 "
        "10111111111 11000010010 11000010100 11000100111 11000111000 11000111110 11001001010 "
        "11001010100 11001011000 11001110111 11001111010 11010000100 11010001011 11010011001 "
        "11011001111 11011010101 11011011011 11011100001 11011100110 11011110001 11100010011 "
        "11100011000 11100100100 11100111100 11101001100 11101011000 11101011110 11110000001 "
        "11110000100 11110100110 11111001100 11111001111 11111010101 11111111010 "
    ),
}

# (triples, doubles, guarantee) of each official reduction
_SPECS = {1: (4, 0, 1), 2: (0, 7, 1), 3: (3, 3, 1), 4: (2, 6, 1), 5: (8, 0, 2), 6: (0, 11, 2)}


def _reduction(number: int) -> Reduction:
    triples, doubles, guarantee = _SPECS[number]
    code = np.array([[int(s) for s in word] for word in _CODES[number].split()], dtype=np.int8)
    code.setflags(write=False)
    return Reduction(number, triples, doubles, code, guarantee)


REDUCTIONS: dict[int, Reduction] = {number: _reduction(number) for number in _SPECS}


@lru_cache(maxsize=8)
def _best_hits(number: int) -> tuple[np.ndarray, np.ndarray]:
    """Every result of a reduction's matches and the hits of its best column.

    Returns:
        ``outcomes`` (rank of the real sign in each match, shape
        ``(3^R, R)``) and ``best`` (hits of the best column, shape ``(3^R,)``).
    """
    code = REDUCTIONS[number].code
    outcomes = np.array(list(product(range(3), repeat=code.shape[1])), dtype=np.int8)
    best = (outcomes[:, None, :] == code[None, :, :]).sum(axis=2).max(axis=1)
    outcomes.setflags(write=False)
    best.setflags(write=False)
    return outcomes, best


def block_distribution(number: int, ranked: np.ndarray) -> np.ndarray:
    """Distribution of the hits of a reduction's best column.

    Args:
        number: Reduction number.
        ranked: Probabilities of the reduction's matches sorted from most to
            least likely, shape ``(R, 3)`` or ``(n_tickets, R, 3)``, with the
            triples first.

    Returns:
        Shape ``(R + 1,)`` (or ``(n_tickets, R + 1)``): probability of 0..R hits.
    """
    outcomes, best = _best_hits(number)
    ranked = np.asarray(ranked, dtype=float)
    single = ranked.ndim == 2
    if single:
        ranked = ranked[None]
    size = outcomes.shape[1]
    prob = np.ones((ranked.shape[0], len(outcomes)))
    for i in range(size):
        prob *= ranked[:, i, outcomes[:, i]]
    onehot = np.zeros((len(outcomes), size + 1))
    onehot[np.arange(len(outcomes)), best] = 1.0
    dist: np.ndarray = prob @ onehot
    return dist[0] if single else dist


def _combine(block: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Hits of two independent blocks: convolution of their distributions (last axis)."""
    size = block.shape[-1] + rest.shape[-1] - 1
    out = np.zeros((*np.broadcast_shapes(block.shape[:-1], rest.shape[:-1]), size))
    for k in range(block.shape[-1]):
        out[..., k : k + rest.shape[-1]] += block[..., k : k + 1] * rest
    return out


def reduced_plan(
    probs: np.ndarray,
    number: int,
    triples: list[int],
    doubles: list[int],
    levels: np.ndarray | None = None,
    min_hits: int = DEFAULT_MIN_HITS,
) -> TicketPlan:
    """Ticket with a reduction on the given matches (and direct picks elsewhere).

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        number: Reduction number (1 to 6).
        triples: Matches of the reduction's triples.
        doubles: Matches of the reduction's doubles.
        levels: Coverage level of the other matches (0 single, 1 direct
            double, 2 direct triple); singles by default.
        min_hits: Hits the plan reports the probability of.

    Returns:
        The ticket.

    Raises:
        ValueError: If the matches do not fit the reduction.
    """
    reduction = REDUCTIONS[number]
    if len(triples) != reduction.triples or len(doubles) != reduction.doubles:
        raise ValueError(f"Reduction {number} has {reduction.label}")
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    order, cover = _sorted_signs(probs)
    block = [*triples, *doubles]
    if len(set(block)) != len(block) or not all(0 <= i < n for i in block):
        raise ValueError("The reduction needs distinct matches of the ticket")
    levels = np.zeros(n, dtype=int) if levels is None else np.asarray(levels, dtype=int).copy()
    levels[list(triples)] = 2
    levels[list(doubles)] = 1
    rest = [i for i in range(n) if i not in block]
    ranked = np.take_along_axis(probs, order, axis=1)
    dist = _combine(
        block_distribution(number, ranked[block]),
        hit_distribution(cover[rest, levels[rest]]),
    )
    extra_doubles = int((levels[rest] == 1).sum())
    extra_triples = int((levels[rest] == 2).sum())
    picks = ["".join("HDA"[k] for k in order[i, : level + 1]) for i, level in enumerate(levels)]
    return TicketPlan(
        picks=picks,
        levels=levels,
        coverage=cover[np.arange(n), levels],
        doubles=int((levels == 1).sum()),
        triples=int((levels == 2).sum()),
        bets=reduction.bets * 2**extra_doubles * 3**extra_triples,
        min_hits=min_hits,
        distribution=dist,
        reduction=number,
        reduced=tuple(block),
        code=reduction.code,
    )


def _search(
    probs: np.ndarray, max_bets: int, min_hits: int, candidates: int, top: int
) -> list[tuple[np.ndarray, int, list[int], list[int], np.ndarray]]:
    """The best reduced tickets of every reduction and mix of extra direct picks.

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        max_bets: Maximum cost in bets.
        min_hits: Target: the probability of at least this many hits ranks them.
        candidates: Least predictable matches the reduction may go to.
        top: Tickets kept per reduction and mix of extra doubles and triples.

    Returns:
        ``(score, number, triples, doubles, levels)`` of each ticket kept: the
        score orders them (target probability, chance of 14, expected hits,
        cheaper), ``levels`` are the extra direct picks.
    """
    n = len(probs)
    order, cover = _sorted_signs(probs)
    ranked = np.take_along_axis(probs, order, axis=1)

    def score(dist: np.ndarray, bets: np.ndarray) -> np.ndarray:
        # Target probability, then chance of 14, then expected hits, then cheaper
        hits = np.arange(dist.shape[-1])
        return np.stack(
            [dist[..., min_hits:].sum(axis=-1), dist[..., -1], (dist * hits).sum(axis=-1), -bets],
            axis=-1,
        )

    found = []
    pool = [int(i) for i in np.argsort(cover[:, 0], kind="stable")[: max(candidates, 0)]]
    for number, reduction in REDUCTIONS.items():
        if reduction.bets > max_bets or reduction.size > min(len(pool), n):
            continue
        # Placements of the reduction on the candidates: (triples, doubles)
        blocks = [
            (list(t), list(d))
            for t in combinations(pool, reduction.triples)
            for d in combinations([i for i in pool if i not in t], reduction.doubles)
        ]
        block_dists = block_distribution(number, np.stack([ranked[[*t, *d]] for t, d in blocks]))
        rests = np.array(
            [[i for i in range(n) if i not in t and i not in d] for t, d in blocks], dtype=int
        )
        for extra_d, extra_t in compositions(max_bets // reduction.bets, n - reduction.size):
            bets = reduction.bets * 2**extra_d * 3**extra_t
            extra = _placements(n - reduction.size, extra_d, extra_t)
            # Every block placement with every placement of the extra direct picks at once
            coverage = cover[rests[:, None, :], extra[None, :, :]]
            rest_dist = hit_distribution(coverage.reshape(-1, coverage.shape[-1]))
            dist = _combine(
                block_dists[:, None, :], rest_dist.reshape(len(blocks), len(extra), -1)
            ).reshape(-1, n + 1)
            scores = score(dist, np.full(len(dist), float(bets)))
            for i in np.lexsort(scores.T[::-1])[::-1][:top]:
                block, placement = divmod(int(i), len(extra))
                levels = np.zeros(n, dtype=int)
                levels[rests[block]] = extra[placement]
                found.append((scores[i], number, *blocks[block], levels))
    return found


def reduced_candidates(
    probs: np.ndarray,
    max_bets: int,
    min_hits: int,
    candidates: int = CANDIDATES,
    top: int = 3,
) -> list[TicketPlan]:
    """Reduced tickets worth trying in another planner (the big-win ticket, ·B).

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        max_bets: Maximum cost in bets.
        min_hits: Hits whose probability picks the best placements.
        candidates: Least predictable matches the reduction may go to.
        top: Tickets per reduction and mix of extra doubles and triples.

    Returns:
        The tickets, each with its reduction (``TicketPlan.code``).
    """
    probs = np.asarray(probs, dtype=float)
    return [
        reduced_plan(probs, number, t, d, levels, min_hits)
        for _, number, t, d, levels in _search(probs, max_bets, min_hits, candidates, top)
    ]


def plan_reduced(
    probs: np.ndarray,
    max_bets: int = DEFAULT_MAX_BETS,
    min_hits: int = DEFAULT_MIN_HITS,
    candidates: int = CANDIDATES,
) -> TicketPlan:
    """Best ticket within a budget, direct or with an official reduction (v3+v3F·R).

    Args:
        probs: 1X2 probabilities of the ticket matches, shape ``(n, 3)``.
        max_bets: Maximum cost in bets.
        min_hits: Target: maximise the probability of at least this many hits.
        candidates: Least predictable matches the reduction may go to.

    Returns:
        The best reduced ticket, or the best direct one (v3+v3F·D) when no
        reduction beats it.
    """
    probs = np.asarray(probs, dtype=float)
    direct = plan_ticket(probs, max_bets=max_bets, min_hits=min_hits)
    hits = np.arange(len(direct.distribution))
    best_score = np.array(
        [
            direct.distribution[min_hits:].sum(),
            direct.distribution[-1],
            (direct.distribution * hits).sum(),
            -float(direct.bets),
        ]
    )
    best = None
    for score, number, t, d, levels in _search(probs, max_bets, min_hits, candidates, top=1):
        if tuple(score) > tuple(best_score):
            best_score, best = score, (number, t, d, levels)
    if best is None:
        return direct
    number, t, d, levels = best
    return reduced_plan(probs, number, t, d, levels, min_hits)
