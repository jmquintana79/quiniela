"""Quiniela ticket rules: which matches form the ticket and which prize it wins.

A quiniela ticket has 14 matches predicted as 1, X or 2, plus the *Pleno al
15* (exact goals of a 15th match). Prize categories by number of hits among
the 14 matches (Loterías y Apuestas del Estado):

======== =========
Hits     Category
======== =========
14       1st
13       2nd
12       3rd
11       4th
10       5th
======== =========

The *Pleno al 15* (special category) needs the exact score of the 15th match
and is not simulated.

The real ticket is chosen every week by Loterías y Apuestas del Estado. It is
approximated here with the matchday's La Liga matches followed by Segunda
matches, each in kick-off order, up to 14. A team appears at most once: when
a rescheduled match falls in the same matchday, the match closest to the
round's central date is kept.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

from quiniela.data.matchdays import LEAGUE_ORDER

TICKET_SIZE = 14

# Minimum hits for each prize category
PRIZE_CATEGORIES: dict[int, str] = {
    14: "1st category",
    13: "2nd category",
    12: "3rd category",
    11: "4th category",
    10: "5th category",
}
MIN_PRIZE_HITS = min(PRIZE_CATEGORIES)

TicketStatus = Literal["incomplete", "pending", "prize", "no_prize"]


def ticket_positions(matches: pd.DataFrame) -> pd.Series:
    """Choose the matches of a matchday that form the 14-match ticket.

    Args:
        matches: Matches of one matchday with ``league``, ``date``,
            ``home_team`` and ``away_team`` columns.

    Returns:
        Ticket position (1-14) aligned with ``matches.index``; ``<NA>`` for
        matches left out of the ticket.
    """
    rank = {league: i for i, league in enumerate(LEAGUE_ORDER)}
    leagues = sorted(matches["league"].unique(), key=lambda lg: (rank.get(lg, len(rank)), lg))

    chosen: list[pd.Index] = []
    for league in leagues:
        group = matches[matches["league"] == league]
        # Matches closest to the round's central date first, so a rescheduled
        # match does not displace the regular one of the same team
        center = group["date"].median()
        by_closeness = group.assign(_gap=(group["date"] - center).abs()).sort_values(
            ["_gap", "date"], kind="stable"
        )
        teams: set[str] = set()
        picked = []
        for idx, home, away in zip(
            by_closeness.index, by_closeness["home_team"], by_closeness["away_team"], strict=True
        ):
            if home not in teams and away not in teams:
                picked.append(idx)
                teams.update((home, away))
        chosen.append(group.loc[picked].sort_values("date", kind="stable").index)

    ordered = [idx for league_index in chosen for idx in league_index][:TICKET_SIZE]
    positions = pd.Series(range(1, len(ordered) + 1), index=ordered, dtype="Int64")
    return positions.reindex(matches.index)


def prize_category(hits: int) -> str | None:
    """Return the prize category won with a number of hits, or None."""
    return next((name for minimum, name in PRIZE_CATEGORIES.items() if hits >= minimum), None)


@dataclass(frozen=True)
class TicketResult:
    """Outcome of a 14-match ticket.

    Attributes:
        status: ``incomplete`` (fewer than 14 matches), ``pending`` (some
            results unknown), ``prize`` or ``no_prize``.
        size: Matches in the ticket.
        hits: Correct predictions among the played matches.
        pending: Matches without a result yet.
        category: Prize won (only when every result is known).
        best_possible: Best prize still reachable while results are pending.
        message: Human-readable summary.
    """

    status: TicketStatus
    size: int
    hits: int
    pending: int
    category: str | None
    best_possible: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-friendly dictionary."""
        return asdict(self)


def evaluate_ticket(size: int, hits: int, pending: int) -> TicketResult:
    """Evaluate a ticket from its number of hits.

    Args:
        size: Matches in the ticket.
        hits: Correct predictions among the played matches.
        pending: Matches not played yet.

    Returns:
        The ticket outcome.
    """
    if size < TICKET_SIZE:
        return TicketResult(
            "incomplete",
            size,
            hits,
            pending,
            None,
            None,
            f"Only {size} matches this matchday: a {TICKET_SIZE}-match ticket can't be built.",
        )
    if pending:
        best = prize_category(hits + pending)
        outlook = f"best possible: {best}" if best else "no prize possible"
        return TicketResult(
            "pending",
            size,
            hits,
            pending,
            None,
            best,
            f"{hits} hits so far, {pending} matches pending ({outlook}).",
        )
    category = prize_category(hits)
    if category:
        return TicketResult(
            "prize", size, hits, 0, category, category, f"{hits}/{size} hits: {category} prize."
        )
    return TicketResult(
        "no_prize",
        size,
        hits,
        0,
        None,
        None,
        f"{hits}/{size} hits: no prize (at least {MIN_PRIZE_HITS} needed).",
    )
