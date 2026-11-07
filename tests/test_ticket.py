import pandas as pd
import pytest

from quiniela.prediction.ticket import evaluate_ticket, prize_category, ticket_positions
from tests.helpers import make_matches


@pytest.mark.parametrize(
    ("hits", "category"),
    [
        (14, "1st category"),
        (13, "2nd category"),
        (12, "3rd category"),
        (11, "4th category"),
        (10, "5th category"),
        (9, None),
        (0, None),
    ],
)
def test_prize_category(hits: int, category: str | None) -> None:
    assert prize_category(hits) == category


def test_ticket_takes_la_liga_first_then_segunda_up_to_14() -> None:
    primera = [
        ("SP1", f"2026-08-{15 + i % 3}", f"P{2 * i}", f"P{2 * i + 1}", 1, 0) for i in range(10)
    ]
    segunda = [("SP2", "2026-08-14", f"S{2 * i}", f"S{2 * i + 1}", 1, 0) for i in range(11)]
    matches = make_matches(segunda + primera)

    positions = ticket_positions(matches)

    ticket = matches.assign(pos=positions).dropna(subset=["pos"]).sort_values("pos")
    assert len(ticket) == 14
    assert ticket["league"].tolist() == ["SP1"] * 10 + ["SP2"] * 4
    # La Liga matches in kick-off order
    assert ticket["date"].iloc[:10].is_monotonic_increasing


def test_ticket_skips_rescheduled_match_of_a_repeated_team() -> None:
    matches = make_matches(
        [
            ("SP1", "2026-08-27", "A", "B", 1, 0),  # rescheduled, Thursday
            ("SP1", "2026-08-29", "A", "C", 1, 0),
            ("SP1", "2026-08-29", "D", "E", 1, 0),
            ("SP1", "2026-08-30", "F", "G", 1, 0),
        ]
    )
    positions = ticket_positions(matches)
    assert positions.isna().tolist() == [True, False, False, False]


def test_evaluate_ticket_statuses() -> None:
    assert evaluate_ticket(size=11, hits=8, pending=0).status == "incomplete"

    pending = evaluate_ticket(size=14, hits=8, pending=3)
    assert (pending.status, pending.best_possible) == ("pending", "4th category")
    assert evaluate_ticket(size=14, hits=5, pending=2).best_possible is None

    prize = evaluate_ticket(size=14, hits=12, pending=0)
    assert (prize.status, prize.category) == ("prize", "3rd category")

    no_prize = evaluate_ticket(size=14, hits=9, pending=0)
    assert (no_prize.status, no_prize.category) == ("no_prize", None)
    assert "at least 10" in no_prize.message


def test_ticket_result_to_dict() -> None:
    assert evaluate_ticket(14, 14, 0).to_dict()["category"] == "1st category"
    assert isinstance(evaluate_ticket(14, 14, 0).to_dict(), dict)
    assert pd.isna(ticket_positions(make_matches([])).max())
