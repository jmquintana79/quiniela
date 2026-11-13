from itertools import product

import numpy as np
import pandas as pd
import pytest

from quiniela.prediction import planner
from quiniela.prediction.confidence import confidence, tier


@pytest.fixture(scope="module")
def probs() -> np.ndarray:
    return np.random.default_rng(7).dirichlet([5, 3, 3], size=14)


# %% Confidence


def test_confidence_and_margin() -> None:
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.35, 0.45], [np.nan] * 3])
    out = confidence(probs)
    np.testing.assert_allclose(out["confidence"], [0.5, 0.45, np.nan])
    np.testing.assert_allclose(out["margin"], [0.2, 0.1, np.nan])
    # With the predicted signs, the confidence is the probability of that sign
    picked = confidence(probs, np.array(["D", "A", None], dtype=object))
    np.testing.assert_allclose(picked["confidence"], [0.3, 0.45, np.nan])


@pytest.mark.parametrize(
    ("value", "expected"), [(0.75, "high"), (0.6, "high"), (0.5, "medium"), (0.3, "low")]
)
def test_tier(value: float, expected: str) -> None:
    assert tier(value) == expected
    assert tier(None) is None and tier(float("nan")) is None


# %% Hit distribution


def test_hit_distribution_matches_enumeration() -> None:
    q = np.array([0.9, 0.5, 0.3, 0.7])
    expected = np.zeros(5)
    for outcome in product([0, 1], repeat=4):
        chance = np.prod([qi if hit else 1 - qi for qi, hit in zip(q, outcome, strict=True)])
        expected[sum(outcome)] += chance
    np.testing.assert_allclose(planner.hit_distribution(q), expected)
    # Several tickets at once
    np.testing.assert_allclose(planner.hit_distribution(np.vstack([q, q]))[1], expected)


def test_category_probabilities() -> None:
    odds = planner.category_probabilities(np.full(14, 0.5))
    assert list(odds) == [10, 11, 12, 13, 14]
    assert odds[14] == pytest.approx(0.5**14)
    assert odds[10] > odds[11] > odds[14]
    assert planner.category_probabilities(np.ones(14))[14] == pytest.approx(1)


# %% Planning


def test_compositions_fit_the_budget() -> None:
    mixes = planner.compositions(48)
    assert all(2**d * 3**t <= 48 for d, t in mixes)
    assert {(0, 0), (4, 1), (5, 0), (2, 2), (0, 3)} <= set(mixes)
    assert (3, 2) not in mixes  # 72 bets
    with pytest.raises(ValueError):
        planner.compositions(0)


def _brute_force(probs: np.ndarray, doubles: int, triples: int, min_hits: int) -> float:
    """Best probability of ``min_hits`` hits over every placement, by enumeration."""
    order = np.sort(probs, axis=1)[:, ::-1]
    cover = np.stack([order[:, 0], order[:, 0] + order[:, 1], np.ones(len(probs))], axis=1)
    best = 0.0
    for levels in product(range(3), repeat=len(probs)):
        levels_array = np.array(levels)
        if (levels_array == 1).sum() != doubles or (levels_array == 2).sum() != triples:
            continue
        coverage = cover[np.arange(len(probs)), levels_array]
        best = max(best, float(planner.hit_distribution(coverage)[min_hits:].sum()))
    return best


@pytest.mark.parametrize(("doubles", "triples", "min_hits"), [(2, 1, 5), (1, 2, 6), (3, 0, 7)])
def test_plan_is_optimal(doubles: int, triples: int, min_hits: int) -> None:
    probs = np.random.default_rng(doubles + triples).dirichlet([4, 3, 3], size=7)
    plan = planner.plan_ticket(probs, doubles, triples, min_hits=min_hits)
    assert (plan.doubles, plan.triples, plan.bets) == (doubles, triples, 2**doubles * 3**triples)
    assert plan.probability == pytest.approx(_brute_force(probs, doubles, triples, min_hits))


def test_plan_picks_follow_the_probabilities(probs: np.ndarray) -> None:
    plan = planner.plan_ticket(probs, doubles=4, triples=1)
    order = np.argsort(-probs, axis=1, kind="stable")
    for i, (picks, level) in enumerate(zip(plan.picks, plan.levels, strict=True)):
        assert len(picks) == level + 1
        # Doubles cover the two most likely signs, singles the most likely one
        assert picks == "".join(planner.OUTCOMES[k] for k in order[i, : level + 1])
    assert sum(len(p) == 2 for p in plan.picks) == 4
    assert sum(len(p) == 3 for p in plan.picks) == 1
    np.testing.assert_allclose(plan.distribution.sum(), 1)


def test_budget_chooses_the_best_mix(probs: np.ndarray) -> None:
    plan = planner.plan_ticket(probs, max_bets=48)
    assert plan.bets <= 48
    for d, t in planner.compositions(48):
        assert plan.probability >= planner.plan_ticket(probs, d, t).probability - 1e-12
    # More bets never lower the chance; a single ticket is the cheapest plan
    assert planner.plan_ticket(probs, max_bets=144).probability >= plan.probability
    assert planner.plan_ticket(probs, max_bets=1).bets == 1


def test_target_changes_the_plan(probs: np.ndarray) -> None:
    for_prize = planner.plan_ticket(probs, 4, 1, min_hits=10)
    for_top = planner.plan_ticket(probs, 4, 1, min_hits=14)
    assert for_prize.at_least(10) >= for_top.at_least(10) - 1e-12
    assert for_top.at_least(14) >= for_prize.at_least(14) - 1e-12


def test_baseline_strategies(probs: np.ndarray) -> None:
    optimal = planner.plan_ticket(probs, 4, 1)
    by_eye = planner.heuristic_plan(probs, 4, 1)
    # Triples on the least predictable match, doubles on the next four
    least = np.argsort(probs.max(axis=1), kind="stable")
    assert by_eye.levels[least[0]] == planner.TRIPLE
    assert (by_eye.levels[least[1:5]] == planner.DOUBLE).all()

    first = planner.random_plan(probs, np.random.default_rng(3), 4, 1)
    again = planner.random_plan(probs, np.random.default_rng(3), 4, 1)
    assert first.picks == again.picks and first.bets == 48

    single = planner.single_plan(probs)
    assert single.bets == 1 and all(len(p) == 1 for p in single.picks)
    assert optimal.probability >= max(by_eye.probability, first.probability, single.probability)


def test_ticket_hits() -> None:
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.4, 0.35, 0.25]])
    plan = planner.plan_ticket(probs, doubles=1, triples=1, min_hits=3)
    assert plan.hits(["H", "D", "A"]) == plan.hits(pd.Series(["H", "D", "A"]).to_numpy())
    assert plan.hits(list("HAA")) >= 2  # the triple always hits


def test_errors(probs: np.ndarray) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        planner.plan_ticket(np.vstack([probs[:13], [np.nan] * 3]))
    with pytest.raises(ValueError, match="min_hits"):
        planner.plan_ticket(probs, min_hits=15)
    with pytest.raises(ValueError, match="placements"):
        planner.plan_ticket(probs, max_bets=10**9)
    with pytest.raises(ValueError, match="shape"):
        planner.plan_ticket(np.ones(14))
