import itertools

import numpy as np
import pytest

from quiniela.prediction import planner, reduced


@pytest.mark.parametrize("number", sorted(reduced.REDUCTIONS))
def test_official_sizes_and_guarantees(number: int) -> None:
    reduction = reduced.REDUCTIONS[number]
    official = {1: 9, 2: 16, 3: 24, 4: 64, 5: 81, 6: 132}
    assert reduction.bets == official[number]
    sizes = [3] * reduction.triples + [2] * reduction.doubles
    assert reduction.code.shape == (reduction.bets, len(sizes))
    assert (reduction.code < np.array(sizes)).all()
    # Every result of the covered signs is within the guarantee of some column
    if np.prod(sizes) <= 3000:
        space = np.array(list(itertools.product(*[range(s) for s in sizes])))
        misses = (space[:, None, :] != reduction.code[None, :, :]).sum(axis=2).min(axis=1)
        assert misses.max() == reduction.guarantee
    # The column of the most likely signs is played
    assert (reduction.code == 0).all(axis=1).any()


def _probs(seed: int, n: int = 14) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet([3, 2, 2], size=n)


def test_reduced_distribution_matches_brute_force() -> None:
    probs = _probs(0, 7)
    # Reduction 1 (4 triples) on matches 0-3, a direct double on 5
    plan = reduced.reduced_plan(probs, 1, [0, 1, 2, 3], [], levels=[0, 0, 0, 0, 0, 1, 0])
    assert plan.bets == 18 and plan.reduction == 1 and plan.reduced == (0, 1, 2, 3)
    dist = np.zeros(8)
    for outcome in itertools.product(range(3), repeat=7):
        results = ["HDA"[k] for k in outcome]
        dist[plan.hits(results)] += np.prod([probs[i, k] for i, k in enumerate(outcome)])
    assert np.allclose(dist, plan.distribution)
    assert plan.columns(["H"] * 7).sum() == plan.bets


def test_reduced_hits_lose_at_most_the_guarantee() -> None:
    probs = _probs(1, 7)
    plan = reduced.reduced_plan(probs, 2, [], list(range(7)))
    # Every result among the covered signs: at most one miss in the best column
    for picked in itertools.product(*[range(2)] * 7):
        results = [plan.picks[i][k] for i, k in enumerate(picked)]
        assert plan.hits(results) >= 6
    # Not played yet: misses in every column
    assert plan.hits([None] * 7) == 0


def test_plan_reduced_never_below_direct() -> None:
    for seed in range(3):
        probs = _probs(seed)
        direct = planner.plan_ticket(probs, max_bets=48)
        plan = reduced.plan_reduced(probs, max_bets=48)
        assert plan.bets <= 48
        assert plan.probability >= direct.probability - 1e-12
        assert plan.distribution.sum() == pytest.approx(1)
    # With a budget too small for any reduction, the direct ticket
    assert reduced.plan_reduced(_probs(0), max_bets=8).reduction is None


def test_reduced_plan_checks_the_matches() -> None:
    with pytest.raises(ValueError):
        reduced.reduced_plan(_probs(0), 1, [0, 1, 2], [])
    with pytest.raises(ValueError):
        reduced.reduced_plan(_probs(0), 1, [0, 0, 1, 2], [])
