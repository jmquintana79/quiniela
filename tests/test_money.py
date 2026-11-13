from itertools import product

import numpy as np
import pandas as pd
import pytest

from quiniela.evaluation import money as backtest_money
from quiniela.prediction import money


@pytest.fixture(scope="module")
def model() -> money.PrizeModel:
    # Synthetic draws with the coefficients fitted on the real ones (2010-2026)
    rng = np.random.default_rng(0)
    surprise = rng.uniform(11, 17, size=400)
    bets = np.full(400, 3_000_000.0)
    draws = pd.DataFrame({"surprise": surprise, "bets": bets})
    real = {
        14: (0.66, -1.05),
        13: (5.84, -1.19),
        12: (6.8, -1.07),
        11: (6.92, -0.93),
        10: (6.34, -0.77),
    }
    for k, (a, b) in real.items():
        share = np.exp(a + b * surprise + rng.normal(0, 0.3, 400))
        draws[f"winners_{k}"] = np.round(bets * share)
    draws["winners_special"] = np.round(draws["winners_14"] * 0.2)
    return money.PrizeModel.fit(draws)


def test_prize_model_fit(model: money.PrizeModel) -> None:
    assert model.slope[13] == pytest.approx(-1.19, abs=0.1)
    assert model.slope[11] == pytest.approx(-0.93, abs=0.1)
    # Few 14-hit winners (rounded to whole people) flatten its slope, still clearly negative
    assert model.slope[14] < -0.5
    assert 0.05 < model.special_share < 0.35


def test_prizes_grow_with_the_surprise(model: money.PrizeModel) -> None:
    prizes, special = model.prizes(np.array([12.0, 16.0]), 3_000_000, jackpot=1_000_000)
    assert (prizes[1, 10:] >= prizes[0, 10:]).all()
    assert (prizes[:, :10] == 0).all()
    assert special[1] > special[0] > 1_000_000 / 10


def test_ten_hit_floor(model: money.PrizeModel) -> None:
    # Very predictable results: so many 10-hit winners that the prize is not paid
    prizes, _ = model.prizes(np.array([5.0]), 3_000_000, 0.0)
    assert prizes[0, 10] == 0


def test_surprise() -> None:
    crowd = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
    assert money.surprise(crowd, np.array([0, 2]))[0] == pytest.approx(-2 * np.log(0.5))
    many = money.surprise(crowd, np.array([[0, 2], [1, 0]]))
    assert many[1] == pytest.approx(-np.log(0.25) - np.log(0.2))


def _brute_force(levels: list[int], rank_hit: list[int]) -> np.ndarray:
    counts = np.zeros(len(levels) + 1)
    for combo in product(*[range(level + 1) for level in levels]):
        counts[sum(r == h for r, h in zip(combo, rank_hit, strict=True))] += 1
    return counts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_column_hits(seed: int) -> None:
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 3, size=7)
    rank_hit = rng.integers(0, 3, size=(4, 7))
    columns = money.column_hits(levels, rank_hit)[0]
    for draw in range(4):
        np.testing.assert_array_equal(
            columns[draw], _brute_force(list(levels), list(rank_hit[draw]))
        )
    assert columns.sum(axis=1).tolist() == [float(np.prod(levels + 1))] * 4


def test_picks_columns() -> None:
    columns = money.picks_columns(["H", "HD", "HDA", "A"], ["H", "A", "D", "A"])
    # 1 * (2 wrong) * (1 right + 2 wrong) * 1: 6 columns, 2 with 3 hits, 4 with 2 hits
    assert columns.tolist() == [0, 0, 4, 2, 0]


def test_prize_payout_shares_the_pools() -> None:
    columns = np.zeros(15)
    columns[14], columns[13] = 1, 2
    winners = {14: 0.0, 13: 2.0, 12: 0.0, 11: 0.0, 10: 0.0}
    payout, top = money.prize_payout(columns, winners, 1_000_000, jackpot=500_000, pleno_hit=True)
    pool_14 = 0.16 * 1_000_000  # nobody else won it: the whole pool
    pool_13 = 0.075 * 1_000_000 / 4 * 2  # shared with two other winners
    special = 0.075 * 1_000_000 + 500_000
    assert payout == pytest.approx(pool_14 + pool_13 + special)
    assert top == pytest.approx(pool_14 + special)


def test_plan_money_prudent_and_full(model: money.PrizeModel) -> None:
    probs = np.random.default_rng(5).dirichlet([5, 3, 3], size=14)
    args = (probs, probs, model, 3_000_000)
    plain = money.plan_money(*args, jackpot=0.0, simulations=1500)
    assert money.MIN_BETS <= plain.plan.bets <= money.DEFAULT_MAX_BETS
    assert plain.cost == pytest.approx(0.75 * plain.plan.bets)
    assert plain.expected_net == pytest.approx(plain.expected_payout - plain.cost)
    assert 0 <= plain.expected_top <= plain.expected_payout
    assert plain.expected_common == pytest.approx(plain.expected_payout - plain.expected_top)
    assert not plain.worth
    # Prudent (default): even a huge jackpot does not count, only 10 to 13 hits
    prudent = money.plan_money(*args, jackpot=5e8, simulations=1500)
    assert not prudent.worth and prudent.top_weight == 0
    # Counting every prize, the huge jackpot makes it worth playing
    full = money.plan_money(*args, jackpot=5e8, simulations=1500, top_weight=1.0)
    assert full.worth and full.expected_top > plain.expected_top


def test_ticket_payout() -> None:
    probs = np.tile([0.5, 0.3, 0.2], (14, 1))
    levels = np.zeros(14, dtype=int)
    results = np.zeros(14, dtype=int)  # every favourite won: 14 hits
    winners = {k: 0.0 for k in money.CATEGORY_HITS}
    payout, best, top = money.ticket_payout(levels, probs, results, winners, 1_000_000)
    assert best == 14 and payout == pytest.approx(0.16 * 1_000_000) == top


def test_estimate_jackpot() -> None:
    draws = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-11-01", "2026-11-04", "2026-11-08"]),
            "jackpot": [1_000_000, 0, 2_000_000],
            "takings": [3_000_000, 1_000_000, 3_000_000],
            "winners_special": [0, 0, 0],
        }
    )
    # Last weekend draw (Sunday 8th) not won: its pool carries over
    assert backtest_money.estimate_jackpot(draws) == pytest.approx(2_000_000 + 225_000)
    draws.loc[2, "winners_special"] = 1
    assert backtest_money.estimate_jackpot(draws) == 0


def test_summarize_money() -> None:
    rows = pd.DataFrame(
        {
            "strategy": ["v4", "v4", "v5", "v5"],
            "played": [True, True, True, False],
            "cost": [36.0, 36.0, 27.0, 0.0],
            "payout": [0.0, 100.0, 50.0, 0.0],
            "net": [-36.0, 64.0, 23.0, 0.0],
            "best_hits": [9, 12, 11, 10],
        }
    )
    summary = backtest_money.summarize_money(rows).set_index("strategy")
    assert summary.loc["v4", "net"] == pytest.approx(28) and summary.loc["v4", "winning"] == 1
    assert summary.loc["v5", "played"] == 1 and summary.loc["v5", "draws"] == 2
    assert summary.loc["v5", "roi"] == pytest.approx(23 / 27)
    assert summary.loc["v4", "12"] == 1
