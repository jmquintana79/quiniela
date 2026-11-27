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
            "strategy": ["v3d", "v3d", "v3e", "v3e"],
            "played": [True, True, True, False],
            "cost": [36.0, 36.0, 27.0, 0.0],
            "payout": [0.0, 100.0, 50.0, 0.0],
            "net": [-36.0, 64.0, 23.0, 0.0],
            "best_hits": [9, 12, 11, 10],
        }
    )
    summary = backtest_money.summarize_money(rows).set_index("strategy")
    assert summary.loc["v3d", "net"] == pytest.approx(28) and summary.loc["v3d", "winning"] == 1
    assert summary.loc["v3e", "played"] == 1 and summary.loc["v3e", "draws"] == 2
    assert summary.loc["v3e", "roi"] == pytest.approx(23 / 27)
    assert summary.loc["v3d", "12"] == 1


def test_plan_big_win(model: money.PrizeModel) -> None:
    probs = np.random.default_rng(5).dirichlet([5, 3, 3], size=14)
    args = (probs, probs, model, 3_000_000)
    big = money.plan_big_win(*args, simulations=3000, per_mix=5, rng=np.random.default_rng(1))
    # Within the weekly budget, and a proper distribution of outcomes
    assert big.cost <= money.BIG_WIN_MAX_COST and big.plan.bets >= money.MIN_BETS
    assert sum(chance for *_, chance in big.outcomes) == pytest.approx(1)
    assert big.outcomes[0][0] == "Nothing back"
    # 14 hits: exact product of the coverage of every match
    assert big.chance_14 == pytest.approx(float(np.prod(big.plan.coverage)))
    assert 0 < big.chance <= big.chance_13 + big.chance_14
    assert big.matchdays_per_win == pytest.approx(1 / big.chance)
    # A larger prize is rarer; a jackpot makes it likelier
    larger = money.plan_big_win(*args, threshold=100_000, simulations=3000, per_mix=5)
    jackpot = money.plan_big_win(*args, jackpot=5e6, simulations=3000, per_mix=5)
    assert larger.chance < big.chance < jackpot.chance


def test_big_win_chance_grows_with_columns(model: money.PrizeModel) -> None:
    columns = np.zeros((2, 3, 15))
    columns[0, :, 13] = 1
    columns[1, :, 13] = 4  # more 13-hit columns: more money if they win
    s = np.array([12.0, 15.0, 18.0])
    chance = money.big_win_chance(columns, model.at_surprise(s), model, 3_000_000, 0.0, 10_000)
    assert (chance[1] >= chance[0]).all()
    assert chance[0, 2] > chance[0, 0]  # more surprise, fewer winners, larger prize


def test_hit_shares_match_brute_force() -> None:
    picked = np.array([0.8, 0.5, 0.3, 0.6])
    dist = money.hit_shares(picked)
    expected = np.zeros(5)
    for right in product([0, 1], repeat=4):
        chance = np.prod([p if r else 1 - p for p, r in zip(picked, right, strict=True)])
        expected[sum(right)] += chance
    assert dist == pytest.approx(expected)
    # Leading dimensions are kept (one distribution per simulated draw)
    many = money.hit_shares(np.stack([picked, picked[::-1]]))
    assert many.shape == (2, 5)
    assert many[1] == pytest.approx(expected)


def test_public_prize_model_winners(model: money.PrizeModel) -> None:
    public = money.PublicPrizeModel(model)
    bets = 2_000_000
    picked = np.full((1, 14), 0.5)
    shares = public.log_shares(picked, bets)
    winners = money.category_winners(public, shares, bets)
    # Every bet picks at random: 14 hits with chance 0.5 ** 14
    assert winners[14][0] == pytest.approx(bets * 0.5**14)
    assert winners[13][0] == pytest.approx(bets * 14 * 0.5**14)
    # Spread and Pleno al 15 share are the market model's
    assert public.sd == model.sd
    assert public.special_share == model.special_share


def test_public_shares_change_the_prizes(model: money.PrizeModel) -> None:
    public = money.PublicPrizeModel(model)
    bets = 2_000_000
    # The same results: popular with the public (many winners) or not (few, larger prizes)
    popular = public.log_shares(np.full((1, 14), 0.8), bets)
    unpopular = public.log_shares(np.full((1, 14), 0.3), bets)
    small, _ = money.category_prizes(public, popular, bets, 0.0)
    large, _ = money.category_prizes(public, unpopular, bets, 0.0)
    assert (large[0, 11:] > small[0, 11:]).all()


def test_big_win_with_public_shares(model: money.PrizeModel) -> None:
    probs = np.tile([0.5, 0.3, 0.2], (14, 1))
    public = money.PublicPrizeModel(model)

    def plan(shares: np.ndarray) -> money.BigWinPlan:
        return money.plan_big_win(probs, shares, public, 2_000_000, simulations=2000, per_mix=5)

    agrees = plan(probs)
    # The public piles on the away win, the least likely result: the likely results
    # leave few winners and large prizes
    contrary = plan(np.tile([0.1, 0.1, 0.8], (14, 1)))
    assert 0 <= agrees.chance <= 1
    assert contrary.cost <= money.BIG_WIN_MAX_COST
    assert contrary.chance > agrees.chance


def test_pleno_picks_prefer_the_market() -> None:
    matched = pd.DataFrame(
        {"position": [15, 15, 3], "league": ["SP1", "SP1", "SP1"], "game": ["a", "b", "c"]}
    )
    matches = pd.DataFrame(
        {
            "league": ["SP1", "SP1", "SP1"],
            "game": ["a", "b", "c"],
            "AvgH": [1.25, 2.0, 2.0],
            "AvgD": [6.5, 3.4, 3.4],
            "AvgA": [12.0, 4.0, 4.0],
            "Avg>2.5": [1.45, np.nan, 1.9],
            "Avg<2.5": [2.8, np.nan, 1.9],
        }
    )
    backtest = pd.DataFrame(
        {
            "model": "dixon_coles",
            "league": "SP1",
            "game": ["a", "b", "c"],
            "pleno_home": ["0", "1", "1"],
            "pleno_away": ["0", "1", "1"],
        }
    )
    picks = backtest_money._pleno_picks(backtest, matched, matches).set_index("game")
    # Match a has over / under odds: the market's pick (a heavy home favourite)
    assert picks.loc["a", "pleno_home"] in {"2", "M"} and picks.loc["a", "pleno_away"] == "0"
    # Match b has none: Dixon-Coles' pick
    assert (picks.loc["b", "pleno_home"], picks.loc["b", "pleno_away"]) == ("1", "1")
