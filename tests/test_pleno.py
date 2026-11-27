import numpy as np
import pandas as pd
import pytest

from quiniela.prediction import pleno


def _odds(home: float, draw: float, away: float, over: float, under: float) -> dict[str, float]:
    return {"AvgH": home, "AvgD": draw, "AvgA": away, "Avg>2.5": over, "Avg<2.5": under}


def test_over_probability_removes_the_margin() -> None:
    matches = pd.DataFrame(
        [
            _odds(2.0, 3.4, 4.0, 1.9, 1.9),
            {"BbAv>2.5": 2.5, "BbAv<2.5": 1.5},
            {"AvgH": 2.0},
        ]
    )
    over = pleno.over_probability(matches)
    assert over[0] == pytest.approx(0.5)
    assert over[1] == pytest.approx((1 / 2.5) / (1 / 2.5 + 1 / 1.5))
    assert np.isnan(over[2])


def test_score_and_category_matrices() -> None:
    matrix = pleno.score_matrix(1.5, 1.0)
    assert matrix.sum() == pytest.approx(1.0)
    cats = pleno.category_matrix(1.5, 1.0)
    assert cats.shape == (4, 4) and cats.sum() == pytest.approx(1.0)
    # Without the correction it is two independent Poisson variables
    plain = pleno.category_matrix(1.5, 1.0, rho=0.0)
    assert plain[0, 0] == pytest.approx(np.exp(-1.5) * np.exp(-1.0), rel=1e-6)
    # A negative rho moves probability to 0-0 and 1-1 (Dixon-Coles)
    assert cats[0, 0] > plain[0, 0] and cats[1, 1] > plain[1, 1]


def test_fit_goals_matches_the_market() -> None:
    home, away = 1.8, 0.9
    matrix = pleno.score_matrix(home, away)
    goals = np.arange(matrix.shape[0])
    outcome = np.array(
        [
            matrix[goals[:, None] > goals[None, :]].sum(),
            matrix[goals[:, None] == goals[None, :]].sum(),
            matrix[goals[:, None] < goals[None, :]].sum(),
        ]
    )
    over = matrix[(goals[:, None] + goals[None, :]) > 2.5].sum()
    fitted = pleno.fit_goals(outcome, float(over))
    assert fitted == pytest.approx((home, away), abs=0.01)


def test_market_pleno() -> None:
    matches = pd.DataFrame(
        [
            # A heavy home favourite with many goals expected
            _odds(1.25, 6.5, 12.0, 1.45, 2.8),
            # An even, low-scoring match
            _odds(2.8, 2.9, 2.9, 2.6, 1.5),
            # No over / under odds: no pick
            {"AvgH": 2.0, "AvgD": 3.4, "AvgA": 4.0},
        ],
        index=[10, 11, 12],
    )
    picks = pleno.market_pleno(matches)
    assert list(picks.index) == [10, 11, 12]
    assert (picks.loc[10, "pleno_home"], picks.loc[10, "pleno_away"]) in {("2", "0"), ("M", "0")}
    assert picks.loc[10, "goals_home"] > 2 * picks.loc[10, "goals_away"]
    assert picks.loc[11, "pleno_home"] in {"0", "1"} and picks.loc[11, "pleno_away"] in {"0", "1"}
    assert 0 < picks.loc[11, "pleno_prob"] < 1
    assert np.array(picks.loc[11, "categories"]).sum() == pytest.approx(1.0, abs=1e-4)
    assert pd.isna(picks.loc[12, "pleno_home"]) and np.isnan(picks.loc[12, "pleno_prob"])


def test_value_pick() -> None:
    cats = np.full((4, 4), 1 / 16)
    cats[1, 1] = 0.2
    cats /= cats.sum()
    # The public piles on 1 for both teams: 1-1 is the most likely, but far more shared
    public = {"0": 10, "1": 70, "2": 10, "M": 10}
    value = pleno.value_pick(cats, public, public)
    assert value is not None and value["ratio"] > 1
    assert (value["home"], value["away"]) != ("1", "1")
    assert value["public"] == pytest.approx(0.1 * 0.1)
    assert value["prob"] == pytest.approx(cats[0, 0])
    assert pleno.value_pick(cats, {}, public) is None
