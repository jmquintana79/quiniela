import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError

from quiniela.prediction import market_blend, pipeline
from quiniela.prediction.dixon_coles import DixonColesModel
from quiniela.prediction.market_blend import MarketBlendModel
from quiniela.prediction.registry import MODELS
from tests.helpers import simulate_league

STRENGTHS = {
    "Strong": 0.6,
    "Good": 0.3,
    "Mid1": 0.0,
    "Mid2": 0.0,
    "Weak": -0.3,
    "Worst": -0.6,
    "Low1": -0.1,
    "High1": 0.1,
}
HOME_ADVANTAGE = 0.3


def _true_probabilities(matches: pd.DataFrame) -> np.ndarray:
    """1X2 probabilities of the Poisson process that ``simulate_league`` draws from."""
    s_home = matches["home_team"].map(STRENGTHS).to_numpy()
    s_away = matches["away_team"].map(STRENGTHS).to_numpy()
    lam = np.exp(0.1 + HOME_ADVANTAGE + s_home - s_away)
    nu = np.exp(0.1 + s_away - s_home)
    goals = np.arange(11)
    matrix = poisson.pmf(goals[None, :, None], lam[:, None, None]) * poisson.pmf(
        goals[None, None, :], nu[:, None, None]
    )
    home = np.tril(np.ones((11, 11)), k=-1)
    probs = np.stack(
        [
            (matrix * home).sum(axis=(1, 2)),
            np.trace(matrix, axis1=1, axis2=2),
            (matrix * home.T).sum(axis=(1, 2)),
        ],
        axis=1,
    )
    normalised: np.ndarray = probs / probs.sum(axis=1, keepdims=True)
    return normalised


def _with_odds(matches: pd.DataFrame, margin: float = 0.05) -> pd.DataFrame:
    """Add bookmaker odds built from the true probabilities plus a margin."""
    odds = 1 / (_true_probabilities(matches) * (1 + margin))
    return matches.assign(AvgH=odds[:, 0], AvgD=odds[:, 1], AvgA=odds[:, 2])


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    # About 1,000 matches over 2.5 years: enough out-of-sample forecasts to fit
    # the combination with the default window
    return _with_odds(
        simulate_league(STRENGTHS, "2022-01-08", rounds=250, home_advantage=HOME_ADVANTAGE, seed=3)
    )


@pytest.fixture(scope="module")
def model(matches: pd.DataFrame) -> MarketBlendModel:
    return MarketBlendModel(dixon_coles=DixonColesModel(xi=0.0)).fit(matches, matches["FTR"])


def _fixture(home: str, away: str, odds: bool = True) -> pd.DataFrame:
    fixture = pd.DataFrame({"league": ["SP1"], "home_team": [home], "away_team": [away]})
    return _with_odds(fixture) if odds else fixture


def test_pool_recovers_known_weights() -> None:
    rng = np.random.default_rng(0)
    dc = rng.dirichlet([4, 3, 3], size=4000)
    market = rng.dirichlet([4, 3, 3], size=4000)
    truth = market_blend._pool(
        np.array([0.3, 0.9, 0.1, -0.1]), market_blend._log(dc), market_blend._log(market)
    )
    y = np.array([rng.choice(["H", "D", "A"], p=p) for p in truth])
    np.testing.assert_allclose(
        market_blend.fit_pool(dc, market, y), [0.3, 0.9, 0.1, -0.1], atol=0.15
    )


def test_is_a_scikit_learn_classifier() -> None:
    model = MarketBlendModel(blend_days=500, dixon_coles=DixonColesModel(xi=0.001))
    assert is_classifier(model)
    params = clone(model).get_params()
    assert params["blend_days"] == 500 and params["dixon_coles__xi"] == 0.001


def test_not_fitted_raises() -> None:
    with pytest.raises(NotFittedError):
        MarketBlendModel().predict_proba(_fixture("Strong", "Worst"))


def test_probabilities(model: MarketBlendModel) -> None:
    X = pd.concat([_fixture("Strong", "Worst"), _fixture("Worst", "Strong")], ignore_index=True)
    probs = model.predict_proba(X)
    assert model.classes_.tolist() == ["H", "D", "A"]
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert probs[0, 0] > 0.6 and probs[1, 2] > probs[1, 0]
    assert model.predict(X).tolist() == ["H", "A"]


def test_combination_is_trained_and_trusts_the_market(model: MarketBlendModel) -> None:
    # The odds are the true probabilities, so the combination should follow them
    assert model.weights_ is not None
    assert model.weights_[1] > model.weights_[0]  # more weight on the market
    assert model.n_blend_matches_ >= market_blend.MIN_BLEND_MATCHES
    X = pd.concat(
        [_fixture(h, a) for h in STRENGTHS for a in STRENGTHS if h != a], ignore_index=True
    )
    truth = _true_probabilities(X)
    blended = np.abs(model.predict_proba(X) - truth).mean()
    assert blended <= np.abs(model.dixon_coles_.predict_proba(X) - truth).mean() + 0.01


def test_without_odds_falls_back_to_dixon_coles(model: MarketBlendModel) -> None:
    X = pd.concat(
        [_fixture("Strong", "Worst", odds=False), _fixture("Mid1", "Mid2", odds=False)],
        ignore_index=True,
    )
    np.testing.assert_allclose(model.predict_proba(X), model.dixon_coles_.predict_proba(X))


def test_mixed_rows_only_blend_those_with_odds(model: MarketBlendModel) -> None:
    X = pd.concat(
        [_fixture("Good", "Weak"), _fixture("Good", "Weak", odds=False)], ignore_index=True
    )
    probs = model.predict_proba(X)
    np.testing.assert_allclose(probs[1], model.dixon_coles_.predict_proba(X)[1])
    assert not np.allclose(probs[0], probs[1])


def test_too_few_matches_with_odds_uses_dixon_coles(matches: pd.DataFrame) -> None:
    no_odds = matches.drop(columns=["AvgH", "AvgD", "AvgA"])
    model = MarketBlendModel(dixon_coles=DixonColesModel(xi=0.0)).fit(no_odds)
    assert model.weights_ is None
    X = _fixture("Strong", "Worst")
    np.testing.assert_allclose(model.predict_proba(X), model.dixon_coles_.predict_proba(X))


def test_out_of_sample_uses_only_earlier_blocks(matches: pd.DataFrame) -> None:
    model = MarketBlendModel(blend_days=60, refit_days=7, dixon_coles=DixonColesModel(xi=0.0))
    probs = model.out_of_sample(matches)
    cutoff = matches["date"].max() - pd.Timedelta(days=60)
    assert np.isnan(probs[(matches["date"] < cutoff - pd.Timedelta(days=7)).to_numpy()]).all()

    # The forecast of the last week equals a Dixon-Coles fitted on the weeks before it
    last = (matches["date"] == matches["date"].max()).to_numpy()
    expected = DixonColesModel(xi=0.0).fit(matches[~last]).predict_proba(matches[last])
    np.testing.assert_allclose(probs[last], expected)


def test_out_of_sample_ignores_later_results(matches: pd.DataFrame) -> None:
    # Changing the goals of the last week must not change the forecasts before it
    model = MarketBlendModel(blend_days=60, dixon_coles=DixonColesModel(xi=0.0))
    last = (matches["date"] == matches["date"].max()).to_numpy()
    changed = matches.copy()
    changed.loc[last, ["FTHG", "FTAG"]] = 7
    np.testing.assert_allclose(
        model.out_of_sample(changed)[~last], model.out_of_sample(matches)[~last], equal_nan=True
    )


def test_cache_does_not_mix_datasets(matches: pd.DataFrame) -> None:
    # Same dates and shape, different goals: the cache must not reuse the fits
    model = MarketBlendModel(blend_days=30, dixon_coles=DixonColesModel(xi=0.0))
    swapped = matches.assign(FTHG=matches["FTAG"], FTAG=matches["FTHG"])
    original, mirrored = model.out_of_sample(matches), model.out_of_sample(swapped)
    recent = ~np.isnan(original).any(axis=1)
    assert recent.any()
    assert not np.allclose(original[recent], mirrored[recent])


def test_registered_and_runs_walk_forward(matches: pd.DataFrame) -> None:
    assert MODELS["market_blend"].version == "v3"
    assert list(MODELS).index("market_blend") == list(MODELS).index("dixon_coles") + 1

    last_day = matches["date"].max()
    target = matches[matches["date"] == last_day].assign(matchday=1)
    out = pipeline.walk_forward("market_blend", target, matches)
    assert len(out) == len(target)
    np.testing.assert_allclose(out[["p_H", "p_D", "p_A"]].sum(axis=1), 1)
    assert out["score_home"].isna().all()  # no exact-score picks in v3
