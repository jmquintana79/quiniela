import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError

from quiniela.prediction.dixon_coles import DixonColesModel
from tests.helpers import simulate_league

STRENGTHS = {"Strong": 0.6, "Good": 0.3, "Mid1": 0.0, "Mid2": 0.0, "Weak": -0.3, "Worst": -0.6}


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return simulate_league(STRENGTHS, "2024-01-06", rounds=60, seed=1)


@pytest.fixture(scope="module")
def model(matches: pd.DataFrame) -> DixonColesModel:
    return DixonColesModel(xi=0.0).fit(matches, matches["FTR"])


def _fixture(home: str, away: str, league: str = "SP1") -> pd.DataFrame:
    return pd.DataFrame({"league": [league], "home_team": [home], "away_team": [away]})


def test_is_a_scikit_learn_classifier() -> None:
    model = DixonColesModel(xi=0.001)
    assert is_classifier(model)
    assert clone(model).get_params()["xi"] == 0.001


def test_recovers_team_ranking_and_home_advantage(model: DixonColesModel) -> None:
    net = (model.attack_ + model.defence_).sort_values()
    assert net.index[-1] == "Strong" and net.index[0] == "Worst"
    assert model.home_advantage_["SP1"] == pytest.approx(0.3, abs=0.12)
    assert model.converged_


def test_probabilities(model: DixonColesModel) -> None:
    probs = model.predict_proba(
        pd.concat([_fixture("Strong", "Worst"), _fixture("Worst", "Strong")])
    )
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert probs[0, 0] > 0.6  # strong team at home is a clear favourite
    assert probs[1, 2] > probs[1, 0]  # and still favourite away
    assert model.predict(_fixture("Strong", "Worst")).tolist() == ["H"]


def test_score_matrix_and_picks(model: DixonColesModel) -> None:
    matrix = model.score_matrix(_fixture("Mid1", "Mid2"))
    assert matrix.shape == (1, 11, 11)
    assert matrix.sum() == pytest.approx(1)
    picks = model.predict_score(_fixture("Strong", "Worst"))
    assert picks.loc[0, "score_home"] > picks.loc[0, "score_away"]
    assert {picks.loc[0, "pleno_home"], picks.loc[0, "pleno_away"]} <= {"0", "1", "2", "M"}


def test_negative_rho_raises_draw_probability(model: DixonColesModel) -> None:
    fixture = _fixture("Mid1", "Mid2")
    draws = {}
    for rho in (-0.15, 0.0):
        model.rho_ = rho
        draws[rho] = model.predict_proba(fixture)[0, 1]
    assert draws[-0.15] > draws[0.0]


def test_unknown_team_and_league_get_averages(model: DixonColesModel) -> None:
    lam, nu = model.expected_goals(_fixture("Newcomer", "Other", league="XX"))
    assert lam[0] > nu[0]  # home advantage still applies
    assert np.isfinite([lam[0], nu[0]]).all()


def test_time_decay_follows_current_form() -> None:
    # A team that was the worst for years and the best in the last season
    old = simulate_league({**STRENGTHS, "Riser": -0.8}, "2020-01-04", rounds=150, seed=2)
    new = simulate_league({**STRENGTHS, "Riser": 0.8}, "2022-11-05", rounds=40, seed=3)
    matches = pd.concat([old, new], ignore_index=True)

    def riser(xi: float) -> float:
        fitted = DixonColesModel(xi=xi, window_days=5000).fit(matches)
        return float(fitted.attack_["Riser"] + fitted.defence_["Riser"])

    assert riser(0.01) > riser(0.0) + 0.5


def test_window_limits_training_matches(matches: pd.DataFrame) -> None:
    fitted = DixonColesModel(window_days=30).fit(matches)
    assert len(fitted.teams_) == len(STRENGTHS)


def test_errors(matches: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError):
        DixonColesModel().predict(matches)
    with pytest.raises(ValueError, match="missing columns"):
        DixonColesModel().fit(matches.drop(columns="FTHG"))
    with pytest.raises(ValueError, match="No played matches"):
        DixonColesModel().fit(matches.assign(FTHG=np.nan))
