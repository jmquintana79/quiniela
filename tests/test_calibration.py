import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier

from quiniela.prediction.calibration import CalibratedModel, OutcomeCalibrator
from quiniela.prediction.dixon_coles import DixonColesModel
from tests.helpers import simulate_league

RNG = np.random.default_rng(0)


def _draw_shy_forecasts(n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """A model that gives draws 15% while they happen 35% of the time."""
    probs = np.tile([0.45, 0.15, 0.40], (n, 1))
    y = RNG.choice(["H", "D", "A"], size=n, p=[0.35, 0.35, 0.30])
    return probs, y


def test_learns_that_draws_are_underestimated() -> None:
    probs, y = _draw_shy_forecasts()
    calibrator = OutcomeCalibrator().fit(probs, y)
    calibrated = calibrator.predict_proba(probs)

    assert calibrator.offsets_[1] > 0.5  # draws boosted
    assert calibrated[0, 1] == pytest.approx(0.35, abs=0.03)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)


def test_draw_margin_is_learned_only_when_it_helps() -> None:
    probs, y = _draw_shy_forecasts()
    # After calibration draws are as likely as home wins: picking them pays off
    assert OutcomeCalibrator().fit(probs, y).draw_margin_ >= 0

    clear = np.tile([0.70, 0.20, 0.10], (500, 1))
    homes = np.array(["H"] * 500)
    assert OutcomeCalibrator().fit(clear, homes).draw_margin_ == 0


def test_draw_rule_decision() -> None:
    calibrator = OutcomeCalibrator(draw_margin=0.1, l2=1e6).fit(
        np.array([[0.5, 0.3, 0.2]]), np.array(["H"])
    )
    probs = np.array([[0.40, 0.32, 0.28], [0.60, 0.25, 0.15], [np.nan, np.nan, np.nan]])
    assert calibrator.predict(probs).tolist() == ["D", "H", None]


def test_is_a_scikit_learn_classifier() -> None:
    assert is_classifier(OutcomeCalibrator())
    assert is_classifier(CalibratedModel(DixonColesModel()))
    assert clone(CalibratedModel(DixonColesModel(), calibration_days=100)).calibration_days == 100
    with pytest.raises(ValueError, match="shape"):
        OutcomeCalibrator().fit(np.ones((3, 2)), ["H", "D", "A"])


def test_calibrated_model_wraps_any_model() -> None:
    matches = simulate_league(
        {"A": 0.4, "B": 0.1, "C": -0.1, "D": -0.4}, "2023-01-07", rounds=104, seed=4
    )
    model = CalibratedModel(DixonColesModel(), calibration_days=180).fit(matches, matches["FTR"])
    X = pd.DataFrame({"league": ["SP1"], "home_team": ["A"], "away_team": ["D"]})

    probs = model.predict_proba(X)
    assert probs.sum() == pytest.approx(1)
    assert model.predict(X).tolist() == ["H"]
    assert model.estimator_.teams_.tolist() == ["A", "B", "C", "D"]  # refit on all the data
    with pytest.raises(ValueError, match="Not enough history"):
        CalibratedModel(DixonColesModel(), calibration_days=10_000).fit(matches, matches["FTR"])
