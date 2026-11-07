import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError

from quiniela.prediction.head_to_head import HeadToHeadModel
from tests.helpers import make_matches

TRAIN = make_matches(
    [
        ("SP1", "2020-01-01", "Madrid", "Barca", 2, 0),
        ("SP1", "2021-01-01", "Madrid", "Barca", 3, 1),
        ("SP1", "2022-01-01", "Madrid", "Barca", 1, 1),
        ("SP1", "2022-02-01", "Barca", "Madrid", 0, 2),
        ("SP2", "2022-03-01", "Cadiz", "Oviedo", 0, 0),
        ("SP2", "2022-04-01", "Oviedo", "Cadiz", 0, 0),
    ]
)


def _fit(alpha: float = 2.0) -> HeadToHeadModel:
    return HeadToHeadModel(alpha=alpha).fit(
        TRAIN[["league", "home_team", "away_team"]], TRAIN["FTR"]
    )


def test_is_a_scikit_learn_classifier() -> None:
    model = HeadToHeadModel(alpha=3.0)
    assert is_classifier(model)
    assert clone(model).get_params() == {"alpha": 3.0}


def test_unsmoothed_probabilities_are_head_to_head_frequencies() -> None:
    model = _fit(alpha=0)
    X = pd.DataFrame({"home_team": ["Madrid"], "away_team": ["Barca"], "league": ["SP1"]})

    np.testing.assert_allclose(model.predict_proba(X), [[2 / 3, 1 / 3, 0]])
    assert model.predict(X).tolist() == ["H"]
    assert model.classes_.tolist() == ["H", "D", "A"]


def test_venue_matters() -> None:
    # Barca at home against Madrid: one meeting, an away win
    X = pd.DataFrame({"home_team": ["Barca"], "away_team": ["Madrid"], "league": ["SP1"]})
    assert _fit(alpha=0).predict(X).tolist() == ["A"]


def test_smoothing_towards_league_prior() -> None:
    model = _fit(alpha=2)
    X = pd.DataFrame({"home_team": ["Madrid"], "away_team": ["Barca"], "league": ["SP1"]})
    # SP1 prior: 2 H, 1 D, 1 A out of 4 matches
    prior = np.array([0.5, 0.25, 0.25])
    expected = (np.array([2, 1, 0]) + 2 * prior) / (3 + 2)
    np.testing.assert_allclose(model.predict_proba(X), [expected])


def test_unknown_fixture_gets_league_prior() -> None:
    model = _fit()
    X = pd.DataFrame(
        {"home_team": ["Nobody", "Nobody"], "away_team": ["Else", "Else"], "league": ["SP2", "XX"]}
    )
    probs = model.predict_proba(X)
    np.testing.assert_allclose(probs[0], [0, 1, 0])  # SP2: only draws
    # Unknown league: global prior (2 home wins, 3 draws, 1 away win)
    np.testing.assert_allclose(probs[1], [2 / 6, 3 / 6, 1 / 6])
    np.testing.assert_allclose(probs.sum(axis=1), 1)


def test_history_size() -> None:
    X = pd.DataFrame({"home_team": ["Madrid", "Cadiz"], "away_team": ["Barca", "Madrid"]})
    assert _fit().history_size(X).tolist() == [3, 0]


def test_errors() -> None:
    with pytest.raises(NotFittedError):
        HeadToHeadModel().predict(TRAIN)
    with pytest.raises(ValueError, match="missing columns"):
        HeadToHeadModel().fit(TRAIN[["home_team"]], TRAIN["FTR"])
    with pytest.raises(ValueError, match="Unknown results"):
        HeadToHeadModel().fit(TRAIN, ["1"] * len(TRAIN))
    with pytest.raises(ValueError, match="alpha"):
        HeadToHeadModel(alpha=-1).fit(TRAIN, TRAIN["FTR"])
