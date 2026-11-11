import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier

from quiniela.prediction.market import MarketModel

X = pd.DataFrame(
    {
        "AvgH": [2.0, np.nan, np.nan],
        "AvgD": [4.0, np.nan, np.nan],
        "AvgA": [4.0, np.nan, np.nan],
        "B365H": [9.0, 3.0, np.nan],
        "B365D": [9.0, 3.0, np.nan],
        "B365A": [9.0, 3.0, np.nan],
    }
)


def test_implied_probabilities_with_fallback() -> None:
    model = MarketModel().fit(X)
    probs = model.predict_proba(X)
    np.testing.assert_allclose(probs[0], [0.5, 0.25, 0.25])  # market average wins
    np.testing.assert_allclose(probs[1], [1 / 3] * 3)  # Bet365 fallback
    assert np.isnan(probs[2]).all()  # no odds: no prediction
    assert model.predict(X).tolist() == ["H", "H", None]


def test_is_a_scikit_learn_classifier() -> None:
    assert is_classifier(MarketModel())
    assert clone(MarketModel(prefixes=("Avg",))).get_params() == {"prefixes": ("Avg",)}
    assert np.isnan(MarketModel(prefixes=("Avg",)).fit(X).predict_proba(X)[1]).all()


@pytest.mark.parametrize("bad_odds", [0.0, 1.0])
def test_ignores_invalid_odds(bad_odds: float) -> None:
    df = pd.DataFrame({"AvgH": [bad_odds], "AvgD": [3.0], "AvgA": [3.0]})
    assert np.isnan(MarketModel().fit(df).predict_proba(df)).all()
