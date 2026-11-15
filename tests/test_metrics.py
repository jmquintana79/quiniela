import numpy as np
import pandas as pd
import pytest

from quiniela.evaluation import metrics

Y = np.array(["H", "D", "A", "H"])
PERFECT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
UNIFORM = np.full((4, 3), 1 / 3)


def test_perfect_forecast() -> None:
    assert metrics.hit_rate(PERFECT, Y) == 1
    assert metrics.log_loss(PERFECT, Y) == pytest.approx(0, abs=1e-9)
    assert metrics.brier_score(PERFECT, Y) == 0
    assert metrics.ranked_probability_score(PERFECT, Y) == 0


def test_uniform_forecast() -> None:
    assert metrics.log_loss(UNIFORM, Y) == pytest.approx(np.log(3))
    assert metrics.brier_score(UNIFORM, Y) == pytest.approx(2 / 3)


def test_rps_rewards_near_misses() -> None:
    # Home win happened: predicting a draw is closer than predicting an away win
    y = np.array(["H"])
    draw = metrics.ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), y)
    away = metrics.ranked_probability_score(np.array([[0.0, 0.0, 1.0]]), y)
    assert draw == pytest.approx(0.5)
    assert away == pytest.approx(1.0)
    # Brier treats both mistakes the same
    assert metrics.brier_score(np.array([[0.0, 1.0, 0.0]]), y) == metrics.brier_score(
        np.array([[0.0, 0.0, 1.0]]), y
    )


def test_calibration_table_pools_outcomes() -> None:
    probs = np.array([[0.75, 0.15, 0.10], [0.72, 0.18, 0.10]])
    table = metrics.calibration_table(probs, np.array(["H", "A"]))
    top = table.set_index("bin").loc[0.7]
    assert top["count"] == 2
    assert top["observed"] == 0.5
    assert table["count"].sum() == 6


def test_evaluate_uses_given_signs() -> None:
    probs = np.array([[0.5, 0.3, 0.2], [0.4, 0.35, 0.25]])
    y = pd.Series(["H", "D"])
    assert metrics.evaluate(probs, y)["hit_rate"] == 0.5
    result = metrics.evaluate(probs, y, predicted=np.array(["H", "D"]))
    assert result["hit_rate"] == 1
    assert result["draw_picks"] == 0.5
    assert result["draw_rate"] == 0.5
    assert metrics.evaluate(np.empty((0, 3)), np.array([])) == {"matches": 0}


def test_macro_f1_and_contingency() -> None:
    predicted = np.array(["H", "H", "H", "A", "A", "H"])
    real = np.array(["H", "D", "H", "A", "H", "D"])
    # H: 2 right of 4 predicted, 3 real -> 4/7; D never predicted -> 0; A: 1 of 2, 1 real -> 2/3
    assert metrics.macro_f1(predicted, real) == pytest.approx((4 / 7 + 0 + 2 / 3) / 3)
    table = metrics.contingency_table(predicted, real)
    # Rows: real H, D, A; columns: predicted H, D, A
    assert table == [[2, 0, 1], [2, 0, 0], [0, 0, 1]]
    assert sum(map(sum, table)) == len(real)
    probs = np.tile([0.5, 0.3, 0.2], (len(real), 1))
    assert metrics.evaluate(probs, real, predicted)["macro_f1"] == pytest.approx(
        metrics.macro_f1(predicted, real)
    )


def test_hit_rate_by_result_counts_doubles_and_triples() -> None:
    picks = np.array(["H", "D", "HDA", "HD", "A"], dtype=object)
    real = np.array(["H", "H", "D", "D", "H"])
    out = metrics.hit_rate_by_result(picks, real)
    # Real 1: hit, miss, miss; real X: the triple and the double 1X both cover it; no real 2
    assert out["hit_H"] == pytest.approx(1 / 3) and out["hit_D"] == 1.0 and out["hit_A"] is None
    assert out["real_H"] == pytest.approx(0.6) and out["real_A"] == 0.0
