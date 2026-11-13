from pathlib import Path

import pandas as pd
import pytest

from quiniela.prediction import pipeline
from tests.helpers import make_matches

HISTORY = make_matches(
    [
        ("SP1", "2020-01-01", "A", "B", 0, 1),
        ("SP1", "2021-01-01", "A", "B", 0, 2),
        ("SP1", "2021-02-01", "C", "D", 1, 0),
    ]
)


def _current() -> pd.DataFrame:
    current = make_matches(
        [
            ("SP1", "2026-08-15", "A", "B", 3, 0),  # matchday 1: home win
            ("SP1", "2026-08-15", "C", "D", 0, 0),
            ("SP1", "2026-08-22", "A", "B", None, None),  # matchday 2: pending
            ("SP1", "2026-08-22", "C", "D", None, None),
        ]
    )
    return current.assign(round=[1, 1, 2, 2], matchday=[1, 1, 2, 2])


def test_predicts_last_matchday_by_default() -> None:
    out = pipeline.predict_matchdays("head_to_head", history=HISTORY, current=_current())
    assert out["matchday"].unique().tolist() == [2]
    assert list(out.columns) == pipeline.PREDICTION_COLUMNS
    assert out[["p_H", "p_D", "p_A"]].sum(axis=1).round(6).eq(1).all()
    # Confidence: probability of the predicted sign; margin over the second one
    probs = out[["p_H", "p_D", "p_A"]].to_numpy()
    assert (out["confidence"] == probs.max(axis=1)).all()
    ordered = sorted(probs[0], reverse=True)
    assert out["margin"].iloc[0] == pytest.approx(ordered[0] - ordered[1])


def test_training_uses_only_earlier_matches() -> None:
    first = pipeline.predict_matchdays(
        "head_to_head", matchdays=[1], history=HISTORY, current=_current()
    )
    second = pipeline.predict_matchdays(
        "head_to_head", matchdays=[2], history=HISTORY, current=_current()
    )

    # Matchday 1 only knows the two A-B away wins; matchday 2 also saw A beat B 3-0
    p_first = first.set_index("home_team").loc["A", "p_H"]
    p_second = second.set_index("home_team").loc["A", "p_H"]
    assert p_second > p_first


def test_unknown_model_and_matchday() -> None:
    with pytest.raises(KeyError):
        pipeline.predict_matchdays("nope", history=HISTORY, current=_current())
    with pytest.raises(ValueError, match="not found"):
        pipeline.predict_matchdays(
            "head_to_head", matchdays=[9], history=HISTORY, current=_current()
        )


def test_save_merges_by_match(tmp_path: Path) -> None:
    all_days = pipeline.predict_matchdays(
        "head_to_head", matchdays=[1, 2], history=HISTORY, current=_current()
    )
    pipeline.save_predictions(all_days, "head_to_head", predictions_dir=tmp_path)

    again = pipeline.predict_matchdays(
        "head_to_head", matchdays=[2], history=HISTORY, current=_current()
    )
    again["prediction"] = "D"
    path = pipeline.save_predictions(again, "head_to_head", predictions_dir=tmp_path)

    saved = pd.read_parquet(path)
    assert len(saved) == 4  # matchday 2 replaced, matchday 1 kept
    assert saved.loc[saved["matchday"] == 2, "prediction"].eq("D").all()
    assert pipeline.load_predictions(tmp_path).keys() == {"head_to_head"}


def test_resolve_models() -> None:
    assert pipeline.resolve_models("all") == list(pipeline.MODELS)
    assert pipeline.resolve_models("market") == ["market"]
    with pytest.raises(KeyError):
        pipeline.resolve_models("nope")


def test_models_never_see_the_result_they_predict() -> None:
    from quiniela.prediction.registry import MODELS, ModelSpec

    seen: list[set[str]] = []

    class Spy:
        classes_ = ["H", "D", "A"]

        def fit(self, X: pd.DataFrame, y: pd.Series) -> "Spy":
            return self

        def predict_proba(self, X: pd.DataFrame) -> list[list[float]]:
            seen.append(set(X.columns))
            return [[0.5, 0.3, 0.2]] * len(X)

        def predict(self, X: pd.DataFrame) -> list[str]:
            return ["H"] * len(X)

    MODELS["spy"] = ModelSpec(name="Spy", factory=Spy, description="")
    try:
        pipeline.predict_matchdays("spy", matchdays=[1], history=HISTORY, current=_current())
    finally:
        del MODELS["spy"]
    assert seen and not seen[0] & {"FTHG", "FTAG", "FTR"}


def test_dixon_coles_predictions_include_scores() -> None:
    from tests.helpers import simulate_league

    history = simulate_league({"A": 0.3, "B": 0.0, "C": -0.1, "D": -0.2}, "2024-01-06", 30)
    current = _current().assign(home_team=["A", "C", "A", "C"], away_team=["B", "D", "B", "D"])
    out = pipeline.predict_matchdays("dixon_coles", history=history, current=current)
    assert out[pipeline.SCORE_COLUMNS].notna().all().all()
    assert set(out["pleno_home"]) <= {"0", "1", "2", "M"}


def test_predict_upcoming(tmp_path: Path) -> None:
    from quiniela.data.prizes import load_upcoming
    from tests.test_prizes import _upcoming_file

    from tests.helpers import make_matches

    _upcoming_file(tmp_path)
    ticket = load_upcoming(tmp_path)
    assert ticket is not None
    rows = [
        ("SP2", "2026-10-03", "Valladolid", "Castellon", 1, 1),
        ("SP2", "2026-10-10", "Castellon", "Valladolid", 0, 2),
    ]
    current = make_matches(rows).assign(season="2026-27")
    history = make_matches([("SP2", "2025-10-03", "Valladolid", "Castellon", 2, 0)]).assign(
        season="2025-26"
    )

    matches, predictions, info = pipeline.predict_upcoming(ticket, history=history, current=current)

    assert matches["official_position"].tolist() == [1, 2]
    assert matches["league"].tolist() == ["SP2", pipeline.WOMEN_LEAGUE]
    assert matches["home_team"].tolist() == ["Valladolid", "Alavés"]
    # No football-data fixture yet: no kick-off, no odds, no result
    assert info == {"ours": 1, "fixtures": 0, "odds": 0, "results": 0, "public": 1}
    assert not matches["_kickoff_known"].any()
    # Only the La Liga / Segunda match is predicted; the market has no odds
    dc = predictions["dixon_coles"]
    assert len(dc) == 1 and dc["prediction"].notna().all()
    assert predictions["market"]["p_H"].isna().all()
    # v3: the model for the Segunda match, the public's picks for the women's one
    v3 = predictions["market_blend"].set_index("home_team")
    assert v3.loc["Valladolid", "source"] == "model"
    assert v3.loc["Alavés", "source"] == "public"
    assert v3.loc["Alavés", "p_H"] == pytest.approx(0.35)
    assert matches[[f"_fill_p_{o}" for o in "HDA"]].sum(axis=1).round(6).eq(1).all()
