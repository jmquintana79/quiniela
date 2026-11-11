import json
from pathlib import Path

import pandas as pd
import pytest

from quiniela.visualization import dashboards


def _matches(rows: list[tuple[str, str, str, int, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "FTHG", "FTAG"])
    df["date"] = pd.to_datetime(df["date"])
    df["FTR"] = ["H" if h > a else "A" if h < a else "D" for h, a in zip(df.FTHG, df.FTAG)]
    return df


MATCHES = _matches(
    [
        ("2026-08-15", "A", "B", 2, 0),
        ("2026-08-22", "B", "C", 1, 1),
        ("2026-08-29", "C", "A", 0, 3),
        ("2026-09-05", "B", "A", 2, 1),
    ]
)


def test_standings_ranking_and_form() -> None:
    table = dashboards.standings(MATCHES)

    assert table["team"].tolist() == ["A", "B", "C"]
    a = table.iloc[0]
    assert (a.played, a.won, a.lost, a.gf, a.ga, a.gd, a.points) == (3, 2, 1, 6, 2, 4, 6)
    assert a.form == "WWL"


def test_standings_two_points_per_win() -> None:
    table = dashboards.standings(MATCHES, points_per_win=2)
    assert table.set_index("team").loc["A", "points"] == 4


def test_implied_probabilities_remove_margin_and_fall_back() -> None:
    fixtures = pd.DataFrame(
        {
            "AvgH": [2.0, None],
            "AvgD": [4.0, None],
            "AvgA": [4.0, None],
            "B365H": [1.5, 3.0],
            "B365D": [4.0, 3.0],
            "B365A": [6.0, 3.0],
        }
    )
    probs = dashboards.implied_probabilities(fixtures)

    assert probs.sum(axis=1).round(6).tolist() == [1.0, 1.0]
    assert probs.iloc[0].tolist() == pytest.approx([0.5, 0.25, 0.25])  # market average
    assert probs.iloc[1].tolist() == pytest.approx([1 / 3] * 3)  # Bet365 fallback


def test_season_summary() -> None:
    summary = dashboards.season_summary(MATCHES)
    assert summary == {"matches": 4, "home": 0.5, "draw": 0.25, "away": 0.25, "goals": 2.5}


def test_render_embeds_data_safely(tmp_path: Path) -> None:
    payload = {"leagues": [{"name": "</script><b>x</b>"}]}
    html = dashboards.render("current.html", payload, tmp_path / "current.html").read_text()

    assert "/*__" not in html  # every placeholder replaced
    assert "</script><b>" not in html
    data = html.split("const DATA = ", 1)[1].split(";</script>", 1)[0]
    assert json.loads(data.replace("<\\/", "</")) == payload


def test_historical_payload_sorts_seasons_by_year(tmp_path: Path) -> None:
    for season in ("2000-01", "1999-00"):
        path = tmp_path / "SP1" / f"{season}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        MATCHES.to_parquet(path)

    league = dashboards.historical_payload(tmp_path)["leagues"][0]

    assert [s["season"] for s in league["seasons"]] == ["1999-00", "2000-01"]
    # 1999-00 already used 3 points per win
    assert league["seasons"][0]["standings"][0]["points"] == 6


def test_predictions_payload_scores_models_and_tickets() -> None:
    from tests.helpers import make_matches

    # Matchday 1: 14 La Liga matches, all home wins, played
    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", 1, 0) for i in range(14)]
    current = make_matches(rows).assign(round=1, matchday=1)
    predictions = current[["league", "game"]].assign(
        prediction=["H"] * 11 + ["A"] * 3, p_H=0.5, p_D=0.3, p_A=0.2
    )

    payload = dashboards.predictions_payload(current, {"head_to_head": predictions})

    assert [m["key"] for m in payload["models"]] == ["head_to_head"]
    summary = payload["matchdays"][0]["summary"]["head_to_head"]
    assert (summary["hits"], summary["misses"], summary["pending"]) == (11, 3, 0)
    assert summary["ticket"]["category"] == "4th category"
    assert summary["missing"] == 0
    assert payload["totals"]["head_to_head"] == {
        "hits": 11,
        "misses": 3,
        "rate": round(11 / 14, 4),
        "missing": 0,
        "prizes": 1,
        "tickets": 1,
    }
    first = payload["matchdays"][0]["matches"][0]
    assert first["head_to_head_sign"] == "H" and first["ticket"] == 1


def test_predictions_payload_without_predictions() -> None:
    from tests.helpers import make_matches

    current = make_matches([("SP1", "2026-08-15", "A", "B", None, None)]).assign(
        round=1, matchday=1
    )
    payload = dashboards.predictions_payload(current, {})
    assert payload["models"] == [] and payload["totals"] == {}
    assert payload["matchdays"][0]["played"] == 0


def test_predictions_payload_counts_matches_without_prediction() -> None:
    from tests.helpers import make_matches

    # 15 played home wins; the model has no prediction for two of them
    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", 1, 0) for i in range(15)]
    current = make_matches(rows).assign(round=1, matchday=1)
    predictions = (
        current[["league", "game"]].iloc[2:].assign(prediction="H", p_H=0.5, p_D=0.3, p_A=0.2)
    )

    payload = dashboards.predictions_payload(current, {"head_to_head": predictions})

    summary = payload["matchdays"][0]["summary"]["head_to_head"]
    # Missing predictions are their own category, outside the hit rate
    assert (summary["hits"], summary["misses"], summary["missing"]) == (13, 0, 2)
    assert summary["rate"] == 1.0
    assert summary["missing_in_ticket"] == 2
    # On the ticket they are wrong lines: 12 of 14 hits
    assert summary["ticket"]["hits"] == 12
    assert summary["ticket"]["category"] == "3rd category"
    assert payload["totals"]["head_to_head"]["missing"] == 2
    missing = [m for m in payload["matchdays"][0]["matches"] if m["head_to_head_sign"] is None]
    assert len(missing) == 2


def test_evaluation_payload() -> None:
    import numpy as np

    rows = []
    for model in ("head_to_head", "market"):
        for i, (league, result) in enumerate([("SP1", "H"), ("SP2", "D"), ("SP1", "A")]):
            rows.append(
                {
                    "model": model,
                    "league": league,
                    "season": "2024-25",
                    "game": f"g{i}",
                    "matchday": 1,
                    "date": pd.Timestamp("2024-08-17"),
                    "home_team": f"H{i}",
                    "away_team": f"A{i}",
                    "prediction": "H",
                    "p_H": 0.5,
                    "p_D": 0.3,
                    "p_A": 0.2,
                    "score_home": np.nan,
                    "score_away": np.nan,
                    "FTHG": 1,
                    "FTAG": 0,
                    "FTR": result,
                }
            )
    payload = dashboards.evaluation_payload(pd.DataFrame(rows))

    assert [m["key"] for m in payload["models"]] == ["head_to_head", "market"]
    assert payload["models"][1]["reference"] is True
    assert payload["seasons"] == ["2024-25"]
    overall = payload["results"]["all|all"]["head_to_head"]
    assert overall["matches"] == 3 and overall["hit_rate"] == round(1 / 3, 5)
    assert payload["results"]["SP1|2024-25"]["market"]["matches"] == 2
    assert {m["key"] for m in payload["metrics"]} >= {"hit_rate", "log_loss", "brier", "rps"}
    assert payload["calibration"]["all"]["market"]


def test_evaluation_payload_without_backtest() -> None:
    payload = dashboards.evaluation_payload(pd.DataFrame())
    assert payload["models"] == [] and payload["results"] == {}


def test_current_payload_hides_played_fixtures(tmp_path: Path) -> None:
    from tests.test_loading import _write

    _write(tmp_path / "SP1" / "results.parquet", [("2026-11-08", "A", "B", 1, 0)])
    _write(tmp_path / "SP1" / "fixtures.parquet", [("2026-11-08", "A", "B", None, None)])

    league = dashboards.current_payload(tmp_path)["leagues"][0]

    assert league["fixtures"] == []  # already played: not "next matchday"
    assert league["odds_downloaded"] is not None
    assert len(league["latest"]) == 1
