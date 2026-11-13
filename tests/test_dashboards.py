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


def test_season_payload_scores_models_and_tickets() -> None:
    from tests.helpers import make_matches

    # Matchday 1: 14 La Liga matches, all home wins, played
    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", 1, 0) for i in range(14)]
    current = make_matches(rows).assign(round=1, matchday=1)
    predictions = current[["league", "game"]].assign(
        prediction=["H"] * 11 + ["A"] * 3, p_H=0.5, p_D=0.3, p_A=0.2
    )

    payload = dashboards.season_payload(current, {"head_to_head": predictions})

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


def test_season_payload_without_predictions() -> None:
    from tests.helpers import make_matches

    current = make_matches([("SP1", "2026-08-15", "A", "B", None, None)]).assign(
        round=1, matchday=1
    )
    payload = dashboards.season_payload(current, {})
    assert payload["models"] == [] and payload["totals"] == {}
    assert payload["matchdays"][0]["played"] == 0


def test_season_payload_counts_matches_without_prediction() -> None:
    from tests.helpers import make_matches

    # 15 played home wins; the model has no prediction for two of them
    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", 1, 0) for i in range(15)]
    current = make_matches(rows).assign(round=1, matchday=1)
    predictions = (
        current[["league", "game"]].iloc[2:].assign(prediction="H", p_H=0.5, p_D=0.3, p_A=0.2)
    )

    payload = dashboards.season_payload(current, {"head_to_head": predictions})

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


def _current_season(played: list[bool]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Two matchdays of 14 La Liga matches; ``played`` says which ones have results."""
    from tests.helpers import make_matches

    rows = []
    for matchday, done in enumerate(played, start=1):
        day = f"2026-08-{14 + 7 * matchday}"
        goals = (1, 0) if done else (None, None)
        rows += [("SP1", day, f"H{matchday}{i}", f"A{matchday}{i}", *goals) for i in range(14)]
    current = make_matches(rows)
    current["matchday"] = [1] * 14 + [2] * 14
    current["round"] = current["matchday"]
    predictions = current[["league", "game"]].assign(prediction="H", p_H=0.5, p_D=0.3, p_A=0.2)
    return current, {"head_to_head": predictions}


@pytest.mark.parametrize(
    ("played", "number", "status"),
    [
        ([True, False], 2, "upcoming"),  # next matchday published, nothing played yet
        ([True, True], 2, "completed"),  # e.g. international break: next one not published
    ],
)
def test_current_payload_picks_the_current_matchday(
    tmp_path: Path, played: list[bool], number: int, status: str
) -> None:
    current, predictions = _current_season(played)
    payload = dashboards.current_payload(current, predictions, current_dir=tmp_path)

    assert payload["matchday"]["number"] == number
    assert payload["status"] == status
    assert payload["matchdays"] == 2
    assert [m["key"] for m in payload["models"]] == ["head_to_head"]


def test_current_matchday_in_progress() -> None:
    matchdays = [
        {"played": 10, "matches": [{}] * 10},
        {"played": 3, "matches": [{}] * 10},
        {"played": 0, "matches": [{}] * 10},
    ]
    assert dashboards.current_matchday_index(matchdays) == 1
    assert dashboards.current_matchday_index([]) is None


def test_current_payload_standings_and_odds_time(tmp_path: Path) -> None:
    from tests.test_loading import _write

    _write(tmp_path / "SP1" / "results.parquet", [("2026-11-08", "A", "B", 1, 0)])
    _write(tmp_path / "SP1" / "fixtures.parquet", [("2026-11-15", "C", "D", None, None)])
    current, predictions = _current_season([True, False])

    payload = dashboards.current_payload(current, predictions, current_dir=tmp_path)

    assert [row["team"] for row in payload["standings"]["SP1"]] == ["A", "B"]
    assert payload["odds_downloaded"] is not None


def test_build_removes_retired_dashboards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("current", "season", "historical", "evaluation"):
        monkeypatch.setattr(
            dashboards, f"{name}_payload", lambda **_: {"leagues": [], "models": []}
        )
    old = tmp_path / "predictions.html"
    old.write_text("old")

    paths = dashboards.build_dashboards(tmp_path)

    assert not old.exists()
    assert sorted(p.name for p in paths) == [
        "current.html",
        "evaluation.html",
        "historical.html",
        "season.html",
    ]


def test_season_payload_confidence_odds_and_planned_ticket() -> None:
    from tests.helpers import make_matches

    # 14 pending La Liga matches predicted by v3 with different confidences
    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", None, None) for i in range(14)]
    current = make_matches(rows).assign(round=1, matchday=1)
    p_home = [0.75, 0.40] + [0.55] * 12
    predictions = current[["league", "game"]].assign(
        prediction="H", p_H=p_home, p_D=[0.15, 0.35] + [0.25] * 12, p_A=[0.10, 0.25] + [0.20] * 12
    )

    payload = dashboards.season_payload(current, {"market_blend": predictions})

    assert [m["key"] for m in payload["models"]] == ["market_blend", dashboards.PLANNED_KEY]
    assert payload["confidence_model"] == "market_blend"
    matches = {m["home_team"]: m for m in payload["matchdays"][0]["matches"]}
    assert matches["H0"]["market_blend_conf"] == 0.75
    assert matches["H0"]["market_blend_margin"] == pytest.approx(0.6)

    summary = payload["matchdays"][0]["summary"]
    odds = summary["market_blend"]["odds"]
    assert set(odds) == {"10", "11", "12", "13", "14"} and odds["10"] > odds["14"]
    plan = summary[dashboards.PLANNED_KEY]["plan"]
    assert plan["bets"] <= 48 and plan["odds"]["10"] > odds["10"]
    # The least predictable match gets more than one sign on the planned ticket
    assert len(matches["H1"]["planned_sign"]) > 1
    assert matches["H0"]["planned_sign"] == "H"


def test_planned_ticket_counts_doubles_as_hits() -> None:
    from tests.helpers import make_matches

    # Every match ends in a draw: only the doubles and triples of v4 hit
    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", 1, 1) for i in range(14)]
    current = make_matches(rows).assign(round=1, matchday=1)
    predictions = current[["league", "game"]].assign(prediction="H", p_H=0.45, p_D=0.33, p_A=0.22)

    payload = dashboards.season_payload(current, {"market_blend": predictions})

    summary = payload["matchdays"][0]["summary"]
    plan = summary[dashboards.PLANNED_KEY]["plan"]
    assert summary["market_blend"]["hits"] == 0
    assert summary[dashboards.PLANNED_KEY]["hits"] == plan["doubles"] + plan["triples"]


def test_evaluation_payload_confidence_and_tickets() -> None:
    rows = [
        {
            "model": "market_blend",
            "league": "SP1",
            "season": "2024-25",
            "game": f"g{i}",
            "matchday": 1,
            "date": pd.Timestamp("2024-08-17"),
            "home_team": f"H{i}",
            "away_team": f"A{i}",
            "prediction": "H",
            "p_H": p,
            "p_D": (1 - p) / 2,
            "p_A": (1 - p) / 2,
            "score_home": float("nan"),
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
        }
        for i, p in enumerate([0.7, 0.5, 0.4])
    ]
    tickets = pd.DataFrame(
        {
            "model": "market_blend",
            "season": "2024-25",
            "matchday": 1,
            "strategy": ["single", "v4"],
            "draw": 0,
            "doubles": [0, 4],
            "triples": [0, 1],
            "bets": [1, 48],
            "hits": [9, 11],
            "category": [None, "4th category"],
            "p_prize": [0.1, 0.4],
            "p_top": [0.001, 0.004],
            "expected_hits": [8.0, 9.5],
        }
    )
    payload = dashboards.evaluation_payload(pd.DataFrame(rows), tickets)

    tiers = payload["confidence"]["all|all"]["market_blend"]
    assert [t["tier"] for t in tiers] == ["high", "medium", "low"]
    assert [t["matches"] for t in tiers] == [1, 1, 1]
    strategies = {r["strategy"]: r for r in payload["tickets"]["all"]}
    assert strategies["v4"]["prizes"] == 1 and strategies["single"]["prizes"] == 0
    assert payload["tickets"]["2024-25"]
    assert payload["ticket_budget"]["max_bets"] == 48


def test_market_is_left_out_of_the_matchday_views() -> None:
    from tests.helpers import make_matches

    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", 1, 0) for i in range(14)]
    current = make_matches(rows).assign(round=1, matchday=1)
    predictions = current[["league", "game"]].assign(prediction="H", p_H=0.5, p_D=0.3, p_A=0.2)

    payload = dashboards.season_payload(
        current, {"head_to_head": predictions, "market": predictions}
    )

    # Only in the evaluation dashboard (test_evaluation_payload keeps it there)
    assert [m["key"] for m in payload["models"]] == ["head_to_head"]
    assert "market" not in payload["totals"]
    assert "market_sign" not in payload["matchdays"][0]["matches"][0]


def test_evaluation_payload_includes_the_planned_ticket() -> None:
    from quiniela.evaluation import backtest
    from tests.test_backtest import _planner_backtest

    predictions = _planner_backtest(2).assign(score_home=float("nan"))
    tickets = backtest.ticket_strategies(predictions, n_jobs=1)

    payload = dashboards.evaluation_payload(predictions, tickets)

    assert [m["key"] for m in payload["models"]] == ["market_blend", dashboards.PLANNED_KEY]
    overall = payload["results"]["all|all"]
    planned = overall[dashboards.PLANNED_KEY]
    # Only hit rate and prizes: v4 has the probabilities of v3
    assert set(planned) == {"matches", "hit_rate", "prizes", "tickets"}
    assert planned["matches"] == overall["market_blend"]["matches"] == 28
    # Doubles and triples can only add hits
    assert planned["hit_rate"] >= overall["market_blend"]["hit_rate"]
    v4 = tickets[tickets["strategy"] == "v4"]
    assert planned["prizes"] == int((v4["hits"] >= 10).sum())
    assert "prizes" not in payload["results"]["SP1|all"][dashboards.PLANNED_KEY]


def _prize_context(jackpot: float = 0.0):  # type: ignore[no-untyped-def]
    from quiniela.evaluation.money import PrizeContext
    from quiniela.prediction.money import PrizeModel

    model = PrizeModel(
        intercept={14: 0.66, 13: 5.84, 12: 6.8, 11: 6.92, 10: 6.34},
        slope={14: -1.05, 13: -1.19, 12: -1.07, 11: -0.93, 10: -0.77},
        sd={k: 0.5 for k in (14, 13, 12, 11, 10)},
        special_share=0.2,
        draws=100,
    )
    empty = pd.DataFrame(columns=["draw_id", "league", "game", "position"])
    return PrizeContext(model=model, bets=3_000_000, jackpot=jackpot, draws=empty, matched=empty)


def test_season_payload_money_ticket_and_euros() -> None:
    from tests.helpers import make_matches

    rows = [("SP1", "2026-08-15", f"H{i}", f"A{i}", 1, 0) for i in range(14)]
    current = make_matches(rows).assign(round=1, matchday=1)
    predictions = current[["league", "game"]].assign(prediction="H", p_H=0.5, p_D=0.3, p_A=0.2)

    without = dashboards.season_payload(current, {"market_blend": predictions})
    assert dashboards.MONEY_KEY not in [m["key"] for m in without["models"]]

    payload = dashboards.season_payload(
        current, {"market_blend": predictions}, prizes=_prize_context(jackpot=5e8)
    )
    keys = [m["key"] for m in payload["models"]]
    assert keys == ["market_blend", dashboards.PLANNED_KEY, dashboards.MONEY_KEY]
    summary = payload["matchdays"][0]["summary"]
    plan = summary[dashboards.MONEY_KEY]["money"]
    # A huge jackpot is all in 14 hits and the Pleno al 15: the prudent verdict ignores it
    assert not plan["worth"] and plan["expected_top"] > 0 and plan["bets"] <= 48
    assert plan["expected_common"] < plan["cost"]
    # Every favourite won: the played matchday is valued in euros (estimated, no real draw)
    euros = summary["market_blend"]["euros"]
    assert euros["played"] and euros["cost"] == 0.75 and euros["source"] == "model"
    assert euros["payout"] > 0
    assert payload["totals"][dashboards.PLANNED_KEY]["euros"]["played"] == 1
