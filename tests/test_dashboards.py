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
