from pathlib import Path

import pandas as pd
import pytest

from quiniela.data import loading


@pytest.mark.parametrize(
    ("code", "label"), [("1516", "2015-16"), ("9900", "1999-00"), ("9394", "1993-94")]
)
def test_season_label(code: str, label: str) -> None:
    assert loading.season_label(code) == label


def test_loaders_keep_season_and_available_odds(tmp_path: Path) -> None:
    index = pd.MultiIndex.from_tuples(
        [("ESP-La Liga", "2526", "2025-08-15 A-B")], names=["league", "season", "game"]
    )
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-08-15 21:00")],
            "home_team": ["A"],
            "away_team": ["B"],
            "FTHG": [1],
            "FTAG": [0],
            "FTR": ["H"],
            "B365H": [1.9],
            "B365D": [3.4],
            "B365A": [4.0],
            "Referee": ["X"],
        },
        index=index,
    )
    (tmp_path / "SP1").mkdir()
    df.to_parquet(tmp_path / "SP1" / "2025-26.parquet")

    matches = loading.load_historical(tmp_path)

    assert matches.loc[0, "season"] == "2025-26"
    assert matches.loc[0, "league"] == "SP1"
    assert matches.loc[0, "B365H"] == 1.9
    assert pd.isna(matches.loc[0, "AvgH"])  # every odds column exists, even if empty
    assert "Referee" not in matches.columns
    assert loading.load_current(tmp_path / "missing").empty


def _write(path: Path, rows: list[tuple[str, str, str, float | None, float | None]]) -> None:
    """Write a results/fixtures file in the download format."""
    index = pd.MultiIndex.from_tuples(
        [("ESP-La Liga", "2627", f"{d} {h}-{a}") for d, h, a, _, _ in rows],
        names=["league", "season", "game"],
    )
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d, *_ in rows],
            "home_team": [h for _, h, *_ in rows],
            "away_team": [a for _, _, a, *_ in rows],
            "FTHG": [g for *_, g, _ in rows],
            "FTAG": [g for *_, g in rows],
            "FTR": [
                None if h is None else "H" if h > a else "A" if h < a else "D" for *_, h, a in rows
            ],
            "AvgH": 2.0,
            "AvgD": 3.4,
            "AvgA": 3.8,
        },
        index=index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_current_drops_fixtures_already_played(tmp_path: Path) -> None:
    _write(tmp_path / "SP1" / "results.parquet", [("2026-11-08", "A", "B", 1, 0)])
    # Stale fixtures file: A-B already played (rescheduled: another date), C-D pending
    _write(
        tmp_path / "SP1" / "fixtures.parquet",
        [("2026-11-07", "A", "B", None, None), ("2026-11-21", "C", "D", None, None)],
    )

    matches = loading.load_current(tmp_path)

    assert matches[["home_team", "away_team"]].values.tolist() == [["A", "B"], ["C", "D"]]
    assert matches.loc[0, "FTHG"] == 1  # the played version is kept


def test_pending_fixtures() -> None:
    fixtures = pd.DataFrame({"home_team": ["A", "C"], "away_team": ["B", "D"]})
    results = pd.DataFrame({"home_team": ["A", "B"], "away_team": ["B", "A"]})
    assert loading.pending_fixtures(fixtures, results)["home_team"].tolist() == ["C"]
