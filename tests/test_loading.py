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
