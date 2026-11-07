import io
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quiniela.io import download

FIXTURES_CSV = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,Referee,B365H,B365D,B365A\n"
    "SP1,08/11/2026,21:00,Real Madrid,Barcelona,,2.1,3.6,3.2\n"
    "SP2,09/11/2026,,Burgos,Cordoba,,1.9,3.2,4.1\n"
    "E0,08/11/2026,15:00,Arsenal,Chelsea,,1.8,3.8,4.2\n"
)


def _fake_games(league: str, seasons: list[int]) -> pd.DataFrame:
    """Two matches per season with soccerdata's index layout."""
    rows = [
        (league, download._season_code(year), f"{year}-09-0{i} A-B", i, 0)
        for year in seasons
        for i in (1, 2)
    ]
    df = pd.DataFrame(rows, columns=["league", "season", "game", "FTHG", "FTAG"])
    return df.set_index(["league", "season", "game"])


@pytest.fixture
def data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(download, "HISTORICAL_DIR", tmp_path / "historical")
    monkeypatch.setattr(download, "CURRENT_DIR", tmp_path / "current")
    monkeypatch.setattr(download, "ensure_data_dirs", lambda: None)
    monkeypatch.setattr(download, "current_season", lambda today=None: 2026)
    monkeypatch.setattr(download, "_read_games", _fake_games)
    return tmp_path


@pytest.mark.parametrize(
    ("today", "expected"),
    [(date(2026, 11, 7), 2026), (date(2027, 3, 1), 2026), (date(2027, 7, 1), 2027)],
)
def test_current_season(today: date, expected: int) -> None:
    assert download.current_season(today) == expected


def test_season_formats() -> None:
    assert download.season_label(2015) == "2015-16"
    assert download.season_label(1999) == "1999-00"
    assert download._season_code(1999) == "9900"


def test_segunda_is_registered() -> None:
    assert "ESP-Segunda" in download.sd.MatchHistory.available_leagues()


def test_unknown_league() -> None:
    with pytest.raises(ValueError, match="Unsupported leagues"):
        download._resolve_leagues(["ENG-Premier League"])


@pytest.mark.parametrize(("start", "end"), [(2020, 2019), (2024, 2026)])
def test_historical_invalid_range(data_dirs: Path, start: int, end: int) -> None:
    with pytest.raises(ValueError):
        download.download_historical(start, end)


def test_historical_writes_one_file_per_season(data_dirs: Path) -> None:
    paths = download.download_historical(1995, 1997)

    # Segunda has no data before 1996-97
    names = sorted(p.relative_to(data_dirs / "historical").as_posix() for p in paths)
    assert names == [
        "SP1/1995-96.parquet",
        "SP1/1996-97.parquet",
        "SP1/1997-98.parquet",
        "SP2/1996-97.parquet",
        "SP2/1997-98.parquet",
    ]
    assert len(pd.read_parquet(paths[0])) == 2


def test_current_writes_results_and_fixtures(
    data_dirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        content = FIXTURES_CSV.encode("utf-8")

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(download.requests, "get", lambda *args, **kwargs: FakeResponse())

    paths = download.download_current()

    names = sorted(p.relative_to(data_dirs / "current").as_posix() for p in paths)
    assert names == [
        "SP1/fixtures.parquet",
        "SP1/results.parquet",
        "SP2/fixtures.parquet",
        "SP2/results.parquet",
    ]
    fixtures = pd.read_parquet(data_dirs / "current" / "SP1" / "fixtures.parquet")
    assert fixtures.index.tolist() == [("ESP-La Liga", "2627", "2026-11-08 Real Madrid-Barcelona")]
    assert fixtures["date"].iloc[0] == pd.Timestamp("2026-11-08 21:00")
    # Missing kick-off time defaults to noon, as soccerdata does for results
    segunda = pd.read_parquet(data_dirs / "current" / "SP2" / "fixtures.parquet")
    assert segunda["date"].iloc[0] == pd.Timestamp("2026-11-09 12:00")


def test_parse_csv_keeps_rows_longer_than_header() -> None:
    # Header with a trailing blank cell; the second match has extra trailing fields
    raw = (
        "﻿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,\n"
        "SP1,07/11/04,Malaga,Real Madrid,0,2,A,\n"
        "SP1,07/11/04,Santander,Numancia,2,0,H,,1.9,,\n"
    ).encode("utf-8")

    df = download._parse_csv(io.BytesIO(raw), "SP1", "0405")

    assert df["HomeTeam"].tolist() == ["Malaga", "Santander"]
    assert df["FTR"].tolist() == ["A", "H"]
    assert [c for c in df.columns if not c.startswith("Unnamed")] == [
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]


def test_parse_csv_falls_back_to_latin1() -> None:
    raw = "Div,Date,HomeTeam,AwayTeam\nSP1,01/09/96,Logroñés,Betis\n".encode("latin-1")
    df = download._parse_csv(io.BytesIO(raw), "SP1", "9697")
    assert df["HomeTeam"].iloc[0] == "Logroñés"


def test_soccerdata_uses_tolerant_parser() -> None:
    assert download.sd_match_history._parse_csv is download._parse_csv
