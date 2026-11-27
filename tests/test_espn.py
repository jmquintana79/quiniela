from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quiniela.io import espn


def _event(
    event_id: str, when: str, home: str, away: str, score: tuple[int, int] | None
) -> dict[str, Any]:
    status = "STATUS_FULL_TIME" if score else "STATUS_SCHEDULED"
    side = lambda where, name, goals: {  # noqa: E731
        "homeAway": where,
        "score": str(goals) if goals is not None else "0",
        "team": {"displayName": name},
    }
    return {
        "id": event_id,
        "date": when,
        "status": {"type": {"name": status}},
        "competitions": [
            {
                "competitors": [
                    side("home", home, score[0] if score else None),
                    side("away", away, score[1] if score else None),
                ]
            }
        ],
    }


def test_parse_events() -> None:
    data = {
        "events": [
            _event("1", "2026-11-08T11:00Z", "Barcelona", "Madrid CFF", (5, 1)),
            _event("2", "2026-11-22T11:00Z", "Sevilla", "Eibar", None),
        ]
    }
    matches = espn.parse_events(data, "2026-27")
    assert matches["league"].eq("LIGAF").all() and list(matches.columns) == espn.COLUMNS
    played, pending = matches.iloc[0], matches.iloc[1]
    assert (played["FTHG"], played["FTAG"], played["FTR"]) == (5, 1, "H")
    # Madrid time (CET in November)
    assert played["date"] == pd.Timestamp("2026-11-08 12:00")
    assert pd.isna(pending["FTR"]) and pending["status"] == "STATUS_SCHEDULED"
    assert espn.parse_events({}, "2026-27").empty


def test_parse_events_reads_the_shots() -> None:
    event = _event("1", "2026-11-08T11:00Z", "Barcelona", "Madrid CFF", (5, 1))
    stats = {
        "home": [
            {"name": "totalShots", "displayValue": "21"},
            {"name": "shotsOnTarget", "displayValue": "9"},
        ],
        "away": [
            {"name": "totalShots", "displayValue": "6"},
            {"name": "shotsOnTarget", "displayValue": "x"},
        ],
    }
    for competitor in event["competitions"][0]["competitors"]:
        competitor["statistics"] = stats[competitor["homeAway"]]
    pending = _event("2", "2026-11-22T11:00Z", "Sevilla", "Eibar", None)
    matches = espn.parse_events({"events": [event, pending]}, "2026-27")
    played = matches.iloc[0]
    assert (played["HS"], played["AS"], played["HST"]) == (21, 6, 9)
    # A statistic that is not a number is missing; matches not played have none
    assert pd.isna(played["AST"])
    assert matches.iloc[1][["HS", "AS", "HST", "AST"]].isna().all()


def test_update_downloads_missing_seasons_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        month = url.split("dates=")[1]
        if month.endswith("10"):
            return {"events": [_event(month, f"{month[:4]}-10-04T10:00Z", "A", "B", (1, 0))]}
        return {"events": []}

    monkeypatch.setattr(espn, "_fetch", fetch)
    monkeypatch.setattr(espn, "PAUSE_SECONDS", 0)
    out, cache = tmp_path / "espn", tmp_path / "cache"
    written = espn.update_liga_f(date(2024, 11, 15), output_dir=out, cache_dir=cache)
    # 2022-23 and 2023-24 completed, 2024-25 in progress
    assert [p.name for p in written] == ["2022-23.parquet", "2023-24.parquet", "2024-25.parquet"]
    assert len(calls) == 36
    matches = espn.load_liga_f_matches(out)
    assert len(matches) == 3 and matches["season"].tolist() == ["2022-23", "2023-24", "2024-25"]
    # A second update only asks again for the months of the season in progress from last month on
    calls.clear()
    espn.update_liga_f(date(2024, 11, 15), output_dir=out, cache_dir=cache)
    assert calls and all(c.split("dates=")[1] >= "202410" for c in calls)
