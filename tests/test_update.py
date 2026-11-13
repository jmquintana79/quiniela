from datetime import date
from pathlib import Path
from typing import Any

import pytest

from quiniela import update

TODAY = date(2026, 11, 11)  # season 2026-27 in progress: last completed is 2025-26


def _touch_seasons(root: Path, code: str, years: range) -> None:
    for year in years:
        path = root / code / f"{year}-{(year + 1) % 100:02d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Replace every step with a recorder (no network, no data needed)."""
    recorded: list[tuple[str, Any]] = []

    def historical(start: int, end: int, leagues: list[str]) -> list[Path]:
        recorded.append(("historical", (leagues[0], start, end)))
        return [Path(f"{leagues[0]}-{y}") for y in range(start, end + 1)]

    def current() -> list[Path]:
        recorded.append(("current", None))
        return [Path("results.parquet")]

    def predict(model_key: str, all_matchdays: bool) -> list[Path]:
        recorded.append(("predict", (model_key, all_matchdays)))
        return [Path("model.parquet")]

    monkeypatch.setattr(update.download, "download_historical", historical)
    monkeypatch.setattr(update.download, "download_current", current)
    monkeypatch.setattr(update.pipeline, "run", predict)
    monkeypatch.setattr(
        update.backtest, "run_backtest", lambda: recorded.append(("backtest", None)) or "bt"
    )
    monkeypatch.setattr(update.backtest, "save_backtest", lambda bt: Path("backtest.parquet"))
    monkeypatch.setattr(update.money, "save_money", lambda: None)
    monkeypatch.setattr(
        update.dashboards,
        "build_dashboards",
        lambda: recorded.append(("dashboards", None)) or [Path("a.html")],
    )
    return recorded


def test_missing_seasons_per_league(tmp_path: Path) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))  # La Liga complete
    _touch_seasons(tmp_path, "SP2", range(1996, 2024))  # Segunda lacks 2024-25 and 2025-26

    assert update.missing_seasons(2025, tmp_path) == {"ESP-Segunda": [2024, 2025]}
    _touch_seasons(tmp_path, "SP2", range(2024, 2026))
    assert update.missing_seasons(2025, tmp_path) == {}


def test_runs() -> None:
    assert update._runs([1993, 1994, 1995, 2001, 2003, 2004]) == [
        (1993, 1995),
        (2001, 2001),
        (2003, 2004),
    ]
    assert update._runs([]) == []


def test_full_update_order(tmp_path: Path, calls: list[tuple[str, Any]]) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))
    _touch_seasons(tmp_path, "SP2", range(1996, 2025))  # only 2025-26 missing

    report = update.run_update(today=TODAY, historical_dir=tmp_path)

    assert calls == [
        ("historical", ("ESP-Segunda", 2025, 2025)),
        ("current", None),
        ("predict", ("all", True)),
        ("dashboards", None),
    ]
    assert report.backtest is None and not report.warnings
    assert "Backtest: not run" in report.summary()


def test_nothing_downloaded_twice(tmp_path: Path, calls: list[tuple[str, Any]]) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))
    _touch_seasons(tmp_path, "SP2", range(1996, 2026))

    report = update.run_update(today=TODAY, historical_dir=tmp_path)

    assert [name for name, _ in calls] == ["current", "predict", "dashboards"]
    assert report.historical == []


def test_first_install_downloads_whole_history(
    tmp_path: Path, calls: list[tuple[str, Any]]
) -> None:
    update.run_update(today=TODAY, historical_dir=tmp_path)
    historical = [args for name, args in calls if name == "historical"]
    assert historical == [("ESP-La Liga", 1993, 2025), ("ESP-Segunda", 1996, 2025)]


def test_new_season_adds_the_season_that_just_ended(
    tmp_path: Path, calls: list[tuple[str, Any]]
) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))
    _touch_seasons(tmp_path, "SP2", range(1996, 2026))

    update.run_update(today=date(2027, 9, 1), historical_dir=tmp_path)  # 2027-28 started

    historical = [args for name, args in calls if name == "historical"]
    assert historical == [("ESP-La Liga", 2026, 2026), ("ESP-Segunda", 2026, 2026)]


def test_backtest_runs_before_dashboards(tmp_path: Path, calls: list[tuple[str, Any]]) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))
    _touch_seasons(tmp_path, "SP2", range(1996, 2026))

    report = update.run_update(run_backtest=True, today=TODAY, historical_dir=tmp_path)

    assert [name for name, _ in calls][-2:] == ["backtest", "dashboards"]
    assert report.backtest == Path("backtest.parquet")


def _failing_download() -> list[Path]:
    raise ConnectionError("Could not download")


def test_download_error_stops_the_update(
    tmp_path: Path, calls: list[tuple[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))
    _touch_seasons(tmp_path, "SP2", range(1996, 2026))
    monkeypatch.setattr(update.download, "download_current", _failing_download)

    with pytest.raises(ConnectionError):
        update.run_update(today=TODAY, historical_dir=tmp_path)
    assert calls == []  # neither predictions nor dashboards were touched


def test_off_season_skips_predictions(
    tmp_path: Path, calls: list[tuple[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))
    _touch_seasons(tmp_path, "SP2", range(1996, 2026))
    monkeypatch.setattr(update.download, "download_current", _failing_download)

    report = update.run_update(today=date(2026, 7, 20), historical_dir=tmp_path)

    assert [name for name, _ in calls] == ["dashboards"]
    assert "off-season" in report.warnings[0]


def test_no_matches_to_predict(
    tmp_path: Path, calls: list[tuple[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch_seasons(tmp_path, "SP1", range(1993, 2026))
    _touch_seasons(tmp_path, "SP2", range(1996, 2026))

    def no_data(model_key: str, all_matchdays: bool) -> list[Path]:
        raise ValueError("No current season data")

    monkeypatch.setattr(update.pipeline, "run", no_data)

    report = update.run_update(today=TODAY, historical_dir=tmp_path)

    assert report.warnings == ["Predictions skipped: No current season data"]
    assert calls[-1] == ("dashboards", None)


def test_cli_parses_update() -> None:
    from quiniela.scripts import _build_parser

    assert _build_parser().parse_args(["update", "--backtest"]).backtest is True
    assert _build_parser().parse_args(["update"]).backtest is False
