import json
from pathlib import Path

import pandas as pd
import pytest

from quiniela.io import lae
from tests.test_prizes import _draw, _match, _upcoming_file

BEFORE = pd.Timestamp("2026-11-13 20:00", tz="Europe/Madrid")  # ticket 21 closes 14/11 14:00
AFTER_DRAW = pd.Timestamp("2026-11-17 10:00", tz="Europe/Madrid")


def _season_file(folder: Path, jornadas: list[int]) -> None:
    draws = []
    for jornada in jornadas:
        draw = _draw(
            str(900 + jornada),
            "2026-11-08",
            [_match(1, "A", "B", "1", "1 - 0", "2026-11-08 12:00:00")],
        )
        draw["temporada"], draw["jornada"] = "2026-2027", jornada
        draws.append(draw)
    (folder / "lae_2026-27.json").write_text(json.dumps(draws))


def test_nothing_missing_before_the_deadline(tmp_path: Path) -> None:
    _upcoming_file(tmp_path)
    _season_file(tmp_path, [20])
    assert not lae.missing_lae(tmp_path, BEFORE).any


def test_missing_ticket_and_draw(tmp_path: Path) -> None:
    needs = lae.missing_lae(tmp_path, BEFORE)
    # Nothing saved: the next ticket and the season's draws
    assert needs.upcoming and needs.results == [2026]
    _upcoming_file(tmp_path)
    _season_file(tmp_path, [20])
    needs = lae.missing_lae(tmp_path, AFTER_DRAW)
    # The saved ticket closed and its draw (jornada 21) is not saved yet
    assert needs.upcoming and needs.results == [2026]
    # Right after the deadline the matches are not over: only the next ticket
    just_closed = pd.Timestamp("2026-11-14 20:00", tz="Europe/Madrid")
    assert lae.missing_lae(tmp_path, just_closed).results == []


def test_collect_downloads(tmp_path: Path) -> None:
    downloads, lae_dir = tmp_path / "Downloads", tmp_path / "lae"
    downloads.mkdir()
    _upcoming_file(downloads)
    (downloads / "upcoming.json").rename(downloads / "quiniela_upcoming_j21.json")
    _season_file(downloads, [20, 21])
    (downloads / "lae_2026-27.json").rename(downloads / "quiniela_lae_2026-27.json")
    (downloads / "quiniela_upcoming_j99.json").write_text("not json")
    moved = lae.collect_downloads(downloads, lae_dir)
    assert sorted(p.name for p in moved) == ["lae_2026-27.json", "upcoming.json"]
    assert json.loads((lae_dir / "upcoming.json").read_text())["jornada"] == 21
    # Valid files are moved; anything else is left in Downloads
    assert [p.name for p in downloads.iterdir()] == ["quiniela_upcoming_j99.json"]
    assert lae.collect_downloads(tmp_path / "missing", lae_dir) == []


def test_fetch_without_terminal_only_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(lae.webbrowser, "open", opened.append)
    saved, warnings = lae.fetch_lae(
        interactive=False, lae_dir=tmp_path, downloads_dir=tmp_path / "d"
    )
    assert saved == [] and opened == []
    assert len(warnings) == 1 and "invoke update" in warnings[0]


def test_bookmarklet_page(tmp_path: Path) -> None:
    page = lae.write_bookmarklet_page(tmp_path)
    text = page.read_text()
    assert 'href="javascript:' in text and "buscadorSorteos" in text
    assert "quiniela_upcoming_j" in text
