import json
from pathlib import Path

import pandas as pd
import pytest

from quiniela.data import prizes


def _match(position: int, home: str, away: str, sign: str, score: str, when: str) -> dict:
    return {
        "posicion": position,
        "local": home,
        "visitante": away,
        "signo": f"{sign} ",
        "marcador": score,
        "fecha_completa": when,
    }


def _draw(draw_id: str, date: str, matches: list[dict]) -> dict:
    return {
        "id_sorteo": draw_id,
        "temporada": "2025-2026",
        "jornada": 7,
        "fecha_sorteo": f"{date} 22:00:00",
        "apuestas": "4000000",
        "recaudacion": "300000000",  # cents: 3,000,000 EUR
        "fondo_bote": "50000000",  # cents: 500,000 EUR
        "escrutinio": [
            {"categoria": 1, "ganadores": "0", "premio": "0"},
            {"categoria": 2, "ganadores": "3", "premio": "160000,00"},
            {"categoria": 3, "ganadores": "100", "premio": "2250,50"},
            {"categoria": 4, "ganadores": "1000", "premio": "225"},
            {"categoria": 5, "ganadores": "10000", "premio": "22,5"},
            {"categoria": 6, "ganadores": "300000", "premio": "0,9"},
        ],
        "partidos": matches,
    }


@pytest.fixture
def lae_dir(tmp_path: Path) -> Path:
    matches = [
        _match(1, "At. Madrid (m)", "Barcelona (m)", "X", "1 - 1", "2025-10-04 21:00:00"),
        _match(2, "R. Sociedad", "Racing", "2", "0 - 2", "2025-10-05 16:00:00"),
        # Same pairing as a league match, but a cup tie weeks away: not matched
        _match(3, "Betis", "Getafe", "1", "2 - 0", "2025-12-17 21:00:00"),
        _match(4, "Athletic Club (f)", "Real Madrid (f)", "1", "1 - 0", "2025-10-05 12:00:00"),
        _match(15, "Sevilla", "Valencia", "M-1", "3 - 1", "2025-10-05 21:00:00"),
    ]
    draw = _draw("111", "2025-10-05", matches)
    (tmp_path / "lae_2025-26.json").write_text(json.dumps([draw]))
    # The same draw saved twice is read once
    (tmp_path / "copy.json").write_text(json.dumps([draw]))
    return tmp_path


@pytest.fixture
def matches() -> pd.DataFrame:
    rows = [
        ("SP1", "2025-10-04", "Ath Madrid", "Barcelona"),
        ("SP2", "2025-10-05", "Sociedad", "Santander"),
        ("SP1", "2025-10-26", "Betis", "Getafe"),
        ("SP1", "2025-10-05", "Sevilla", "Valencia"),
    ]
    df = pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team"])
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = "2025-26"
    df["game"] = df["date"].dt.strftime("%Y-%m-%d") + " " + df["home_team"] + "-" + df["away_team"]
    return df


def test_load_draws(lae_dir: Path) -> None:
    draws = prizes.load_draws(lae_dir)
    assert len(draws) == 1
    draw = draws.iloc[0]
    assert draw["season"] == "2025-26" and draw["jornada"] == 7
    assert draw["takings"] == 3_000_000 and draw["jackpot"] == 500_000
    assert draw["winners_14"] == 3 and draw["prize_13"] == pytest.approx(2250.5)
    assert draw["prize_11"] == pytest.approx(22.5) and draw["winners_special"] == 0


def test_load_tickets(lae_dir: Path) -> None:
    tickets = prizes.load_tickets(lae_dir).sort_values("position")
    assert tickets["position"].tolist() == [1, 2, 3, 4, 15]
    assert tickets["sign"].tolist()[:4] == ["D", "A", "H", "H"]
    assert tickets["sign"].isna().tolist()[-1]  # the Pleno al 15 has no 1X2 sign
    assert tickets.iloc[-1]["FTHG"] == 3 and tickets.iloc[-1]["FTAG"] == 1


def test_match_tickets(lae_dir: Path, matches: pd.DataFrame) -> None:
    matched = prizes.match_tickets(prizes.load_tickets(lae_dir), matches).set_index("position")
    # Aliases and the men's team mark
    assert matched.loc[1, "game"] == "2025-10-04 Ath Madrid-Barcelona"
    assert matched.loc[2, "league"] == "SP2"
    # A cup tie between league teams is far from the league date
    assert pd.isna(matched.loc[3, "game"])
    # Women's teams never match
    assert pd.isna(matched.loc[4, "home_team"])
    coverage = prizes.draw_coverage(matched.reset_index())
    assert coverage["matched"].tolist() == [2]


def test_normalise() -> None:
    assert prizes.normalise("Málaga (m)") == "malaga"
    assert prizes.normalise("At.  Madrid") == "at madrid"
    assert prizes.normalise("Eibar (f)") == "eibar (f)"


def test_empty_folder(tmp_path: Path) -> None:
    assert prizes.load_draws(tmp_path).empty
    assert prizes.load_tickets(tmp_path).empty
