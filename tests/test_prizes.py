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


def _upcoming_file(folder: Path, closes: str = "14/11/2026 - 14:00") -> Path:
    matches = [
        {
            "position": 1,
            "home": "R. Valladolid (M)",
            "away": "Castellón (M)",
            "public": {"1": 30, "X": 25, "2": 45},
        },
        {
            "position": 2,
            "home": "Alavés (F)",
            "away": "Eibar (F)",
            "public": {"1": 35, "X": 34, "2": 31},
        },
        {
            "position": 15,
            "home": "España (M)",
            "away": "Inglaterra (M)",
            "public_home": {"0": 4, "1": 19, "2": 51, "M": 26},
            "public_away": {"0": 14, "1": 53, "2": 25, "M": 8},
        },
    ]
    path = folder / prizes.UPCOMING_FILE
    path.write_text(
        json.dumps(
            {
                "jornada": 21,
                "closes": closes,
                "saved_at": "2026-11-13T15:00:00Z",
                "matches": matches,
            }
        )
    )
    return path


def test_load_upcoming(tmp_path: Path) -> None:
    _upcoming_file(tmp_path)
    ticket = prizes.load_upcoming(tmp_path)
    assert ticket is not None and ticket.jornada == 21
    # Madrid time (CET in November: UTC+1)
    assert ticket.closes.isoformat() == "2026-11-14T14:00:00+01:00"
    first = ticket.matches.iloc[0]
    assert (first["lae_home"], first["public_2"]) == ("R. Valladolid (M)", 45)
    assert ticket.pleno_public["home"]["2"] == 51
    assert prizes.load_upcoming(tmp_path / "missing") is None


def test_load_upcoming_ignores_a_broken_file(tmp_path: Path) -> None:
    (tmp_path / prizes.UPCOMING_FILE).write_text('{"jornada": 21}')
    assert prizes.load_upcoming(tmp_path) is None


def test_translate_team() -> None:
    known = {prizes.normalise(t): t for t in ["Valladolid", "Ath Madrid", "Castellon"]}
    assert prizes.translate_team(known, "R. Valladolid (M)") == "Valladolid"
    assert prizes.translate_team(known, "At. Madrid") == "Ath Madrid"
    assert prizes.translate_team(known, "Castellón (M)") == "Castellon"
    assert prizes.translate_team(known, "At. Madrid (F)") is None


@pytest.mark.parametrize(
    ("home", "away", "league", "expected"),
    [
        (None, None, "SP1", "M"),
        ("At. Madrid (m)", "Barcelona (m)", "SP1", "M"),
        ("Athletic Club (F)", "Real Madrid (F)", None, "F"),
        ("Bélgica (M)", "Italia (M)", None, "O"),
        ("Arsenal", "Chelsea", None, "O"),
        (None, None, None, "U"),
    ],
)
def test_match_category(
    home: str | None, away: str | None, league: str | None, expected: str
) -> None:
    assert prizes.match_category(home, away, league) == expected
    assert set(prizes.CATEGORY_LEGEND) == {"M", "F", "O", "U"}


def test_ticket_flags_and_official_tickets(lae_dir: Path, matches: pd.DataFrame) -> None:
    matched = prizes.match_tickets(prizes.load_tickets(lae_dir), matches)
    flags = prizes.ticket_flags(matched)
    # The two football-data matches of the ticket, with its jornada
    assert len(flags) == 3 and flags["quiniela"].eq(7).all()
    assert set(flags["game"]) >= {"2025-10-04 Ath Madrid-Barcelona"}
    tickets = prizes.official_tickets(prizes.load_draws(lae_dir), matched)
    ticket = tickets["2025-26"][0]
    assert ticket["jornada"] == 7 and ticket["date"] == "2025-10-05"
    categories = {m["position"]: m["category"] for m in ticket["matches"]}
    assert categories == {1: "M", 2: "M", 3: "O", 4: "F", 15: "M"}
    first = ticket["matches"][0]
    assert (first["home"], first["sign"], first["score"]) == ("Ath Madrid", "D", "1-1")
