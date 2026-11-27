from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quiniela.data import clubs
from quiniela.evaluation import clubs as evaluation
from quiniela.io import espn_clubs
from quiniela.prediction.elo_clubs import ClubEloModel, club_ratings


def _event(
    event_id: str,
    when: str,
    home: tuple[str, str],
    away: tuple[str, str],
    score: tuple[int, int] | None,
) -> dict:
    status = "STATUS_FULL_TIME" if score else "STATUS_SCHEDULED"
    return {
        "id": event_id,
        "date": when,
        "status": {"type": {"name": status}},
        "competitions": [
            {
                "neutralSite": False,
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": str(score[0]) if score else "0",
                        "team": {"id": home[0], "displayName": home[1]},
                    },
                    {
                        "homeAway": "away",
                        "score": str(score[1]) if score else "0",
                        "team": {"id": away[0], "displayName": away[1]},
                    },
                ],
            }
        ],
    }


def _espn(years: range = range(2009, 2014), seed: int = 0) -> pd.DataFrame:
    """Two leagues of four clubs and a cup between them every year (ESPN-like matches)."""
    rng = np.random.default_rng(seed)
    strength = {
        "1": ("Real Madrid", 0.9, "esp.1"),
        "2": ("Atlético Madrid", 0.6, "esp.1"),
        "3": ("Getafe", 0.0, "esp.1"),
        "4": ("Eibar", -0.3, "esp.1"),
        "5": ("Manchester City", 0.8, "eng.1"),
        "6": ("Sheffield United", -0.2, "eng.1"),
        "7": ("Burnley", -0.4, "eng.1"),
        "8": ("Luton Town", -0.6, "eng.1"),
    }
    rows = []
    for year in years:
        for month in (2, 9, 11):
            for home in strength:
                for away in strength:
                    same = strength[home][2] == strength[away][2]
                    if home == away or (month == 11) == same:
                        continue
                    gap = strength[home][1] - strength[away][1]
                    rows.append(
                        {
                            "competition": strength[home][2] if same else "uefa.champions",
                            "espn_id": str(len(rows)),
                            "date": pd.Timestamp(year, month, 1 + len(rows) % 27, 21),
                            "home_id": home,
                            "away_id": away,
                            "home_team": strength[home][0],
                            "away_team": strength[away][0],
                            "FTHG": float(rng.poisson(np.exp(0.3 + 0.5 * gap))),
                            "FTAG": float(rng.poisson(np.exp(-0.5 * gap))),
                            "status": "STATUS_FULL_TIME",
                            "neutral": False,
                        }
                    )
    out = pd.DataFrame(rows)
    out["FTR"] = np.select([out["FTHG"] > out["FTAG"], out["FTHG"] == out["FTAG"]], ["H", "D"], "A")
    return out[espn_clubs.COLUMNS]


# ESPN download


def test_parse_events() -> None:
    data = {
        "events": [
            _event("10", "2024-09-18T19:00Z", ("83", "Barcelona"), ("94", "AS Monaco"), (1, 2)),
            _event("11", "2024-10-01T19:00Z", ("83", "Barcelona"), ("86", "Real Madrid"), None),
            {"id": "12", "status": {"type": {"name": "STATUS_FULL_TIME"}}, "competitions": [{}]},
        ]
    }
    matches = espn_clubs.parse_events(data, "uefa.champions")
    assert list(matches.columns) == espn_clubs.COLUMNS
    assert matches.loc[0, "FTR"] == "A" and pd.isna(matches.loc[1, "FTR"])
    assert matches.loc[0, "date"] == pd.Timestamp("2024-09-18 21:00")
    assert matches.loc[0, "home_id"] == "83" and pd.isna(matches.loc[1, "FTHG"])
    assert espn_clubs.parse_events({}, "esp.1").empty


def test_download_competition_keeps_old_months(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real loader, not the empty one of conftest.py
    monkeypatch.undo()
    asked: list[str] = []

    def fetch(url: str) -> dict:
        asked.append(url.split("dates=")[1][:6])
        return {"events": [_event(url[-20:], "2026-10-04T19:00Z", ("1", "A"), ("2", "B"), (1, 0))]}

    monkeypatch.setattr(espn_clubs, "_fetch", fetch)
    monkeypatch.setattr(espn_clubs, "PAUSE_SECONDS", 0)
    today = date(2026, 11, 27)
    assert espn_clubs.download_competition("esp.1", today, tmp_path, first_year=2026) == 12
    asked.clear()
    # Only the last month, the current one and the fixtures are asked again
    assert espn_clubs.download_competition("esp.1", today, tmp_path, first_year=2026) == 3
    assert asked == ["202610", "202611", "202612"]
    assert len(espn_clubs.load_club_matches(tmp_path)) == 12
    assert espn_clubs.load_club_matches(tmp_path / "none").empty


# Names


def test_similarity_of_names() -> None:
    assert clubs.similarity("Inter De Milán", "Internazionale") == 1.0
    assert clubs.similarity("Manch. United", "Manchester United") == 1.0
    assert clubs.similarity("Sheffield U.", "Sheffield United") == 1.0
    assert clubs.similarity("Club Brujas", "Club Brugge") == 1.0
    assert clubs.similarity("At. Madrid (m)", "Atlético Madrid") == 1.0
    assert clubs.similarity("Hammarby", "Hammarby IF") == 1.0
    assert clubs.similarity("Bayer Leverkusen", "B. Leverkusen") > 0.6
    assert clubs.similarity("Real Madrid", "Real Betis") < 0.6
    assert clubs.is_club_match("Chelsea", "Benfica")
    assert not clubs.is_club_match("España", "Malta")
    assert not clubs.is_club_match("Chelsea (F)", "PSG (F)")
    assert not clubs.is_club_match("Italia S21", "España S21")


def test_ticket_clubs() -> None:
    espn = _espn(range(2012, 2013))
    match = espn[(espn["home_team"] == "Manchester City") & (espn["away_team"] == "Getafe")].iloc[0]
    other = espn[(espn["home_team"] == "Eibar") & (espn["away_team"] == "Burnley")].iloc[0]
    matched = pd.DataFrame(
        {
            "season": ["2012-13"] * 4,
            "jornada": [30] * 4,
            "position": [1, 2, 3, 4],
            "draw_date": [match["date"].normalize() + pd.Timedelta(days=4)] * 4,
            "kickoff": [match["date"], pd.NaT, pd.NaT, pd.NaT],
            "lae_home": ["Man.city", "Burnley", "Real Madrid", "España"],
            "lae_away": ["Getafe", "Eibar", "Barcelona", "Malta"],
            "sign": ["H", None, None, None],
            "league": [None, None, "SP1", None],
        }
    )
    # The second match was played at Eibar's ground and with no kick-off on the ticket
    matched.loc[1, "draw_date"] = other["date"].normalize() + pd.Timedelta(days=3)
    linked = clubs.ticket_clubs(matched, espn)
    assert linked["position"].tolist() == [1, 2]
    assert linked.loc[0, "espn_id"] == match["espn_id"] and not linked.loc[0, "swapped"]
    assert linked.loc[0, "FTR"] == "H" and linked.loc[0, "competition"] == "uefa.champions"
    second = linked.iloc[1]
    assert second["swapped"] and second["home_id"] == other["away_id"]
    assert second["FTR"] == {"H": "A", "A": "H", "D": "D"}[other["FTR"]]
    assert clubs.ticket_clubs(matched, espn.iloc[:0]).empty


# Club Elo (v1C)


def test_new_teams_start_low_in_their_competition() -> None:
    espn = _espn(range(2009, 2012))
    late = pd.DataFrame(
        [
            {"competition": "esp.1", "espn_id": "x", "date": pd.Timestamp("2012-01-05"),
             "home_id": "99", "away_id": "1", "home_team": "Leganés", "away_team": "Real Madrid",
             "FTHG": 0.0, "FTAG": 0.0, "FTR": "D", "status": "STATUS_FULL_TIME", "neutral": False}
        ]
    )  # fmt: skip
    before, ratings = club_ratings(pd.concat([espn, late], ignore_index=True))
    # A new club enters near the bottom of the league where it first plays
    assert before[-1, 0] < np.median(list(ratings.values()))
    assert ratings["1"] > ratings["4"] and ratings["5"] > ratings["8"]


def test_club_elo_model() -> None:
    model = ClubEloModel().fit(_espn())
    assert list(model.classes_) == ["H", "D", "A"]
    assert model.rating("1") > model.rating("8") and model.rating("nobody") is None
    probs = model.predict_proba(pd.DataFrame({"home_id": ["1", "8"], "away_id": ["8", "1"]}))
    assert probs.sum(axis=1) == pytest.approx(np.ones(2))
    assert probs[0, 0] > 0.5 and probs[1, 2] > probs[1, 0]
    assert list(model.predict(pd.DataFrame({"home_id": ["1"], "away_id": ["8"]}))) == ["H"]


# Evaluation


def test_club_backtest_tickets_and_summary() -> None:
    espn = _espn()
    backtest = evaluation.club_backtest(espn, first_year=2011)
    assert set(backtest["year"]) == {2011, 2012, 2013}
    assert set(backtest["model"]) == {evaluation.ELO_KEY, evaluation.SHARES_KEY}
    match = espn[
        (espn["date"].dt.year == 2012)
        & (espn["home_team"] == "Luton Town")
        & (espn["away_team"] == "Real Madrid")
    ].iloc[0]
    linked = pd.DataFrame(
        {
            "season": ["2012-13"],
            "jornada": [5],
            "position": [2],
            "lae_home": ["Real Madrid"],
            "lae_away": ["Luton"],
            "espn_id": [match["espn_id"]],
            "swapped": [True],
            "competition": ["uefa.champions"],
            "date": [match["date"]],
            "FTR": ["H"],
            "game": ["2012 Real Madrid-Luton"],
        }
    )
    public = {("2012-13", 5): np.tile([50.0, 30.0, 20.0], (14, 1))}
    rows = evaluation.ticket_predictions(linked, backtest, public)
    assert set(rows["model"]) == {evaluation.ELO_KEY, evaluation.SHARES_KEY, evaluation.PUBLIC_KEY}
    elo = rows[rows["model"] == evaluation.ELO_KEY].iloc[0]
    # Swapped: Real Madrid is the ticket's home team, and the favourite
    assert elo["p_H"] > elo["p_A"] and elo["home_team"] == "Real Madrid"
    summary = evaluation.summarize_clubs(backtest, rows)
    assert (
        summary["all"][evaluation.ELO_KEY]["log_loss"]
        < summary["all"][evaluation.SHARES_KEY]["log_loss"]
    )
    assert summary["tickets"][evaluation.ELO_KEY]["matches"] == 1
    assert evaluation.club_model(espn.iloc[:0]) is None
    assert evaluation.club_model(espn, before=pd.Timestamp("2010-01-01")) is not None
