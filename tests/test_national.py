import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quiniela.data import national
from quiniela.evaluation import national as evaluation
from quiniela.io import international
from quiniela.prediction import elo_national
from quiniela.prediction.elo_national import NationalEloModel

RAW = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2024-03-21,Spain,England,2,0,Friendly,Madrid,Spain,FALSE
2024-03-21,Wales,Portugal,NA,NA,Friendly,Cardiff,Wales,FALSE
2024-06-20,England,Spain,1,1,UEFA Euro,Berlin,Germany,TRUE
2024-06-10,Portugal,Wales,3,0,FIFA World Cup qualification,Lisbon,Portugal,FALSE
"""


def _internationals(years: range = range(2000, 2013), seed: int = 0) -> pd.DataFrame:
    """Round robins of four national teams of known strength, two a year."""
    rng = np.random.default_rng(seed)
    strength = {"Spain": 0.8, "England": 0.4, "Wales": -0.3, "Malta": -1.0}
    teams = list(strength)
    rows = []
    for year in years:
        for month in (3, 9):
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    gap = strength[home] - strength[away]
                    rows.append(
                        {
                            "date": pd.Timestamp(year, month, 1 + len(rows) % 27),
                            "home_team": home,
                            "away_team": away,
                            "home_score": rng.poisson(np.exp(0.3 + 0.5 * gap)),
                            "away_score": rng.poisson(np.exp(0.0 - 0.5 * gap)),
                            "tournament": "UEFA Euro qualification" if month == 9 else "Friendly",
                            "city": "",
                            "country": home,
                            "neutral": False,
                        }
                    )
    raw = pd.DataFrame(rows)
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
    return international.parse_results(raw)


# International results


def test_parse_results() -> None:
    results = international.parse_results(pd.read_csv(io.StringIO(RAW)))
    # Matches without a score are left out, and the rest are sorted by date
    assert results["home_team"].tolist() == ["Spain", "Portugal", "England"]
    assert results["FTR"].tolist() == ["H", "H", "D"]
    assert results["neutral"].tolist() == [False, False, True]
    assert list(results.columns) == international.COLUMNS


def test_download_and_load_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

    monkeypatch.setattr(
        international.urllib.request, "urlopen", lambda *_, **__: Response(RAW.encode())
    )
    path = international.download_results(tmp_path)
    assert path == tmp_path / international.RESULTS_FILE
    assert len(international.load_results(tmp_path)) == 3
    # A file that is not the results keeps the saved one
    monkeypatch.setattr(
        international.urllib.request, "urlopen", lambda *_, **__: Response(b"<html>")
    )
    with pytest.raises(ValueError):
        international.download_results(tmp_path)
    assert len(international.load_results(tmp_path)) == 3
    assert international.load_results(tmp_path / "missing").empty


# LAE names


def test_national_team_names() -> None:
    assert national.national_team("Inglaterra") == "England"
    assert national.national_team("Gales (m)") == "Wales"
    assert national.national_team("Rep. Checa (M)") == "Czech Republic"
    # Clubs, women's and under-21 teams are not national teams of the results file
    assert national.national_team("Barcelona") is None
    assert national.national_team("España Fem") is None
    assert national.national_team("Italia S21") is None
    assert national.national_team(None) is None
    assert national.is_national("España", "Malta") and not national.is_national("España", "Celta")


def test_ticket_national_links_both_orders() -> None:
    results = international.parse_results(pd.read_csv(io.StringIO(RAW)))
    tickets = pd.DataFrame(
        {
            "season": ["2023-24"] * 4,
            "jornada": [40, 40, 50, 50],
            "position": [1, 2, 1, 2],
            "draw_date": pd.to_datetime(["2024-03-24", "2024-03-24", "2024-06-23", "2024-06-23"]),
            "lae_home": ["España", "Barcelona", "España", "Gales"],
            "lae_away": ["Inglaterra", "Celta", "Inglaterra (m)", "Portugal"],
            "kickoff": pd.to_datetime(["2024-03-21", None, None, "2024-06-10"]),
            "sign": ["H", "H", None, None],
        }
    )
    linked = national.ticket_national(tickets, results)
    assert linked["jornada"].tolist() == [40, 50, 50]
    # The Euro match is listed England-Spain in the results: swapped, and the draw stays a draw
    euro = linked.iloc[1]
    assert euro["swapped"] and euro["neutral"] and euro["FTR"] == "D"
    # Portugal won 3-0 at home: from the ticket's home team (Wales) it is an away win
    assert linked.iloc[2]["swapped"] and linked.iloc[2]["FTR"] == "A"
    assert linked.iloc[0]["game"] == "2024-03-21 Spain-England"
    assert national.ticket_national(tickets, results.iloc[:0]).empty


# Elo model (v1N)


def test_tournament_weights_and_goal_factor() -> None:
    assert elo_national.tournament_k("FIFA World Cup") == 60
    assert elo_national.tournament_k("UEFA Euro") == 50
    assert elo_national.tournament_k("UEFA Euro qualification") == 40
    assert elo_national.tournament_k("UEFA Nations League") == 40
    assert elo_national.tournament_k("Friendly") == 20
    assert elo_national.tournament_k("Merdeka Tournament") == 30
    assert [elo_national.goal_factor(n) for n in (0, 1, -2, 3, 5)] == [1, 1, 1.5, 1.75, 2]


def test_elo_ratings_update() -> None:
    matches = pd.DataFrame(
        {
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "FTHG": [1, 0],
            "FTAG": [0, 0],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, False],
        }
    )
    before, ratings = elo_national.elo_ratings(matches)
    # Equal teams at a neutral ground: 20 * 1 * (1 - 0.5) = 10 points
    assert before[0].tolist() == [1500, 1500]
    assert before[1].tolist() == pytest.approx([1510, 1490])
    # A draw at home for the stronger team loses it points; the total stays the same
    assert ratings["A"] < 1510 and ratings["A"] + ratings["B"] == pytest.approx(3000)


def test_national_elo_model() -> None:
    results = _internationals()
    model = NationalEloModel().fit(results)
    assert list(model.classes_) == ["H", "D", "A"]
    assert model.rating("Spain") > model.rating("England") > model.rating("Malta")
    assert model.rating("Nowhere") == elo_national.INITIAL_RATING
    fixtures = pd.DataFrame(
        {
            "home_team": ["Spain", "Malta", "Spain"],
            "away_team": ["Malta", "Spain", "Malta"],
            "neutral": [False, False, True],
        }
    )
    probs = model.predict_proba(fixtures)
    assert probs.sum(axis=1) == pytest.approx(np.ones(3))
    assert probs[0, 0] > 0.6 and probs[1, 2] > 0.5
    # Home advantage: Spain is less likely to win at a neutral ground than at home
    assert probs[2, 0] < probs[0, 0]
    assert list(model.predict(fixtures)) == ["H", "A", "H"]


# Evaluation


def test_national_backtest_and_summary() -> None:
    results = _internationals()
    backtest = evaluation.national_backtest(results, first_year=2010)
    assert set(backtest["year"]) == {2010, 2011, 2012}
    assert set(backtest["model"]) == {evaluation.ELO_KEY, evaluation.SHARES_KEY}
    elo = backtest[backtest["model"] == evaluation.ELO_KEY]
    shares = backtest[backtest["model"] == evaluation.SHARES_KEY]
    assert len(elo) == len(shares) == len(results[results["date"].dt.year >= 2010])
    tickets = pd.DataFrame(
        {
            "season": ["2011-12"],
            "jornada": [10],
            "position": [3],
            "lae_home": ["Malta"],
            "lae_away": ["España"],
            "draw_date": [pd.Timestamp("2011-09-30")],
            "kickoff": [pd.NaT],
            "sign": [None],
        }
    )
    # The Malta-Spain match of September 2011 of the synthetic results
    september = results[
        (results["date"].dt.year == 2011)
        & (results["home_team"] == "Malta")
        & (results["away_team"] == "Spain")
        & (results["date"].dt.month == 9)
    ]
    tickets["kickoff"] = september["date"].to_numpy()
    linked = national.ticket_national(tickets, results)
    public = {("2011-12", 10): np.tile([20.0, 20.0, 60.0], (14, 1))}
    rows = evaluation.ticket_predictions(linked, backtest, public)
    assert set(rows["model"]) == {evaluation.ELO_KEY, evaluation.SHARES_KEY, evaluation.PUBLIC_KEY}
    elo_row = rows[rows["model"] == evaluation.ELO_KEY].iloc[0]
    assert elo_row["p_A"] > elo_row["p_H"]
    assert rows.loc[rows["model"] == evaluation.PUBLIC_KEY, "p_A"].iloc[0] == pytest.approx(0.6)
    summary = evaluation.summarize_national(backtest, rows)
    assert summary["all"][evaluation.ELO_KEY]["matches"] == len(elo)
    # v1N beats the result shares on the synthetic internationals
    assert (
        summary["all"][evaluation.ELO_KEY]["log_loss"]
        < summary["all"][evaluation.SHARES_KEY]["log_loss"]
    )
    assert summary["tickets"][evaluation.ELO_KEY]["matches"] == 1
    assert set(summary["public"]) == {
        evaluation.ELO_KEY,
        evaluation.SHARES_KEY,
        evaluation.PUBLIC_KEY,
    }
    assert set(summary["by_year"]) == {"2010", "2011", "2012"}
    assert evaluation.national_model(results.iloc[:0]) is None
    assert evaluation.national_model(results, before=pd.Timestamp("2005-01-01")) is not None
