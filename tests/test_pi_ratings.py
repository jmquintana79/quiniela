import numpy as np
import pandas as pd
import pytest

from quiniela.prediction.pi_ratings import PiRatingsModel
from tests.helpers import make_matches, simulate_league


def _league(seed: int = 0, league: str = "SP1", season: str = "2020-21") -> pd.DataFrame:
    strengths = {"Strong": 0.8, "Good": 0.3, "Mid": 0.0, "Weak": -0.4, "Poor": -0.6, "Bad": -0.8}
    matches = simulate_league(strengths, "2020-08-15", 40, league=league, seed=seed)
    return matches.assign(season=season)


def test_ratings_rank_the_teams_and_probabilities_add_up() -> None:
    matches = _league()
    model = PiRatingsModel().fit(matches, matches["FTR"])
    total = {team: sum(r) for team, r in model.ratings_.items()}
    assert total["Strong"] > total["Mid"] > total["Bad"]

    upcoming = matches.head(4).drop(columns=["FTHG", "FTAG", "FTR"])
    probs = model.predict_proba(upcoming)
    assert probs.shape == (4, 3) and np.allclose(probs.sum(axis=1), 1)
    assert list(model.classes_) == ["H", "D", "A"]
    strong_home = upcoming.assign(home_team="Strong", away_team="Bad")
    weak_home = upcoming.assign(home_team="Bad", away_team="Strong")
    assert model.predict_proba(strong_home)[0, 0] > model.predict_proba(weak_home)[0, 0]
    assert model.predict(strong_home)[0] == "H"


def test_one_match_moves_both_ratings() -> None:
    rows = [("SP1", "2020-08-15", "A", "B", 2, 0)]
    matches = make_matches(rows).assign(season="2020-21")
    model = PiRatingsModel(learning_rate=0.1, catch_up=0.5)
    g = model._replay(matches)
    assert g[0] == 0  # both start at zero
    home, away = model.ratings_["A"], model.ratings_["B"]
    step = 0.1 * 3 * np.log10(3)
    assert home[0] == pytest.approx(step) and home[1] == pytest.approx(0.5 * step)
    assert away[1] == pytest.approx(-step) and away[0] == pytest.approx(-0.5 * step)


def test_promoted_team_takes_the_ratings_of_the_relegated_ones() -> None:
    first = _league(season="2020-21")
    # Next season a Segunda team replaces "Bad" in La Liga
    second = first.assign(
        season="2021-22",
        date=first["date"] + pd.Timedelta(days=365),
        home_team=first["home_team"].replace("Bad", "Newcomer"),
        away_team=first["away_team"].replace("Bad", "Newcomer"),
    )
    segunda = make_matches([("SP2", "2021-04-01", "Newcomer", "Other", 3, 0)]).assign(
        season="2020-21"
    )
    model = PiRatingsModel().fit(pd.concat([first, segunda]), pd.concat([first, segunda])["FTR"])
    finals = model.finals_[("SP1", "2020-21")]
    lowest = sorted(finals.values(), key=sum)[:3]
    upcoming = second.head(1).drop(columns=["FTHG", "FTAG", "FTR"])
    upcoming = upcoming.assign(home_team="Newcomer", away_team="Strong")
    g = model.expected_goal_difference(upcoming)
    # Its Segunda rating is ignored: it starts like the three lowest La Liga teams
    home = np.mean([r[0] for r in lowest])
    expected = np.sign(home) * (10 ** (abs(home) / 3) - 1)
    away = model.ratings_["Strong"][1]
    expected -= np.sign(away) * (10 ** (abs(away) / 3) - 1)
    assert g[0] == pytest.approx(expected) and g[0] < 0
    # Predicting does not change the fitted ratings
    before = {k: list(v) for k, v in model.ratings_.items()}
    model.predict_proba(upcoming)
    assert model.ratings_ == before


def test_missing_columns() -> None:
    matches = _league()
    with pytest.raises(ValueError):
        PiRatingsModel().fit(matches.drop(columns=["season"]), matches["FTR"])
    with pytest.raises(TypeError):
        PiRatingsModel().fit(matches.to_numpy(), matches["FTR"])


def test_shots_model_learns_from_shots_on_target() -> None:
    from quiniela.prediction.pi_ratings_shots_f import PiRatingsShotsModel
    from tests.helpers import make_matches

    # Every match ends 1-1, but the "Strong" team always has many more shots on target
    rows = []
    for week in range(30):
        date = (pd.Timestamp("2023-09-02") + pd.Timedelta(days=7 * week)).strftime("%Y-%m-%d")
        home, away = ("Strong", "Weak") if week % 2 == 0 else ("Weak", "Strong")
        rows.append(("LIGAF", date, home, away, 1, 1))
    matches = make_matches(rows).assign(season="2023-24")
    strong_home = matches["home_team"] == "Strong"
    matches["HST"] = np.where(strong_home, 8.0, 2.0)
    matches["AST"] = np.where(strong_home, 2.0, 8.0)
    target = matches.iloc[[0]]

    goals_only = PiRatingsModel().fit(matches, matches["FTR"]).expected_goal_difference(target)
    shots = PiRatingsShotsModel(shots_weight=0.5).fit(matches, matches["FTR"])
    # Goals alone see two equal teams; the shots on target make "Strong" the better one
    assert goals_only[0] == pytest.approx(0.0, abs=1e-9)
    assert shots.expected_goal_difference(target)[0] > 0.1
    # Goals per shot on target of the training matches: 60 goals / 300 shots
    assert shots.shots_scale_ == pytest.approx(0.2)
    # Without shot columns (or weight 0) it is the goals-only model
    plain = PiRatingsShotsModel().fit(matches.drop(columns=["HST", "AST"]), matches["FTR"])
    assert plain.expected_goal_difference(target)[0] == pytest.approx(0.0, abs=1e-9)
    zero = PiRatingsShotsModel(shots_weight=0.0).fit(matches, matches["FTR"])
    assert zero.expected_goal_difference(target)[0] == pytest.approx(0.0, abs=1e-9)
