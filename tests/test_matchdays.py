import pandas as pd

from quiniela.data.matchdays import assign_matchdays, league_rounds
from tests.helpers import make_matches, round_robin

TEAMS = ["A", "B", "C", "D", "E", "F"]


def test_league_rounds_regular_schedule() -> None:
    matches = round_robin("SP1", TEAMS, "2026-08-15", rounds=4)
    assert league_rounds(matches).tolist() == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3


def test_postponed_match_joins_the_round_it_is_played_in() -> None:
    matches = round_robin("SP1", TEAMS, "2026-08-15", rounds=4)
    # Round 1 match of A is played on the Tuesday after round 3 (closer to it than to round 4)
    first = matches.index[0]
    matches.loc[first, "date"] = pd.Timestamp("2026-09-01")

    rounds = league_rounds(matches)

    assert rounds[first] == 3
    # Every other match keeps its round
    assert rounds.drop(first).tolist() == [1] * 2 + [2] * 3 + [3] * 3 + [4] * 3


def test_matchdays_pair_leagues_and_keep_lonely_rounds() -> None:
    primera = round_robin("SP1", TEAMS, "2026-08-15", rounds=2)
    segunda = round_robin("SP2", ["S1", "S2", "S3", "S4"], "2026-08-16", rounds=3)
    matches = pd.concat([primera, segunda], ignore_index=True)

    out = assign_matchdays(matches)

    # Segunda's third round (no La Liga that week) is a matchday on its own
    by_league = out.join(matches["league"]).groupby(["matchday", "league"]).size()
    assert by_league.to_dict() == {
        (1, "SP1"): 3,
        (1, "SP2"): 2,
        (2, "SP1"): 3,
        (2, "SP2"): 2,
        (3, "SP2"): 2,
    }


def test_assign_matchdays_empty() -> None:
    out = assign_matchdays(make_matches([]))
    assert out.empty
    assert list(out.columns) == ["round", "matchday"]
