"""Builders for small synthetic datasets shared by the tests."""

import pandas as pd


def make_matches(
    rows: list[tuple[str, str, str, str, int | None, int | None]],
) -> pd.DataFrame:
    """Build matches from ``(league, date, home, away, home_goals, away_goals)`` tuples.

    ``None`` goals mean the match has not been played yet.
    """
    df = pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "FTHG", "FTAG"])
    df["date"] = pd.to_datetime(df["date"])
    df["FTR"] = [
        None if pd.isna(h) else "H" if h > a else "A" if h < a else "D"
        for h, a in zip(df["FTHG"], df["FTAG"], strict=True)
    ]
    df["game"] = df["date"].dt.strftime("%Y-%m-%d") + " " + df["home_team"] + "-" + df["away_team"]
    return df


def round_robin(league: str, teams: list[str], first_day: str, rounds: int) -> pd.DataFrame:
    """Weekly rounds (every Saturday) where every team plays once, all home wins."""
    rows = []
    for r in range(rounds):
        day = pd.Timestamp(first_day) + pd.Timedelta(weeks=r)
        # Rotate the opponents so fixtures change every round
        rotated = teams[:1] + teams[1:][r % (len(teams) - 1) :] + teams[1:][: r % (len(teams) - 1)]
        for i in range(len(teams) // 2):
            home, away = rotated[i], rotated[-1 - i]
            rows.append((league, str(day.date()), home, away, 1, 0))
    return make_matches(rows)


def simulate_league(
    strengths: dict[str, float],
    first_day: str,
    rounds: int,
    league: str = "SP1",
    home_advantage: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Weekly double round-robin with Poisson goals driven by team strengths.

    Expected goals: ``exp(0.1 + home_advantage + s_home - s_away)`` for the home
    side and ``exp(0.1 + s_away - s_home)`` for the away side.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    teams = list(strengths)
    rows = []
    for r in range(rounds):
        day = pd.Timestamp(first_day) + pd.Timedelta(weeks=r)
        order = rng.permutation(teams)
        for i in range(0, len(order) - 1, 2):
            home, away = order[i], order[i + 1]
            lam = np.exp(0.1 + home_advantage + strengths[home] - strengths[away])
            nu = np.exp(0.1 + strengths[away] - strengths[home])
            rows.append((league, str(day.date()), home, away, rng.poisson(lam), rng.poisson(nu)))
    return make_matches(rows)
