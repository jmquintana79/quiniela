r"""National-team Elo (v1N): World Football Elo ratings mapped to 1X2.

The ratings follow the World Football Elo system (eloratings.net), computed
on every men's international since 1872 (``quiniela.io.international``).
Every team starts at 1500 and, after each match,

$$
R' = R + K \cdot G \cdot (W - W_e), \qquad
W_e = \frac{1}{1 + 10^{-(R - R_{opp} + H)/400}}
$$

with $W$ 1, 0.5 or 0 (win, draw, loss), $H$ = ``HOME_ADVANTAGE`` points for
the home team (0 at a neutral ground), $K$ by tournament (``tournament_k``:
60 for the World Cup, 50 for the continental championships, 40 for the
qualifiers and the Nations Leagues, 30 for other tournaments, 20 for
friendlies) and $G$ by goal difference $N$ (1 for $N \le 1$, 1.5 for 2,
$(11 + N) / 8$ for 3 or more).

**From ratings to 1X2.** The Elo difference before the match, $d = (R_h -
R_a) / 400$, and whether the home team plays at home, $h$, feed an ordered
logistic regression,

$$
P(\text{1}) = \sigma(\beta d + \gamma h - c_1), \qquad
P(\text{1 or X}) = \sigma(\beta d + \gamma h - c_2), \qquad c_2 < c_1,
$$

fitted by maximum likelihood on the matches of the last ``window_years``
years (recent football: fewer draws between unequal teams than in the old
days).

**Backtest** (``quiniela.evaluation.national``, every international from 2010,
the regression refitted every year on the earlier ones): 59.4% of hits and a
log-loss of 0.885 on 16,512 matches, against 1.050 for the shares of 1, X and
2; 60.2% and 0.874 on the 840 national-team matches of the official tickets.
"""

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

OUTCOMES = ["H", "D", "A"]

# Rating of a team in its first match, and Elo points of playing at home
INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0

# Years of matches the 1X2 regression is fitted on
WINDOW_YEARS = 12


def tournament_k(tournament: str) -> float:
    """Weight $K$ of a match by its tournament (World Football Elo).

    Args:
        tournament: Tournament as the results file writes it.

    Returns:
        60, 50, 40, 30 or 20.
    """
    name = tournament.lower()
    if name == "fifa world cup":
        return 60.0
    continental = (
        "uefa euro",
        "copa américa",
        "african cup of nations",
        "afc asian cup",
        "gold cup",
        "confederations cup",
    )
    if name in continental:
        return 50.0
    if "qualification" in name or "nations league" in name:
        return 40.0
    if name == "friendly":
        return 20.0
    return 30.0


def goal_factor(difference: int) -> float:
    """Multiplier $G$ of the rating change by the goal difference."""
    goals = abs(int(difference))
    if goals <= 1:
        return 1.0
    if goals == 2:
        return 1.5
    return (11 + goals) / 8


def elo_ratings(
    matches: pd.DataFrame, home_advantage: float = HOME_ADVANTAGE
) -> tuple[np.ndarray, dict[str, float]]:
    """Ratings of both teams before every match, and every team's rating after the last one.

    Args:
        matches: Played matches in date order (``home_team``, ``away_team``,
            ``FTHG``, ``FTAG``, ``tournament``, ``neutral``).
        home_advantage: Elo points of the home team (not at a neutral ground).

    Returns:
        ``(before, ratings)``: shape ``(n, 2)`` home and away ratings before
        each match, and the final rating of every team.
    """
    ratings: dict[str, float] = {}
    before = np.zeros((len(matches), 2))
    neutral = _neutral(matches)
    columns = zip(
        matches["home_team"],
        matches["away_team"],
        matches["FTHG"].astype(int),
        matches["FTAG"].astype(int),
        matches["tournament"].astype(str),
        neutral,
        strict=True,
    )
    for i, (home, away, home_goals, away_goals, tournament, at_neutral) in enumerate(columns):
        rating_home = ratings.get(home, INITIAL_RATING)
        rating_away = ratings.get(away, INITIAL_RATING)
        before[i] = rating_home, rating_away
        gap = rating_home - rating_away + (0.0 if at_neutral else home_advantage)
        expected = 1 / (10 ** (-gap / 400) + 1)
        result = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        change = (
            tournament_k(tournament) * goal_factor(home_goals - away_goals) * (result - expected)
        )
        ratings[home] = rating_home + change
        ratings[away] = rating_away - change
    return before, ratings


def ordinal_probabilities(coef: np.ndarray, difference: np.ndarray, home: np.ndarray) -> np.ndarray:
    """1X2 probabilities of the ordered logistic regression.

    Args:
        coef: ``(beta, gamma, c1, log(c1 - c2))``.
        difference: Elo difference before the match divided by 400.
        home: 1 when the home team plays at home, 0 at a neutral ground.

    Returns:
        Shape ``(n, 3)``: probabilities of 1, X and 2.
    """
    beta, gamma, first, log_gap = coef
    score = beta * difference + gamma * home
    home_win = 1 / (1 + np.exp(-(score - first)))
    not_away = 1 / (1 + np.exp(-(score - (first - np.exp(log_gap)))))
    probs = np.column_stack([home_win, not_away - home_win, 1 - not_away])
    clipped: np.ndarray = np.clip(probs, 1e-9, 1.0)
    return clipped


def fit_ordinal(difference: np.ndarray, home: np.ndarray, result: np.ndarray) -> np.ndarray:
    """Maximum-likelihood coefficients of ``ordinal_probabilities``.

    Args:
        difference: Elo differences divided by 400.
        home: 1 at home, 0 at a neutral ground.
        result: 0, 1 or 2 (1, X, 2).

    Returns:
        ``(beta, gamma, c1, log(c1 - c2))``.
    """
    rows = np.arange(len(result))

    def loss(coef: np.ndarray) -> float:
        probs = ordinal_probabilities(coef, difference, home)
        return float(-np.log(probs[rows, result]).mean())

    start = np.array([2.0, 0.5, 0.6, np.log(1.3)])
    fit = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-5})
    coef: np.ndarray = fit.x
    return coef


def _neutral(matches: pd.DataFrame) -> np.ndarray:
    """Whether each match is at a neutral ground (``False`` without the column)."""
    if "neutral" not in matches.columns:
        return np.zeros(len(matches), dtype=bool)
    neutral: np.ndarray = matches["neutral"].fillna(False).astype(bool).to_numpy()
    return neutral


class NationalEloModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """World Football Elo ratings of the national teams mapped to 1X2 (v1N).

    Args:
        home_advantage: Elo points of the home team in the rating updates.
        window_years: Years of matches (up to the last one) the 1X2
            regression is fitted on.
    """

    def __init__(
        self, home_advantage: float = HOME_ADVANTAGE, window_years: int = WINDOW_YEARS
    ) -> None:
        self.home_advantage = home_advantage
        self.window_years = window_years

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Rate every team on the played matches and fit the 1X2 regression.

        Args:
            X: Played international matches in date order (``date``,
                ``home_team``, ``away_team``, ``FTHG``, ``FTAG``,
                ``tournament``, ``neutral``).
            y: Results (``H``, ``D``, ``A``); defaults to ``X["FTR"]``.

        Returns:
            The fitted model.
        """
        result = X["FTR"] if y is None else y
        before, self.ratings_ = elo_ratings(X, self.home_advantage)
        recent = (
            pd.to_datetime(X["date"])
            > pd.to_datetime(X["date"]).max() - pd.DateOffset(years=self.window_years)
        ).to_numpy()
        codes = pd.Series(result).map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
        self.coef_ = fit_ordinal(
            (before[recent, 0] - before[recent, 1]) / 400,
            (~_neutral(X)[recent]).astype(float),
            codes[recent].astype(int),
        )
        self.classes_ = np.array(OUTCOMES)
        return self

    def rating(self, team: str) -> float:
        """Current rating of a team (``INITIAL_RATING`` for a team never seen)."""
        return float(self.ratings_.get(team, INITIAL_RATING))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilities of 1, X and 2 from the current ratings.

        Args:
            X: Matches (``home_team``, ``away_team`` and, if known, ``neutral``).

        Returns:
            Shape ``(n, 3)`` in ``classes_`` order.
        """
        homes = np.array([self.rating(team) for team in X["home_team"]])
        aways = np.array([self.rating(team) for team in X["away_team"]])
        return ordinal_probabilities(
            self.coef_, (homes - aways) / 400, (~_neutral(X)).astype(float)
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely result of each match."""
        predicted: np.ndarray = self.classes_[self.predict_proba(X).argmax(axis=1)]
        return predicted
