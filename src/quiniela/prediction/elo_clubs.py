r"""Club Elo (v1C): one rating of every club of many leagues and cups, mapped to 1X2.

The ratings are learned match by match on every ESPN match of
``quiniela.io.espn_clubs`` (the main European leagues, the Swedish,
Norwegian, Brazilian, Argentinian, American and Mexican ones, the European
cups and their qualifiers, the Copa del Rey and a few other cups) since 2009.
Teams are ESPN ids, the same in every competition, so the European cups link
the leagues of every country into one scale. After each match,

$$
R' = R + K \cdot G \cdot (W - W_e), \qquad
W_e = \frac{1}{1 + 10^{-(R - R_{opp} + H)/400}}
$$

with $K$ = ``K_FACTOR`` (the same for every competition), $H$ =
``HOME_ADVANTAGE`` points (0 at a neutral ground) and $G$ the goal factor of
the national-team Elo (``quiniela.prediction.elo_national.goal_factor``).

**New teams.** A team seen for the first time starts at the
``ENTRY_QUANTILE`` quantile of the ratings of the last matches of that
competition: a promoted team, a small club of the Copa del Rey or the
champion of a small country in the qualifiers starts near the bottom of the
competition where it appears, not at the average.

**From ratings to 1X2.** As in v1N: an ordered logistic regression on the Elo
difference before the match divided by 400 and whether the home team plays
at home (``quiniela.prediction.elo_national.ordinal_probabilities``), fitted
on the last ``window_years`` years.

**Backtest** (``quiniela.evaluation.clubs``): 50.7% of hits and a log-loss of
1.005 on the ESPN matches from 2011 (v3, with the market, gets 50.0% and 1.003
on La Liga and Segunda); 55.1% and 0.959 on the club matches of the official
tickets, against 45.6% and 1.061 for the shares of 1, X and 2.
"""

from collections import defaultdict, deque
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from quiniela.prediction.elo_national import (
    OUTCOMES,
    fit_ordinal,
    goal_factor,
    ordinal_probabilities,
)

# Rating change weight, Elo points of playing at home and the rating of a new team
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0
INITIAL_RATING = 1500.0
ENTRY_QUANTILE = 0.1

# Ratings of the last matches of a competition a new team's rating is taken from
ENTRY_POOL = 400
MIN_POOL = 20

# Years of matches the 1X2 regression is fitted on
WINDOW_YEARS = 8


def club_ratings(
    matches: pd.DataFrame,
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    entry_quantile: float = ENTRY_QUANTILE,
) -> tuple[np.ndarray, dict[str, float]]:
    """Ratings of both teams before every match, and every team's rating after the last one.

    Args:
        matches: Played matches in date order (``competition``, ``home_id``,
            ``away_id``, ``FTHG``, ``FTAG``, ``neutral``).
        k_factor: Weight $K$ of the rating changes.
        home_advantage: Elo points of the home team (not at a neutral ground).
        entry_quantile: Quantile of the competition's recent ratings a new team starts at.

    Returns:
        ``(before, ratings)``: shape ``(n, 2)`` home and away ratings before
        each match, and the final rating of every team (by ESPN id).
    """
    ratings: dict[str, float] = {}
    recent: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=ENTRY_POOL))
    before = np.zeros((len(matches), 2))
    neutral = _neutral(matches)
    columns = zip(
        matches["competition"].astype(str),
        matches["home_id"].astype(str),
        matches["away_id"].astype(str),
        matches["FTHG"].astype(int),
        matches["FTAG"].astype(int),
        neutral,
        strict=True,
    )
    for i, (competition, home, away, home_goals, away_goals, at_neutral) in enumerate(columns):
        pool = recent[competition]
        for team in (home, away):
            if team not in ratings:
                ratings[team] = (
                    float(np.quantile(pool, entry_quantile))
                    if len(pool) >= MIN_POOL
                    else INITIAL_RATING
                )
        rating_home, rating_away = ratings[home], ratings[away]
        before[i] = rating_home, rating_away
        gap = rating_home - rating_away + (0.0 if at_neutral else home_advantage)
        expected = 1 / (10 ** (-gap / 400) + 1)
        result = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        change = k_factor * goal_factor(home_goals - away_goals) * (result - expected)
        ratings[home] = rating_home + change
        ratings[away] = rating_away - change
        pool.extend((rating_home, rating_away))
    return before, ratings


def _neutral(matches: pd.DataFrame) -> np.ndarray:
    """Whether each match is at a neutral ground (``False`` without the column)."""
    if "neutral" not in matches.columns:
        return np.zeros(len(matches), dtype=bool)
    neutral: np.ndarray = matches["neutral"].fillna(False).astype(bool).to_numpy()
    return neutral


class ClubEloModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Elo ratings of every club of the ESPN competitions mapped to 1X2 (v1C).

    Args:
        k_factor: Weight of the rating changes.
        home_advantage: Elo points of the home team in the rating updates.
        entry_quantile: Quantile of a competition's recent ratings a new team starts at.
        window_years: Years of matches (up to the last one) the 1X2 regression is fitted on.
    """

    def __init__(
        self,
        k_factor: float = K_FACTOR,
        home_advantage: float = HOME_ADVANTAGE,
        entry_quantile: float = ENTRY_QUANTILE,
        window_years: int = WINDOW_YEARS,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.entry_quantile = entry_quantile
        self.window_years = window_years

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Rate every club on the played matches and fit the 1X2 regression.

        Args:
            X: Played ESPN club matches in date order (``date``,
                ``competition``, ``home_id``, ``away_id``, ``FTHG``, ``FTAG``,
                ``neutral``).
            y: Results (``H``, ``D``, ``A``); defaults to ``X["FTR"]``.

        Returns:
            The fitted model.
        """
        result = X["FTR"] if y is None else y
        before, self.ratings_ = club_ratings(
            X, self.k_factor, self.home_advantage, self.entry_quantile
        )
        dates = pd.to_datetime(X["date"])
        recent = (dates > dates.max() - pd.DateOffset(years=self.window_years)).to_numpy()
        codes = pd.Series(result).map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
        self.coef_ = fit_ordinal(
            (before[recent, 0] - before[recent, 1]) / 400,
            (~_neutral(X)[recent]).astype(float),
            codes[recent].astype(int),
        )
        self.classes_ = np.array(OUTCOMES)
        return self

    def rating(self, team: str) -> float | None:
        """Current rating of a club by its ESPN id (``None`` for a club never seen)."""
        value = self.ratings_.get(str(team))
        return None if value is None else float(value)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilities of 1, X and 2 from the current ratings.

        Args:
            X: Matches (``home_id``, ``away_id`` and, if known, ``neutral``);
                a club never seen gets the initial rating.

        Returns:
            Shape ``(n, 3)`` in ``classes_`` order.
        """
        homes = np.array([self.ratings_.get(str(t), INITIAL_RATING) for t in X["home_id"]])
        aways = np.array([self.ratings_.get(str(t), INITIAL_RATING) for t in X["away_id"]])
        return ordinal_probabilities(
            self.coef_, (homes - aways) / 400, (~_neutral(X)).astype(float)
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely result of each match."""
        predicted: np.ndarray = self.classes_[self.predict_proba(X).argmax(axis=1)]
        return predicted
