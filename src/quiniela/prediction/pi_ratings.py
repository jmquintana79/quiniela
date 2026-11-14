r"""Pi-ratings (v4): separate home and away team ratings learned from goal differences.

Constantinou & Fenton (2013), "Determining the level of ability of football
teams by dynamic ratings based on the relative discrepancies in scores
between adversaries", *Journal of Quantitative Analysis in Sports* 9(1).

**Ratings.** Every team $t$ has a home rating $R^H_t$ and an away rating
$R^A_t$, on a scale where a rating $r$ means an expected goal difference
against an average team of

$$
\psi(r) = \operatorname{sign}(r)\left(b^{|r|/c} - 1\right), \qquad b = 10, \; c = 3
$$

so the predicted goal difference of a match between home team $h$ and away
team $a$ is $\hat g = \psi(R^H_h) - \psi(R^A_a)$.

**Learning.** After every match, with real goal difference $g$, the error
$e = g - \hat g$ is weighted with diminishing returns (a 5-0 teaches less
than five 1-0 wins),

$$
\phi(e) = \operatorname{sign}(e)\, c \log_b(1 + |e|)
$$

and the ratings of both teams move, the one of the ground they played at
by the learning rate $\lambda$ and the other one by a share $\gamma$ of that
change (catch-up learning):

$$
\Delta R^H_h = \lambda\,\phi(e), \quad \Delta R^A_h = \gamma\,\Delta R^H_h, \qquad
\Delta R^A_a = -\lambda\,\phi(e), \quad \Delta R^H_a = \gamma\,\Delta R^A_a
$$

Matches are replayed in date order from the first season, so every match is
predicted with the ratings before it (no look-ahead).

**Promoted and relegated teams.** La Liga and Segunda teams never meet in the
league, so their ratings are relative to their own league. A team that
enters a league from below (or for the first time) takes the mean ratings of
the ``NEWCOMER_TEAMS`` lowest-rated teams of that league's previous season
(the ones that went down); one that enters from above, of the highest-rated
ones (the ones that went up).

**Parameters.** $\lambda = 0.035$ (the paper's) and $\gamma = 0.9$ (0.7 in the paper)
were chosen on seasons before the backtest (log-loss of 2010-11 to 2014-15,
mapping fitted on 2005-06 to 2009-10). $\gamma = 1$ scores about the same but
merges the two ratings of a team into one.

**Probabilities.** The predicted goal difference becomes 1X2 probabilities
with an ordered logistic regression fitted per league on the last
``calibration_seasons`` seasons of training matches (their pre-match
$\hat g$):

$$
P(A) = \sigma(\theta_1 - \beta \hat g), \qquad
P(A \cup D) = \sigma(\theta_2 - \beta \hat g), \qquad \theta_1 < \theta_2
$$
"""

import math
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

OUTCOMES = ["H", "D", "A"]

# Columns the model reads from X (and the goals when fitting)
FEATURES = ["league", "season", "date", "home_team", "away_team"]
GOALS = ["FTHG", "FTAG"]

# Teams whose ratings a newcomer to a league takes (the ones it replaces)
NEWCOMER_TEAMS = 3

# League order, top first: a team moving to an earlier league was promoted
LEAGUE_ORDER = ["SP1", "SP2"]


def _psi(rating: float, base: float, scale: float) -> float:
    """Expected goal difference against an average team for a rating."""
    return math.copysign(base ** (abs(rating) / scale) - 1.0, rating)


def _weighted_error(error: float, base: float, scale: float) -> float:
    """Goal-difference error with diminishing returns."""
    return math.copysign(scale * math.log(1.0 + abs(error), base), error)


def _rank(league: str) -> int:
    """Position of a league in ``LEAGUE_ORDER`` (unknown leagues go last)."""
    return LEAGUE_ORDER.index(league) if league in LEAGUE_ORDER else len(LEAGUE_ORDER)


class PiRatingsModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Predict 1X2 results from pi-ratings (home and away ratings per team).

    Args:
        learning_rate: How much a match moves the ratings ($\\lambda$).
        catch_up: Share of the change passed to the other ground's rating ($\\gamma$).
        base: Base of the logarithmic goal scale ($b$).
        scale: Rating points per order of magnitude of goal difference ($c$).
        calibration_seasons: Last seasons of training matches the ordered
            logistic regression is fitted on.
    """

    def __init__(
        self,
        learning_rate: float = 0.035,
        catch_up: float = 0.9,
        base: float = 10.0,
        scale: float = 3.0,
        calibration_seasons: int = 5,
    ) -> None:
        self.learning_rate = learning_rate
        self.catch_up = catch_up
        self.base = base
        self.scale = scale
        self.calibration_seasons = calibration_seasons

    # Ratings ---------------------------------------------------------------

    def _entry(
        self,
        team: str,
        league: str,
        season: str,
        ratings: dict[str, list[float]],
        where: dict[str, tuple[str, str]],
        finals: dict[tuple[str, str], dict[str, tuple[float, float]]],
    ) -> list[float]:
        """Ratings of a team in a league, setting those of a newcomer to it."""
        last = where.get(team)
        if team in ratings and last is not None and last[0] == league:
            return ratings[team]
        earlier = sorted(s for (lg, s) in finals if lg == league and s < season)
        start = [0.0, 0.0]
        if earlier:
            previous = sorted(finals[(league, earlier[-1])].values(), key=lambda r: r[0] + r[1])
            # From above (relegated): like the teams that went up; otherwise like those that went down
            from_above = last is not None and _rank(last[0]) < _rank(league)
            chosen = previous[-NEWCOMER_TEAMS:] if from_above else previous[:NEWCOMER_TEAMS]
            if chosen:
                start = [
                    float(np.mean([r[0] for r in chosen])),
                    float(np.mean([r[1] for r in chosen])),
                ]
        ratings[team] = start
        return start

    def _replay(self, X: pd.DataFrame) -> np.ndarray:
        """Replay the matches in date order; returns the pre-match predicted goal difference."""
        ratings: dict[str, list[float]] = {}
        where: dict[str, tuple[str, str]] = {}
        finals: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
        order = np.argsort(X["date"].to_numpy(), kind="stable")
        leagues = X["league"].to_numpy()[order]
        seasons = X["season"].astype(str).to_numpy()[order]
        homes = X["home_team"].to_numpy()[order]
        aways = X["away_team"].to_numpy()[order]
        goals = (X["FTHG"].to_numpy(dtype=float) - X["FTAG"].to_numpy(dtype=float))[order]
        lam, gamma, b, c = self.learning_rate, self.catch_up, self.base, self.scale
        predicted = np.empty(len(X))
        for i in range(len(X)):
            league, season, home, away = leagues[i], seasons[i], homes[i], aways[i]
            rh = self._entry(home, league, season, ratings, where, finals)
            ra = self._entry(away, league, season, ratings, where, finals)
            expected = _psi(rh[0], b, c) - _psi(ra[1], b, c)
            predicted[i] = expected
            step = lam * _weighted_error(goals[i] - expected, b, c)
            rh[0] += step
            rh[1] += gamma * step
            ra[1] -= step
            ra[0] -= gamma * step
            where[home] = where[away] = (league, season)
            table = finals.setdefault((league, season), {})
            table[home] = (rh[0], rh[1])
            table[away] = (ra[0], ra[1])
        self.ratings_ = ratings
        self.where_ = where
        self.finals_ = finals
        out = np.empty(len(X))
        out[order] = predicted
        return out

    # Calibration -----------------------------------------------------------

    @staticmethod
    def _ordered_probs(g: np.ndarray, beta: float, t1: float, t2: float) -> np.ndarray:
        """Ordered logistic probabilities of H, D and A for goal differences ``g``."""
        away = 1.0 / (1.0 + np.exp(-(t1 - beta * g)))
        not_home = 1.0 / (1.0 + np.exp(-(t2 - beta * g)))
        probs = np.column_stack([1.0 - not_home, not_home - away, away])
        return np.clip(probs, 1e-9, 1.0)

    @classmethod
    def _fit_ordered(cls, g: np.ndarray, result: np.ndarray) -> tuple[float, float, float]:
        """Maximum-likelihood ordered logistic regression (``result`` 0 H, 1 D, 2 A)."""

        def loss(params: np.ndarray) -> float:
            beta, t1, gap = params
            probs = cls._ordered_probs(g, beta, t1, t1 + math.exp(gap))
            return float(-np.log(probs[np.arange(len(g)), result]).mean())

        fit = minimize(loss, x0=np.array([1.0, -1.0, 0.0]), method="Nelder-Mead")
        beta, t1, gap = fit.x
        return float(beta), float(t1), float(t1 + math.exp(gap))

    # Estimator API ---------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Replay every training match and fit the per-league probability mapping.

        Args:
            X: Played matches with ``FEATURES`` and the goals (``FTHG``, ``FTAG``).
            y: Full-time result of each match (``H``, ``D`` or ``A``).

        Returns:
            The fitted model.
        """
        X = self._validate(X, fitting=True)
        self.classes_ = np.array(OUTCOMES)
        result = pd.Series(np.asarray(y), index=X.index).map({o: i for i, o in enumerate(OUTCOMES)})
        if result.isna().any():
            raise ValueError(f"Unknown results; expected {OUTCOMES}")
        g = self._replay(X)
        seasons = sorted(X["season"].astype(str).unique())
        recent = X["season"].astype(str).isin(seasons[-self.calibration_seasons :]).to_numpy()
        codes = result.to_numpy(dtype=int)
        self.calibration_: dict[str, tuple[float, float, float]] = {}
        # One mapping for all leagues, and one per league with enough matches
        self.default_ = self._fit_ordered(g[recent], codes[recent])
        for league in X["league"].unique():
            rows = recent & (X["league"] == league).to_numpy()
            if rows.sum() >= 300:
                self.calibration_[str(league)] = self._fit_ordered(g[rows], codes[rows])
        return self

    def expected_goal_difference(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted goal difference (home minus away) of each match.

        Args:
            X: Matches with ``FEATURES``.

        Returns:
            Array of predicted goal differences.
        """
        check_is_fitted(self, "ratings_")
        X = self._validate(X)
        # Work on copies: predicting must not change the fitted ratings
        ratings = {team: list(r) for team, r in self.ratings_.items()}
        where = dict(self.where_)
        out = np.empty(len(X))
        b, c = self.base, self.scale
        for i, (league, season, home, away) in enumerate(
            zip(X["league"], X["season"].astype(str), X["home_team"], X["away_team"], strict=True)
        ):
            rh = self._entry(home, league, season, ratings, where, self.finals_)
            ra = self._entry(away, league, season, ratings, where, self.finals_)
            where[home] = where[away] = (league, season)
            out[i] = _psi(rh[0], b, c) - _psi(ra[1], b, c)
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of each result from the pre-match ratings.

        Args:
            X: Matches with ``FEATURES``.

        Returns:
            Array of shape ``(n_matches, 3)`` in ``classes_`` order (``H``, ``D``, ``A``).
        """
        g = self.expected_goal_difference(X)
        probs = np.empty((len(X), 3))
        leagues = X["league"].to_numpy()
        for league in np.unique(leagues):
            rows = leagues == league
            beta, t1, t2 = self.calibration_.get(str(league), self.default_)
            probs[rows] = self._ordered_probs(g[rows], beta, t1, t2)
        result: np.ndarray = probs / probs.sum(axis=1, keepdims=True)
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely result of each match (``H``, ``D`` or ``A``).

        Args:
            X: Matches with ``FEATURES``.

        Returns:
            Array of predicted results.
        """
        predicted: np.ndarray = self.classes_[self.predict_proba(X).argmax(axis=1)]
        return predicted

    @staticmethod
    def _validate(X: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        """Check that X is a DataFrame with the columns the model needs."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        needed = FEATURES + (GOALS if fitting else [])
        missing = sorted(set(needed) - set(X.columns))
        if missing:
            raise ValueError(f"X is missing columns {missing}")
        return X
