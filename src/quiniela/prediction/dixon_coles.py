r"""Dixon-Coles model (v2): team attack and defence strengths with time decay.

Goals are modelled as Poisson counts whose means depend on the teams'
current strength, so a team is judged by how it plays now (weighted towards
recent matches) and against whom, instead of by its head-to-head history.

**Expected goals.** For a match of league $\ell$ between home team $h$ and
away team $a$:

$$
\lambda = \exp(\mu_\ell + \eta_\ell + \alpha_h - \delta_a), \qquad
\nu = \exp(\mu_\ell + \alpha_a - \delta_h)
$$

with $\lambda$ and $\nu$ the expected home and away goals, $\alpha_t$ the
attack and $\delta_t$ the defence of team $t$ (higher is better), $\mu_\ell$
the league's goal level and $\eta_\ell$ its home advantage. Teams keep one
set of strengths across divisions, while $\mu_\ell$ and $\eta_\ell$ adjust
for each league.

**Score probabilities.** Independent Poisson goals, corrected for low scores
(0-0, 1-0, 0-1, 1-1) with the Dixon-Coles factor $\tau$:

$$
P(X = x, Y = y) = \tau_{\lambda,\nu}(x, y)\,
\frac{\lambda^x e^{-\lambda}}{x!}\,\frac{\nu^y e^{-\nu}}{y!}
$$

$$
\tau_{\lambda,\nu}(x, y) =
\begin{cases}
1 - \lambda\nu\rho & x = y = 0 \\\\
1 + \lambda\rho & x = 0,\ y = 1 \\\\
1 + \nu\rho & x = 1,\ y = 0 \\\\
1 - \rho & x = y = 1 \\\\
1 & \text{otherwise}
\end{cases}
$$

A negative $\rho$ makes 0-0 and 1-1 more likely than independent Poisson goals
predict, which raises the draw probability.

**Fitting.** The parameters maximise the time-weighted log-likelihood of the
past matches, with a Gaussian prior (ridge penalty) that keeps teams with few
matches close to average:

$$
\max \sum_k e^{-\xi\, t_k} \log P(x_k, y_k) - \frac{1}{2\sigma^2} \sum_t \left(\alpha_t^2 + \delta_t^2\right)
$$

where $t_k$ is the age of match $k$ in days. With $\xi = 0.0018$ a match
loses half its weight in about a year, so early in a season the previous
seasons dominate and the current one takes over as it advances.

**Predictions.** Summing the score matrix gives the 1X2 probabilities (home
win below the diagonal, draw on it, away win above), the most likely exact
score and the *Pleno al 15* pick (goals of each team as 0, 1, 2 or M = 3+).
"""

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

OUTCOMES = ["H", "D", "A"]

# Columns read from X (goals only when fitting)
FEATURES = ["league", "date", "home_team", "away_team"]
GOALS = ["FTHG", "FTAG"]

# Pleno al 15 categories: 0, 1, 2 and M (3 or more goals)
PLENO_LABELS = ["0", "1", "2", "M"]

# Smallest value of tau inside the logarithm (tau can approach 0 for extreme rho)
TAU_FLOOR = 1e-10


def _tau_terms(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray, nu: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Low-score correction and its derivatives for every match.

    Returns:
        ``tau`` and the derivatives of ``log tau`` with respect to ``lambda``,
        ``nu`` and ``rho``.
    """
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_nu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1 - lam[m00] * nu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + nu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, TAU_FLOOR)

    d_lam[m00] = -nu[m00] * rho / tau[m00]
    d_nu[m00] = -lam[m00] * rho / tau[m00]
    d_rho[m00] = -lam[m00] * nu[m00] / tau[m00]
    d_lam[m01] = rho / tau[m01]
    d_rho[m01] = lam[m01] / tau[m01]
    d_nu[m10] = rho / tau[m10]
    d_rho[m10] = nu[m10] / tau[m10]
    d_rho[m11] = -1 / tau[m11]
    return tau, d_lam, d_nu, d_rho


# scikit-learn ships without type hints, so its base classes are typed as Any
class DixonColesModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    r"""Predict 1X2 results, exact scores and Pleno al 15 from team strengths.

    Args:
        xi: Time decay per day of the match weights $e^{-\xi t}$ (0 weighs
            every match equally).
        window_days: Only matches played in the last ``window_days`` days of
            the training data are used (older ones weigh almost nothing).
        prior_sd: Standard deviation $\sigma$ of the Gaussian prior on attack
            and defence strengths (smaller shrinks teams more towards average).
        max_goals: Highest number of goals per team in the score matrix.
    """

    def __init__(
        self,
        xi: float = 0.0018,
        window_days: int = 1460,
        prior_sd: float = 0.5,
        max_goals: int = 10,
    ) -> None:
        self.xi = xi
        self.window_days = window_days
        self.prior_sd = prior_sd
        self.max_goals = max_goals

    # %% Fitting

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Estimate team strengths, league goal levels, home advantage and rho.

        Args:
            X: Played matches with ``league``, ``date``, ``home_team``,
                ``away_team`` and the goals ``FTHG`` and ``FTAG`` (the model
                learns from goals, not only from the result).
            y: Full-time results; not needed (the goals already contain them),
                accepted for compatibility with the scikit-learn API.

        Returns:
            The fitted model.
        """
        self._check_columns(X, FEATURES + GOALS)
        matches = X.dropna(subset=GOALS)
        reference = matches["date"].max()
        age = (reference - matches["date"]).dt.days.to_numpy()
        matches, age = matches[age <= self.window_days], age[age <= self.window_days]
        if matches.empty:
            raise ValueError("No played matches with goals to fit the model")

        self.classes_ = np.array(OUTCOMES)
        self.teams_ = np.array(
            sorted(set(matches["home_team"]) | set(matches["away_team"])), dtype=object
        )
        self.leagues_ = np.array(sorted(matches["league"].unique()), dtype=object)
        team_index = {team: i for i, team in enumerate(self.teams_)}
        league_index = {league: i for i, league in enumerate(self.leagues_)}

        home = matches["home_team"].map(team_index).to_numpy()
        away = matches["away_team"].map(team_index).to_numpy()
        league = matches["league"].map(league_index).to_numpy()
        x = matches["FTHG"].to_numpy(dtype=float)
        yy = matches["FTAG"].to_numpy(dtype=float)
        weights = np.exp(-self.xi * age)

        n_teams, n_leagues = len(self.teams_), len(self.leagues_)
        start = np.zeros(2 * n_teams + 2 * n_leagues + 1)
        # Start from each league's average away goals and home/away goal ratio
        for i in range(n_leagues):
            in_league = league == i
            start[2 * n_teams + i] = np.log(max(yy[in_league].mean(), 0.1))
            start[2 * n_teams + n_leagues + i] = np.log(
                max(x[in_league].mean(), 0.1) / max(yy[in_league].mean(), 0.1)
            )

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            attack, defence = params[:n_teams], params[n_teams : 2 * n_teams]
            level = params[2 * n_teams : 2 * n_teams + n_leagues]
            home_adv = params[2 * n_teams + n_leagues : -1]
            rho = params[-1]

            lam = np.exp(level[league] + home_adv[league] + attack[home] - defence[away])
            nu = np.exp(level[league] + attack[away] - defence[home])
            tau, dt_lam, dt_nu, dt_rho = _tau_terms(x, yy, lam, nu, rho)

            loglik = x * np.log(lam) - lam + yy * np.log(nu) - nu + np.log(tau)
            penalty = (attack @ attack + defence @ defence) / (2 * self.prior_sd**2)
            value = -(weights @ loglik) + penalty

            # Derivatives of the weighted log-likelihood w.r.t. log(lambda) and log(nu)
            g_lam = weights * (x - lam + lam * dt_lam)
            g_nu = weights * (yy - nu + nu * dt_nu)
            grad = np.zeros_like(params)
            grad[:n_teams] = np.bincount(home, g_lam, n_teams) + np.bincount(away, g_nu, n_teams)
            grad[n_teams : 2 * n_teams] = -np.bincount(away, g_lam, n_teams) - np.bincount(
                home, g_nu, n_teams
            )
            grad[2 * n_teams : 2 * n_teams + n_leagues] = np.bincount(
                league, g_lam + g_nu, n_leagues
            )
            grad[2 * n_teams + n_leagues : -1] = np.bincount(league, g_lam, n_leagues)
            grad[-1] = weights @ dt_rho
            grad = -grad
            grad[: 2 * n_teams] += params[: 2 * n_teams] / self.prior_sd**2
            return float(value), grad

        bounds = [(None, None)] * (len(start) - 1) + [(-0.3, 0.3)]
        result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds)

        params = result.x
        self.attack_ = pd.Series(params[:n_teams], index=self.teams_)
        self.defence_ = pd.Series(params[n_teams : 2 * n_teams], index=self.teams_)
        self.level_ = pd.Series(params[2 * n_teams : 2 * n_teams + n_leagues], index=self.leagues_)
        self.home_advantage_ = pd.Series(params[2 * n_teams + n_leagues : -1], index=self.leagues_)
        self.rho_ = float(params[-1])
        self.converged_ = bool(result.success)
        return self

    # %% Predictions

    def expected_goals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected home and away goals of each match.

        Teams not seen in training get average strength (0); leagues not seen
        get the average goal level and home advantage.

        Args:
            X: Matches with ``league``, ``home_team`` and ``away_team``.

        Returns:
            ``(lambda, nu)``: expected home and away goals.
        """
        check_is_fitted(self, "attack_")
        self._check_columns(X, ["league", "home_team", "away_team"])
        attack_h = self.attack_.reindex(X["home_team"]).fillna(0).to_numpy()
        attack_a = self.attack_.reindex(X["away_team"]).fillna(0).to_numpy()
        defence_h = self.defence_.reindex(X["home_team"]).fillna(0).to_numpy()
        defence_a = self.defence_.reindex(X["away_team"]).fillna(0).to_numpy()
        level = self.level_.reindex(X["league"]).fillna(self.level_.mean()).to_numpy()
        home_adv = (
            self.home_advantage_.reindex(X["league"]).fillna(self.home_advantage_.mean()).to_numpy()
        )
        lam = np.exp(level + home_adv + attack_h - defence_a)
        nu = np.exp(level + attack_a - defence_h)
        return lam, nu

    def score_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of every score up to ``max_goals`` goals per team.

        Args:
            X: Matches with ``league``, ``home_team`` and ``away_team``.

        Returns:
            Array of shape ``(n_matches, max_goals + 1, max_goals + 1)`` where
            ``[i, x, y]`` is the probability that match ``i`` ends ``x``-``y``
            (normalised to add up to 1).
        """
        lam, nu = self.expected_goals(X)
        goals = np.arange(self.max_goals + 1)
        matrix = poisson.pmf(goals[None, :, None], lam[:, None, None]) * poisson.pmf(
            goals[None, None, :], nu[:, None, None]
        )
        # Low-score correction
        matrix[:, 0, 0] *= np.maximum(1 - lam * nu * self.rho_, TAU_FLOOR)
        matrix[:, 0, 1] *= 1 + lam * self.rho_
        matrix[:, 1, 0] *= 1 + nu * self.rho_
        matrix[:, 1, 1] *= 1 - self.rho_
        normalised: np.ndarray = matrix / matrix.sum(axis=(1, 2), keepdims=True)
        return normalised

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of a home win, a draw and an away win.

        Args:
            X: Matches with ``league``, ``home_team`` and ``away_team``.

        Returns:
            Array of shape ``(n_matches, 3)`` in ``H``, ``D``, ``A`` order.
        """
        matrix = self.score_matrix(X)
        home = np.tril(np.ones(matrix.shape[1:]), k=-1)  # x > y
        return np.stack(
            [
                (matrix * home).sum(axis=(1, 2)),
                np.trace(matrix, axis1=1, axis2=2),
                (matrix * home.T).sum(axis=(1, 2)),
            ],
            axis=1,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely result (``H``, ``D`` or ``A``) of each match.

        Args:
            X: Matches with ``league``, ``home_team`` and ``away_team``.

        Returns:
            Array of results.
        """
        # predict_proba first: it raises NotFittedError before classes_ is read
        best = self.predict_proba(X).argmax(axis=1)
        predicted: np.ndarray = self.classes_[best]
        return predicted

    def predict_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Most likely exact score and Pleno al 15 pick of each match.

        Args:
            X: Matches with ``league``, ``home_team`` and ``away_team``.

        Returns:
            DataFrame aligned with ``X`` with ``score_home``, ``score_away``
            (most likely score) and ``pleno_home``, ``pleno_away`` (most
            likely goals category of each team: ``0``, ``1``, ``2`` or ``M``).
        """
        matrix = self.score_matrix(X)
        n, size = matrix.shape[0], matrix.shape[1]
        best = matrix.reshape(n, -1).argmax(axis=1)

        # Collapse goals into the four Pleno al 15 categories: 0, 1, 2 and M (3+)
        blocks = [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, None)]
        pleno = np.zeros((n, len(PLENO_LABELS), len(PLENO_LABELS)))
        for i, home_goals in enumerate(blocks):
            for j, away_goals in enumerate(blocks):
                pleno[:, i, j] = matrix[:, home_goals, away_goals].sum(axis=(1, 2))
        best_pleno = pleno.reshape(n, -1).argmax(axis=1)
        labels = np.asarray(PLENO_LABELS)

        return pd.DataFrame(
            {
                "score_home": best // size,
                "score_away": best % size,
                "pleno_home": labels[best_pleno // len(PLENO_LABELS)],
                "pleno_away": labels[best_pleno % len(PLENO_LABELS)],
            },
            index=X.index,
        )

    @staticmethod
    def _check_columns(X: pd.DataFrame, columns: list[str]) -> None:
        """Check that X is a DataFrame with the given columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        missing = sorted(set(columns) - set(X.columns))
        if missing:
            raise ValueError(f"X is missing columns {missing}")
