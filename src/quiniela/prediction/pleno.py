r"""Pleno al 15 from the market (v2P): the goals categories the odds expect.

The Pleno al 15 is the exact score of one match in goals categories, 0, 1, 2 or M
(3 or more) for each team: 16 possible picks, one per ticket. Its own counter has a
P: **v1P** is the most likely pair of categories of Dixon-Coles (v2's score matrix,
``quiniela.prediction.dixon_coles``), **v2P** the same from the market.

**Expected goals from the odds.** The 1X2 odds and the over / under 2.5 goals odds
(margin removed) are three numbers, $m_1, m_X, m_2$ and $m_{>2.5}$; a score model
with two expected goals $\lambda_h$, $\lambda_a$ has the same three. They are fitted
by minimising the cross-entropy between both,

$$
\min_{\lambda_h, \lambda_a} \; -\sum_{k \in \lbrace 1, X, 2 \rbrace} m_k \log P_k
- m_{>2.5} \log P_{>2.5} - (1 - m_{>2.5}) \log (1 - P_{>2.5})
$$

with the scores of an independent Poisson model with the Dixon-Coles correction of
the low scores (0-0, 1-0, 0-1, 1-1),

$$
P(x, y) \propto \tau_\rho(x, y)\, \frac{\lambda_h^x e^{-\lambda_h}}{x!}\, \frac{\lambda_a^y e^{-\lambda_a}}{y!},
\qquad \rho = -0.05
$$

($\rho$ chosen on 2010-11 to 2014-15, the seasons before the backtest, by the log-loss of
the 16 categories). The score matrix gives the probability of each pair of categories;
the pick is the most likely one.

**Backtest** (every La Liga / Segunda match with both odds, 2015-16 to 2025-26, 9,241
matches): v2P hits **15.5%** of the Plenos against 14.5% for v1P, better in all 11
seasons, with a log-loss of the 16 categories of 2.525.

**Against the public.** The Pleno al 15 only pays with 14 hits, and then it is shared
among the bets that hit it too. The betting page has the public's share of each
category of each team, $s^h_c$ and $s^a_c$; with the bets picking each team's category
independently, a pair $(c_h, c_a)$ is marked by $s^h_{c_h} s^a_{c_a}$ of them. The
**best value** pick maximises $P(c_h, c_a) / (s^h_{c_h} s^a_{c_a})$: the most expected
Pleno prize for the 14-hit tickets, a less likely but less shared score
(``value_pick``).
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quiniela.prediction.market import MarketModel

# Goals categories of each team (M: 3 or more)
PLENO_CATEGORIES = ["0", "1", "2", "M"]

# Dixon-Coles correction of the low scores (chosen on 2010-11 to 2014-15)
RHO = -0.05

# Largest number of goals of a team in the score matrix
MAX_GOALS = 10

# Over / under 2.5 goals odds, in order of preference (quiniela.data.loading.TOTALS_COLUMNS)
TOTALS_PREFIXES = ["Avg", "BbAv"]

_GOALS = np.arange(MAX_GOALS + 1)
_CATEGORY = np.minimum(_GOALS, 3)
_HOME = _GOALS[:, None] > _GOALS[None, :]
_DRAW = _GOALS[:, None] == _GOALS[None, :]
_OVER = (_GOALS[:, None] + _GOALS[None, :]) > 2.5
_FACTORIAL = np.array([math.factorial(g) for g in _GOALS], dtype=float)


def over_probability(matches: pd.DataFrame) -> np.ndarray:
    """Market probability of more than 2.5 goals (margin removed), ``NaN`` without odds.

    Args:
        matches: Matches with ``Avg>2.5`` / ``Avg<2.5`` or ``BbAv>2.5`` / ``BbAv<2.5``.

    Returns:
        One probability per match.
    """
    out = np.full(len(matches), np.nan)
    for prefix in TOTALS_PREFIXES:
        columns = [f"{prefix}>2.5", f"{prefix}<2.5"]
        if not set(columns) <= set(matches.columns):
            continue
        odds = matches[columns].to_numpy(dtype=float)
        usable = np.isnan(out) & np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
        inverse = 1 / odds[usable]
        out[usable] = inverse[:, 0] / inverse.sum(axis=1)
    return out


def score_matrix(home: float, away: float, rho: float = RHO) -> np.ndarray:
    """Probability of every score up to ``MAX_GOALS`` goals per team.

    Args:
        home: Expected goals of the home team.
        away: Expected goals of the away team.
        rho: Dixon-Coles correction of the low scores.

    Returns:
        Shape ``(MAX_GOALS + 1, MAX_GOALS + 1)``: home goals by away goals, adding up to 1.
    """
    home_goals = np.exp(-home) * home**_GOALS / _FACTORIAL
    away_goals = np.exp(-away) * away**_GOALS / _FACTORIAL
    matrix = np.outer(home_goals, away_goals)
    matrix[0, 0] *= 1 - home * away * rho
    matrix[0, 1] *= 1 + home * rho
    matrix[1, 0] *= 1 + away * rho
    matrix[1, 1] *= 1 - rho
    matrix = np.clip(matrix, 1e-12, None)
    normalised: np.ndarray = matrix / matrix.sum()
    return normalised


def fit_goals(outcome: np.ndarray, over: float, rho: float = RHO) -> tuple[float, float]:
    """Expected goals whose 1X2 and over 2.5 probabilities match the market's.

    Args:
        outcome: Market probabilities of 1, X and 2.
        over: Market probability of more than 2.5 goals.
        rho: Dixon-Coles correction of the low scores.

    Returns:
        ``(home, away)`` expected goals.
    """
    target = np.array([*outcome, over, 1 - over])

    def loss(logs: np.ndarray) -> float:
        matrix = score_matrix(math.exp(logs[0]), math.exp(logs[1]), rho)
        home, draw, over_model = matrix[_HOME].sum(), matrix[_DRAW].sum(), matrix[_OVER].sum()
        model = np.array([home, draw, 1 - home - draw, over_model, 1 - over_model])
        return float(-(target * np.log(np.clip(model, 1e-9, 1.0))).sum())

    fit = minimize(
        loss,
        np.log([1.4, 1.1]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-8},
    )
    return math.exp(fit.x[0]), math.exp(fit.x[1])


def category_matrix(home: float, away: float, rho: float = RHO) -> np.ndarray:
    """Probability of every pair of goals categories (0, 1, 2, M of each team).

    Args:
        home: Expected goals of the home team.
        away: Expected goals of the away team.
        rho: Dixon-Coles correction of the low scores.

    Returns:
        Shape ``(4, 4)``: home category by away category, adding up to 1.
    """
    matrix = score_matrix(home, away, rho)
    out = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            out[i, j] = matrix[np.ix_(_CATEGORY == i, _CATEGORY == j)].sum()
    return out


def market_pleno(matches: pd.DataFrame, rho: float = RHO) -> pd.DataFrame:
    """Pleno al 15 pick of every match with 1X2 and over / under 2.5 odds (v2P).

    Args:
        matches: Matches with the 1X2 odds (``quiniela.data.loading.ODDS_COLUMNS``)
            and the over / under 2.5 odds (``TOTALS_COLUMNS``).
        rho: Dixon-Coles correction of the low scores.

    Returns:
        Indexed like ``matches``: ``pleno_home``, ``pleno_away`` (the most likely
        categories), ``pleno_prob`` (its probability), ``goals_home`` and
        ``goals_away`` (the market's expected goals) and ``categories`` (the 4x4
        probabilities as nested lists); ``NaN`` / ``None`` without both odds.
    """
    outcome = MarketModel().fit(matches).predict_proba(matches)
    over = over_probability(matches)
    rows: list[dict[str, object]] = []
    for probs, total in zip(outcome, over, strict=True):
        if np.isnan(probs).any() or np.isnan(total):
            rows.append({"pleno_home": None, "pleno_away": None, "pleno_prob": np.nan,
                         "goals_home": np.nan, "goals_away": np.nan, "categories": None})  # fmt: skip
            continue
        home, away = fit_goals(probs, float(total), rho)
        cats = category_matrix(home, away, rho)
        i, j = np.unravel_index(int(cats.argmax()), cats.shape)
        rows.append(
            {
                "pleno_home": PLENO_CATEGORIES[i],
                "pleno_away": PLENO_CATEGORIES[j],
                "pleno_prob": float(cats[i, j]),
                "goals_home": home,
                "goals_away": away,
                "categories": cats.round(5).tolist(),
            }
        )
    return pd.DataFrame(rows, index=matches.index)


def value_pick(
    categories: np.ndarray, public_home: dict[str, float], public_away: dict[str, float]
) -> dict[str, object] | None:
    """The pair of categories with the best probability against the public's picks.

    Args:
        categories: Probability of every pair (``category_matrix``), shape ``(4, 4)``.
        public_home: Public's share (percent) of each category of the home team.
        public_away: The same for the away team.

    Returns:
        ``home``, ``away``, ``prob`` (its probability), ``public`` (share of the
        bets marking that pair, if they pick each team independently) and
        ``ratio`` (``prob / public``); ``None`` without the public's shares.
    """
    shares_home = np.array([float(public_home.get(c, 0) or 0) for c in PLENO_CATEGORIES])
    shares_away = np.array([float(public_away.get(c, 0) or 0) for c in PLENO_CATEGORIES])
    if shares_home.sum() <= 0 or shares_away.sum() <= 0:
        return None
    public = np.outer(shares_home / shares_home.sum(), shares_away / shares_away.sum())
    ratio = np.asarray(categories, dtype=float) / np.clip(public, 1e-4, None)
    i, j = np.unravel_index(int(ratio.argmax()), ratio.shape)
    return {
        "home": PLENO_CATEGORIES[i],
        "away": PLENO_CATEGORIES[j],
        "prob": float(categories[i][j]),
        "public": float(public[i, j]),
        "ratio": float(ratio[i, j]),
    }
