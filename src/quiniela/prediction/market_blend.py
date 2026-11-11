r"""Dixon-Coles + market (v3): a learned combination of both forecasts.

Dixon-Coles (v2) only sees past goals; the bookmakers' odds also reflect
injuries, line-ups, news and the money of many bettors. A **log-linear
pool** (a multinomial logistic regression with shared weights) learns how
much to trust each one:

$$
z_k = a \log p^{\mathrm{DC}}_k + b \log p^{\mathrm{M}}_k + c_k,
\qquad
P(k) = \frac{e^{z_k}}{\sum_{k'} e^{z_{k'}}},
\qquad k \in \lbrace H, D, A \rbrace
$$

with $p^{\mathrm{DC}}$ the Dixon-Coles probabilities, $p^{\mathrm{M}}$ the
market's (implied by the pre-match odds, margin removed), $a$ and $b$ the
weights of each forecast and $c_k$ an intercept per outcome ($c_H = 0$)
that corrects a systematic bias, e.g. towards draws. Equivalently
$P(k) \propto (p^{\mathrm{DC}}_k)^a \, (p^{\mathrm{M}}_k)^b \, e^{c_k}$. The
four parameters maximise the log-likelihood of the training results.

In the backtest the pool did better than a full multinomial regression on
the six log-probabilities (21 parameters) and than one combination per
league: with so few parameters it does not overfit the noise of a few
thousand matches.

**Out-of-sample training.** The pool must learn from forecasts made the way
they are made in a real matchday: a Dixon-Coles fitted on its own training
matches is over-confident about them. So, inside ``fit``, Dixon-Coles is run
walk-forward over the last ``blend_days`` days of the training data:
refitted every ``refit_days`` days on the matches before that block and used
to predict the block. Those out-of-sample probabilities and the odds of the
same matches train the pool. A block's Dixon-Coles only depends on the
matches before it, so the fitted models are cached and reused by later fits
in the same process (every matchday of a backtest).

**Without odds.** Matches without odds (e.g. fixtures whose odds are not
published yet) get the Dixon-Coles probabilities unchanged, exactly v2.
"""

from collections import OrderedDict
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from quiniela.prediction.dixon_coles import DixonColesModel
from quiniela.prediction.market import MarketModel

OUTCOMES = ["H", "D", "A"]

# Columns that identify a training match for the cache fingerprint
FINGERPRINT_COLUMNS = ["league", "date", "home_team", "away_team", "FTHG", "FTAG"]

# Smallest probability inside the logarithms
PROB_FLOOR = 1e-6

# Fewest matches with odds needed to fit the pool (otherwise Dixon-Coles alone)
MIN_BLEND_MATCHES = 100

# A Monday: blocks start on fixed weekdays, so they are the same in every fit
BLOCK_EPOCH = pd.Timestamp("1970-01-05")

# Dixon-Coles fitted per block, shared by every fit of the process
_CACHE_SIZE = 2048
_dixon_coles_cache: OrderedDict[tuple[object, ...], DixonColesModel] = OrderedDict()


def _log(probs: np.ndarray) -> np.ndarray:
    """Logarithm of probabilities, floored to avoid ``log(0)``."""
    logs: np.ndarray = np.log(np.clip(probs, PROB_FLOOR, 1.0))
    return logs


def _pool(params: np.ndarray, log_dc: np.ndarray, log_market: np.ndarray) -> np.ndarray:
    """Probabilities of the log-linear pool, ``params = (a, b, c_D, c_A)``."""
    z = params[0] * log_dc + params[1] * log_market + np.array([0.0, params[2], params[3]])
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs: np.ndarray = exp / exp.sum(axis=1, keepdims=True)
    return probs


def fit_pool(dixon_coles: np.ndarray, market: np.ndarray, y: pd.Series | np.ndarray) -> np.ndarray:
    """Maximum-likelihood parameters of the log-linear pool.

    Args:
        dixon_coles: Dixon-Coles probabilities, shape ``(n, 3)`` (``H``, ``D``, ``A``).
        market: Market probabilities of the same matches.
        y: Real results (``H``, ``D``, ``A``).

    Returns:
        ``(a, b, c_D, c_A)``: weights of Dixon-Coles and the market and the
        draw and away intercepts.
    """
    log_dc, log_market = _log(dixon_coles), _log(market)
    labels = np.asarray(y)
    observed = np.stack([labels == outcome for outcome in OUTCOMES], axis=1).astype(float)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        probs = _pool(params, log_dc, log_market)
        loss = -float((observed * np.log(np.clip(probs, PROB_FLOOR, 1.0))).sum())
        # Derivative of the softmax log-likelihood w.r.t. z is (P - O)
        residual = probs - observed
        grad = np.array(
            [
                (residual * log_dc).sum(),
                (residual * log_market).sum(),
                residual[:, 1].sum(),
                residual[:, 2].sum(),
            ]
        )
        return loss, grad

    # Start from an even mix of both forecasts
    result = minimize(objective, np.array([0.5, 0.5, 0.0, 0.0]), jac=True, method="L-BFGS-B")
    params: np.ndarray = result.x
    return params


# scikit-learn ships without type hints, so its base classes are typed as Any
class MarketBlendModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Combine Dixon-Coles and market probabilities with a log-linear pool.

    Args:
        blend_days: Days of training data (counted back from the last match)
            whose out-of-sample forecasts train the pool (about five seasons
            by default).
        refit_days: Days between Dixon-Coles refits in the out-of-sample
            walk-forward.
        dixon_coles: Dixon-Coles model to combine (cloned before fitting).
            Defaults to ``DixonColesModel()``.
    """

    def __init__(
        self,
        blend_days: int = 1825,
        refit_days: int = 7,
        dixon_coles: DixonColesModel | None = None,
    ) -> None:
        self.blend_days = blend_days
        self.refit_days = refit_days
        self.dixon_coles = dixon_coles

    def _base(self) -> DixonColesModel:
        """New, unfitted Dixon-Coles with the configured parameters."""
        base: DixonColesModel = (
            DixonColesModel() if self.dixon_coles is None else clone(self.dixon_coles)
        )
        return base

    # %% Fitting

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit Dixon-Coles on every match and the pool on out-of-sample forecasts.

        Args:
            X: Played matches with ``league``, ``date``, ``home_team``,
                ``away_team``, the goals ``FTHG`` and ``FTAG``, the result
                ``FTR`` (unless ``y`` is given) and the odds columns.
            y: Full-time results (``H``, ``D``, ``A``). Defaults to ``X["FTR"]``.

        Returns:
            The fitted model. ``weights_`` holds ``(a, b, c_D, c_A)``, or is
            ``None`` when there were too few matches with odds (Dixon-Coles
            alone is then used).
        """
        self.classes_ = np.array(OUTCOMES)
        self.dixon_coles_ = self._base().fit(X)
        self.market_ = MarketModel().fit(X)

        results = (X["FTR"] if y is None else pd.Series(np.asarray(y), index=X.index)).rename("FTR")
        matches = X.assign(FTR=results).dropna(subset=["FTHG", "FTAG", "FTR"])
        dixon_coles = self.out_of_sample(matches)
        market = self.market_.predict_proba(matches)
        usable = ~np.isnan(dixon_coles).any(axis=1) & ~np.isnan(market).any(axis=1)
        labels = matches["FTR"].to_numpy()[usable]

        self.n_blend_matches_ = int(usable.sum())
        self.weights_: np.ndarray | None = None
        if self.n_blend_matches_ >= MIN_BLEND_MATCHES and len(set(labels)) == len(OUTCOMES):
            self.weights_ = fit_pool(dixon_coles[usable], market[usable], labels)
        return self

    def out_of_sample(self, matches: pd.DataFrame) -> np.ndarray:
        """Walk-forward Dixon-Coles probabilities of the last ``blend_days`` days.

        Each block of ``refit_days`` days is predicted by a Dixon-Coles fitted
        only on the matches before the block.

        Args:
            matches: Played matches with goals.

        Returns:
            Array of shape ``(n_matches, 3)`` aligned with ``matches``; ``NaN``
            for matches outside the window or without earlier matches to train on.
        """
        probs = np.full((len(matches), len(OUTCOMES)), np.nan)
        if matches.empty:
            return probs
        dates = matches["date"]
        block = ((dates - BLOCK_EPOCH).dt.days // self.refit_days).to_numpy()
        first = (dates.max() - pd.Timedelta(days=self.blend_days) - BLOCK_EPOCH).days
        first_block = first // self.refit_days

        # Order-independent fingerprint of the matches before each block: the
        # same data always gives the same Dixon-Coles, so it can be cached
        hashes = pd.util.hash_pandas_object(matches[FINGERPRINT_COLUMNS], index=False).to_numpy()
        params = tuple(sorted(self._base().get_params().items()))

        for b in np.unique(block[block >= first_block]):
            before = block < b
            if not before.any():
                continue
            key = (b, self.refit_days, int(before.sum()), int(hashes[before].sum()), params)
            model = _dixon_coles_cache.get(key)
            if model is None:
                model = self._base().fit(matches[before])
                _dixon_coles_cache[key] = model
                if len(_dixon_coles_cache) > _CACHE_SIZE:
                    _dixon_coles_cache.popitem(last=False)
            else:
                _dixon_coles_cache.move_to_end(key)
            in_block = block == b
            probs[in_block] = model.predict_proba(matches[in_block])
        return probs

    # %% Predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of a home win, a draw and an away win.

        Args:
            X: Matches with ``league``, ``home_team``, ``away_team`` and the
                odds columns.

        Returns:
            Array of shape ``(n_matches, 3)`` in ``H``, ``D``, ``A`` order: the
            pool for matches with odds, Dixon-Coles for the rest.
        """
        check_is_fitted(self, "dixon_coles_")
        probs = self.dixon_coles_.predict_proba(X)
        market = self.market_.predict_proba(X)
        with_odds = ~np.isnan(market).any(axis=1)
        if self.weights_ is not None and with_odds.any():
            probs[with_odds] = _pool(self.weights_, _log(probs[with_odds]), _log(market[with_odds]))
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely result (``H``, ``D`` or ``A``) of each match.

        Args:
            X: Matches with ``league``, ``home_team``, ``away_team`` and the
                odds columns.

        Returns:
            Array of results.
        """
        best = self.predict_proba(X).argmax(axis=1)
        predicted: np.ndarray = self.classes_[best]
        return predicted
