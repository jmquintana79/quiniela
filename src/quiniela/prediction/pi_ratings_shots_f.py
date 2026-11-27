r"""Liga F pi-ratings learned from goals and shots on target (v3F).

The v2F pi-ratings (``quiniela.prediction.pi_ratings``) learn from the goal
difference of every match. With about 240 Liga F matches a season, goals are a
noisy measure of how a match went: shots on target say more about it with the
same few matches. v3F learns the same ratings from a mix of both, on the goal
scale:

$$
g' = (1 - w)\,(G_h - G_a) + w\,\kappa\,(T_h - T_a), \qquad
\kappa = \frac{\sum (G_h + G_a)}{\sum (T_h + T_a)}
$$

with $G$ the goals, $T$ the shots on target of each team (ESPN's scoreboard,
``quiniela.io.espn``) and $\kappa$ the goals per shot on target of the training
matches (about 0.32). A match without shots on target learns from its goals.
The ratings, the learning rules and the 1X2 mapping (ordered logistic regression
on the pre-match goal difference) are v2F's.

**Weight.** $w = 0.25$ was the best of 0, 0.25, 0.5, 0.75 and 1 on the weekly
walk-forward log-loss of 2023-24 and 2024-25 (0.8900 against 0.8906 for goals
alone); on 2025-26 and 2026-27, not used to choose it, it scored 0.9047 against
0.9096 for v2F (hit rate 59.4% against 58.2%).
"""

import numpy as np
import pandas as pd

from quiniela.prediction.pi_ratings import PiRatingsModel

# Shots-on-target columns (football-data names, as quiniela.io.espn saves them)
SHOTS_ON_TARGET = ["HST", "AST"]


# A subclass of v2F's model (already a scikit-learn estimator): no type: ignore needed
class PiRatingsShotsModel(PiRatingsModel):
    """Pi-ratings learning from a mix of goal and shots-on-target differences.

    Args:
        shots_weight: Weight $w$ of the shots on target in what the ratings learn
            (0: goals only, the v2F model; 1: shots on target only).
        learning_rate: How much a match moves the ratings ($\\lambda$).
        catch_up: Share of the change passed to the other ground's rating ($\\gamma$).
        base: Base of the logarithmic goal scale ($b$).
        scale: Rating points per order of magnitude of goal difference ($c$).
        calibration_seasons: Last seasons of training matches the ordered
            logistic regression is fitted on.
    """

    def __init__(
        self,
        shots_weight: float = 0.25,
        learning_rate: float = 0.035,
        catch_up: float = 0.9,
        base: float = 10.0,
        scale: float = 3.0,
        calibration_seasons: int = 5,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            catch_up=catch_up,
            base=base,
            scale=scale,
            calibration_seasons=calibration_seasons,
        )
        self.shots_weight = shots_weight

    def _signal(self, X: pd.DataFrame) -> np.ndarray:
        """Mix of the goal and the scaled shots-on-target differences of each match."""
        goals: np.ndarray = super()._signal(X)
        if not set(SHOTS_ON_TARGET) <= set(X.columns):
            self.shots_scale_ = float("nan")
            return goals
        home = X["HST"].to_numpy(dtype=float)
        away = X["AST"].to_numpy(dtype=float)
        known = ~(np.isnan(home) | np.isnan(away))
        shots = (home + away)[known].sum()
        total_goals = (X["FTHG"].to_numpy(dtype=float) + X["FTAG"].to_numpy(dtype=float))[known]
        # Goals per shot on target, so the shots difference is on the goal scale
        self.shots_scale_ = float(np.nansum(total_goals) / shots) if shots > 0 else float("nan")
        if not np.isfinite(self.shots_scale_):
            return goals
        mix = (1 - self.shots_weight) * goals + self.shots_weight * self.shots_scale_ * (
            home - away
        )
        signal: np.ndarray = np.where(known, mix, goals)
        return signal
