"""Model registry: the interface every model follows and the available models.

To add a model, implement it in its own module (scikit-learn estimator API)
and register it in ``MODELS``; tasks, the pipeline, the backtest and the
dashboards pick it up by its key. The registry order is the column order in
the dashboards (oldest version first, references last).

Models receive the whole match table as ``X`` (league, date, teams, odds and,
when fitting, goals) and read the columns they need; ``y`` is the full-time
result (``H``, ``D``, ``A``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd

from quiniela.prediction.dixon_coles import DixonColesModel
from quiniela.prediction.elo_clubs import ClubEloModel
from quiniela.prediction.elo_national import NationalEloModel
from quiniela.prediction.head_to_head import HeadToHeadModel
from quiniela.prediction.market import MarketModel
from quiniela.prediction.market_blend import MarketBlendModel
from quiniela.prediction.pi_ratings import PiRatingsModel
from quiniela.prediction.pi_ratings_shots_f import PiRatingsShotsModel
from quiniela.prediction.public_picks import PublicPicksModel


class Predictor(Protocol):
    """Interface shared by every model (scikit-learn classifier API)."""

    classes_: np.ndarray

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Train on played matches."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted result (``H``, ``D`` or ``A``) of each match."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the probability of each result, in ``classes_`` order."""
        ...


@dataclass(frozen=True)
class ModelSpec:
    """A registered model.

    Attributes:
        name: Display name (dashboard column header).
        factory: Builds a new, unfitted model.
        description: One-line summary shown in the dashboard.
        version: Model version (``v1``, ``v2``...); empty for references.
        reference: True for baselines that are not candidate models (the market).
        show_in_matchday: Show the model in the current matchday and season
            dashboards. The market is hidden there (v3 already follows it
            closely) but kept in the evaluation dashboard as the benchmark.
    """

    name: str
    factory: Callable[[], Any]
    description: str
    version: str = ""
    reference: bool = False
    show_in_matchday: bool = True


# Registry: key (used in tasks and file names) -> model
MODELS: dict[str, ModelSpec] = {
    "head_to_head": ModelSpec(
        name="Head-to-head",
        factory=HeadToHeadModel,
        description="Past results between the same teams at the same ground, smoothed "
        "towards the league's home/draw/away shares.",
        version="v1",
    ),
    "dixon_coles": ModelSpec(
        name="Dixon-Coles",
        factory=DixonColesModel,
        description="Poisson goals from time-weighted team attack and defence strengths, "
        "with league goal levels, home advantage and a low-score correction.",
        version="v2",
    ),
    "market_blend": ModelSpec(
        name="DC + Market",
        factory=MarketBlendModel,
        description="Log-linear pool of the Dixon-Coles and market probabilities, "
        "trained on out-of-sample forecasts; Dixon-Coles alone without odds.",
        version="v3",
    ),
    "pi_ratings": ModelSpec(
        name="Pi-ratings",
        factory=PiRatingsModel,
        description="Separate home and away ratings of every team, learned match by match "
        "from goal differences (Constantinou & Fenton, 2013), mapped to 1X2 by an ordered "
        "logistic regression.",
        version="v4",
    ),
    "market": ModelSpec(
        name="Market",
        factory=MarketModel,
        description="Probabilities implied by the bookmakers' pre-match odds (margin "
        "removed): the reference every model is compared against.",
        reference=True,
        show_in_matchday=False,
    ),
}

DEFAULT_MODEL = "head_to_head"

# Liga F (women's league) models: their own version counter with an "F", evaluated
# apart from the men's models (quiniela.evaluation.women) on the Liga F matches of
# the official tickets (quiniela.data.women)
WOMEN_MODELS: dict[str, ModelSpec] = {
    "public_picks": ModelSpec(
        name="Public picks",
        factory=PublicPicksModel,
        description='Liga F: the share of quiniela bettors marking each sign ("Pronósticos más '
        'apostados"), normalised to probabilities; the most marked sign is the prediction.',
        version="v1F",
    ),
    "pi_ratings_f": ModelSpec(
        name="Pi-ratings",
        factory=PiRatingsModel,
        description="Liga F: home and away pi-ratings of every team learned from the goal "
        "differences of every Liga F match since 2022-23 (ESPN), mapped to 1X2 by an ordered "
        "logistic regression; the v4 model applied to the women's league.",
        version="v2F",
    ),
    "pi_ratings_shots_f": ModelSpec(
        name="Pi-ratings + shots",
        factory=PiRatingsShotsModel,
        description="Liga F: the v2F pi-ratings learned from a mix of the goal difference (75%) "
        "and the shots-on-target difference on the goal scale (25%) of every Liga F match since "
        "2022-23 (ESPN): shots on target are a less noisy measure of a match than goals.",
        version="v3F",
    ),
}

# Liga F model whose probabilities the tickets use for the women's matches: v3F (pi-ratings
# of the ESPN goals and shots on target, since v0.18; v2F until v0.17); v1F (the public's
# picks) where it has no prediction
WOMEN_PLANNER_MODEL = "pi_ratings_shots_f"

# National-team models: their own version counter with an "N", trained on every men's
# international since 1872 (quiniela.io.international) and evaluated apart
# (quiniela.evaluation.national) on every international since 2010 and on the
# national-team matches of the official tickets (quiniela.data.national)
NATIONAL_MODELS: dict[str, ModelSpec] = {
    "elo_national": ModelSpec(
        name="Elo",
        factory=NationalEloModel,
        description="National teams: World Football Elo ratings of every men's international "
        "since 1872 (weighted by tournament and goal difference), mapped to 1X2 by an ordered "
        "logistic regression on the Elo difference and home advantage.",
        version="v1N",
    ),
}

# National-team model whose probabilities the tickets use (since v0.20; the public's picks
# before, and still where it has no prediction)
NATIONAL_PLANNER_MODEL = "elo_national"

# Club models of the other competitions (European cups, Copa del Rey, foreign leagues): their
# own version counter with a "C", trained on every ESPN match of quiniela.io.espn_clubs and
# evaluated apart (quiniela.evaluation.clubs) on those matches and on the club matches of the
# official tickets outside La Liga / Segunda (quiniela.data.clubs)
CLUB_MODELS: dict[str, ModelSpec] = {
    "elo_clubs": ModelSpec(
        name="Club Elo",
        factory=ClubEloModel,
        description="Other club competitions: one Elo rating of every club of 24 leagues and 14 "
        "cups (ESPN, since 2009), linked by the European cups, mapped to 1X2 by an ordered "
        "logistic regression on the Elo difference and home advantage.",
        version="v1C",
    ),
}

# Club model whose probabilities the tickets use for the club matches outside La Liga /
# Segunda (since v0.21; the public's picks before, and still where it has no prediction)
CLUB_PLANNER_MODEL = "elo_clubs"
