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
        name="Enfrentamientos",
        factory=HeadToHeadModel,
        description="Resultados anteriores entre los mismos equipos en el mismo campo, suavizados "
        "hacia las proporciones de local, empate y visitante de la liga.",
        version="v1",
    ),
    "dixon_coles": ModelSpec(
        name="Dixon-Coles",
        factory=DixonColesModel,
        description="Goles de Poisson a partir del ataque y la defensa de cada equipo (los partidos "
        "recientes pesan más), con el nivel de goles de cada liga, la ventaja de campo y "
        "una corrección de los marcadores bajos.",
        version="v2",
    ),
    "market_blend": ModelSpec(
        name="DC + mercado",
        factory=MarketBlendModel,
        description="Combinación log-lineal de las probabilidades de Dixon-Coles y del mercado, "
        "entrenada con pronósticos fuera de muestra; sin cuotas, Dixon-Coles combinado "
        "con los pi-ratings (v4).",
        version="v3",
    ),
    "pi_ratings": ModelSpec(
        name="Pi-ratings",
        factory=PiRatingsModel,
        description="Valoraciones de cada equipo como local y como visitante, aprendidas partido a "
        "partido de las diferencias de goles (Constantinou y Fenton, 2013) y pasadas a "
        "1X2 con una regresión logística ordenada.",
        version="v4",
    ),
    "market": ModelSpec(
        name="Mercado",
        factory=MarketModel,
        description="Probabilidades que dan las cuotas previas al partido de las casas de apuestas "
        "(sin su margen): la referencia con la que se compara cada modelo.",
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
        name="Público",
        factory=PublicPicksModel,
        description="Liga F: el porcentaje de apostantes de la quiniela que marca cada signo "
        "(«Pronósticos más apostados»), normalizado a probabilidades; el signo más "
        "marcado es el pronóstico.",
        version="v1F",
    ),
    "pi_ratings_f": ModelSpec(
        name="Pi-ratings",
        factory=PiRatingsModel,
        description="Liga F: pi-ratings de local y visitante de cada equipo aprendidos de las "
        "diferencias de goles de todos los partidos de Liga F desde 2022-23 (ESPN) y "
        "pasados a 1X2 con una regresión logística ordenada; el modelo v4 aplicado a la "
        "liga femenina.",
        version="v2F",
    ),
    "pi_ratings_shots_f": ModelSpec(
        name="Pi-ratings + tiros",
        factory=PiRatingsShotsModel,
        description="Liga F: los pi-ratings de v2F aprendidos de una mezcla de la diferencia de goles "
        "(75 %) y de la de tiros a puerta en la escala de goles (25 %) de todos los "
        "partidos de Liga F desde 2022-23 (ESPN): los tiros a puerta miden un partido con "
        "menos ruido que los goles.",
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
        description="Selecciones: valoraciones World Football Elo de todos los partidos "
        "internacionales masculinos desde 1872 (ponderadas por torneo y diferencia de "
        "goles), pasadas a 1X2 con una regresión logística ordenada sobre la diferencia "
        "de Elo y la ventaja de campo.",
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
        name="Elo de clubes",
        factory=ClubEloModel,
        description="Otras competiciones de clubes: una valoración Elo de cada club de 24 ligas y 14 "
        "copas (ESPN, desde 2009), enlazadas por las copas europeas y pasadas a 1X2 con "
        "una regresión logística ordenada sobre la diferencia de Elo y la ventaja de "
        "campo.",
        version="v1C",
    ),
}

# Club model whose probabilities the tickets use for the club matches outside La Liga /
# Segunda (since v0.21; the public's picks before, and still where it has no prediction)
CLUB_PLANNER_MODEL = "elo_clubs"
