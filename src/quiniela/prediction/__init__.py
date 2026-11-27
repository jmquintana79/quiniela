"""Prediction models and the pipeline that applies them to the current season.

Every model follows the scikit-learn estimator API (``fit``, ``predict``,
``predict_proba`` and ``classes_`` in ``H``, ``D``, ``A`` order). To add a
model, implement it in its own module and register it in
``quiniela.prediction.registry.MODELS``.

Submodules:

- ``registry``: model interface and registered models.
- ``head_to_head``: v1, head-to-head benchmark.
- ``dixon_coles``: v2, Dixon-Coles team strengths with time decay.
- ``market``: market reference (bookmakers' implied probabilities).
- ``market_blend``: v3, learned combination of Dixon-Coles and the market.
- ``confidence``: confidence and margin of a prediction, from its probabilities.
- ``pi_ratings``: v4, home and away pi-ratings learned from goal differences.
- ``public_picks``: v1F, the Liga F model (the public's picks).
- ``pi_ratings_shots_f``: v3F, Liga F pi-ratings learned from goals and shots on target.
- ``elo_national``: v1N, World Football Elo ratings of the national teams mapped to 1X2.
- ``elo_clubs``: v1C, one Elo rating of every club of the ESPN competitions mapped to 1X2.
- ``pipeline``: walk-forward predictions of matchdays and their storage.
- ``planner``: the ·D ticket (direct doubles and triples of the quiniela ticket).
- ``reduced``: the ·R ticket (official reductions of the quiniela ticket).
- ``money``: the ·B and ·E tickets (big win and best expected prize in euros).
- ``pleno``: v2P, the Pleno al 15 pick from the market's expected goals.
- ``ticket``: quiniela ticket composition and prize rules.
"""

from quiniela.prediction import (
    confidence,
    dixon_coles,
    elo_clubs,
    elo_national,
    head_to_head,
    market,
    market_blend,
    money,
    pi_ratings,
    pi_ratings_shots_f,
    pipeline,
    planner,
    pleno,
    public_picks,
    reduced,
    registry,
    ticket,
)
from quiniela.prediction.dixon_coles import DixonColesModel
from quiniela.prediction.head_to_head import HeadToHeadModel
from quiniela.prediction.market import MarketModel
from quiniela.prediction.market_blend import MarketBlendModel
from quiniela.prediction.pi_ratings import PiRatingsModel
from quiniela.prediction.public_picks import PublicPicksModel
from quiniela.prediction.registry import DEFAULT_MODEL, MODELS, ModelSpec, Predictor

# Submodules are listed so the API docs (pdoc) include them
__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "DixonColesModel",
    "HeadToHeadModel",
    "MarketBlendModel",
    "MarketModel",
    "PiRatingsModel",
    "PublicPicksModel",
    "ModelSpec",
    "Predictor",
    "confidence",
    "dixon_coles",
    "elo_clubs",
    "elo_national",
    "head_to_head",
    "market",
    "market_blend",
    "money",
    "pi_ratings",
    "pi_ratings_shots_f",
    "pipeline",
    "planner",
    "pleno",
    "public_picks",
    "reduced",
    "registry",
    "ticket",
]
