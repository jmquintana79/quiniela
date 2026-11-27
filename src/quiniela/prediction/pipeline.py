"""Walk-forward predictions of matchdays, for the current season and the backtest.

For each matchday a model is trained only on matches played **before** it
(every earlier season plus the earlier matches of the same season), so a
prediction never uses information that was unknown at the time. The same
:func:`walk_forward` routine predicts the current season and past seasons
(``quiniela.evaluation.backtest``).

Models receive the whole match table as ``X``: league, date, teams, odds and,
for the training matches, goals. The goals and result of the matches being
predicted are removed before calling ``predict``.

Current-season predictions are stored in ``data/predictions/<model>.parquet``
with one row per match, keyed by ``league`` and ``game``. Predicting a single
matchday replaces that matchday's rows and keeps the rest.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quiniela.config import CURRENT_DIR, HISTORICAL_DIR, PREDICTIONS_DIR
from quiniela.data.loading import load_current, load_historical
from quiniela.data.matchdays import assign_matchdays
from quiniela.prediction.confidence import CONFIDENCE_COLUMNS, confidence
from quiniela.prediction.planner import PLANNER_MODEL
from quiniela.prediction.registry import (
    CLUB_PLANNER_MODEL,
    MODELS,
    NATIONAL_PLANNER_MODEL,
    WOMEN_PLANNER_MODEL,
)

# The Liga F model of the public's picks (v1F)
PUBLIC_PICKS_MODEL = "public_picks"

if TYPE_CHECKING:
    from quiniela.data.prizes import UpcomingTicket

logger = logging.getLogger(__name__)

# Outcome order of every probability array
OUTCOMES = ["H", "D", "A"]

# Columns of the match being predicted that a model must not see
RESULT_COLUMNS = ["FTHG", "FTAG", "FTR"]

# Exact score and Pleno al 15 picks (only for models with ``predict_score``)
SCORE_COLUMNS = ["score_home", "score_away", "pleno_home", "pleno_away"]

# Columns stored for every prediction
PREDICTION_COLUMNS = [
    "league",
    "season",
    "game",
    "matchday",
    "date",
    "home_team",
    "away_team",
    "prediction",
    "p_H",
    "p_D",
    "p_A",
    *CONFIDENCE_COLUMNS,
    *SCORE_COLUMNS,
]

# Value accepted wherever a model key is expected, meaning every registered model
ALL_MODELS = "all"


def resolve_models(model_key: str) -> list[str]:
    """Expand a model key (or ``"all"``) into registered model keys.

    Args:
        model_key: A key of ``MODELS`` or ``"all"``.

    Returns:
        The selected keys, in registry order.

    Raises:
        KeyError: If the key is not registered.
    """
    if model_key == ALL_MODELS:
        return list(MODELS)
    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Available: {sorted(MODELS)} or 'all'")
    return [model_key]


def current_matchdays(current_dir: Path = CURRENT_DIR) -> pd.DataFrame:
    """Load the current season with its league rounds and quiniela matchdays.

    Args:
        current_dir: Folder with the current season files.

    Returns:
        Current matches (played and upcoming) with ``round`` and ``matchday`` columns.
    """
    matches = load_current(current_dir)
    return matches.join(assign_matchdays(matches))


def walk_forward(model_key: str, target: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """Predict the matchdays in ``target`` one by one, training on earlier matches.

    Args:
        model_key: Key of the model in ``MODELS``.
        target: Matches to predict, with a ``matchday`` column.
        played: Pool of played matches to train on; only the ones before each
            matchday's first day are used.

    Returns:
        One row per target match with ``PREDICTION_COLUMNS``. Matchdays
        without any earlier match to train on get no prediction (empty values).
    """
    spec = MODELS[model_key]
    predictions = []
    for matchday, matches in target.groupby("matchday", sort=True):
        # Train only on matches finished before the matchday starts
        cutoff = matches["date"].min().normalize()
        train = played[played["date"] < cutoff]
        out = matches.assign(prediction=None).reindex(
            columns=list(dict.fromkeys([*matches.columns, *PREDICTION_COLUMNS]))
        )
        if train.empty:
            predictions.append(out[PREDICTION_COLUMNS])
            continue

        model = spec.factory()
        model.fit(train, train["FTR"])
        X = matches.drop(columns=RESULT_COLUMNS, errors="ignore")
        probs = np.asarray(model.predict_proba(X), dtype=float)
        out[["p_H", "p_D", "p_A"]] = probs
        out["prediction"] = model.predict(X)
        # How predictable each match is, from the probabilities alone (any model)
        out[CONFIDENCE_COLUMNS] = confidence(probs, out["prediction"]).to_numpy()
        if hasattr(model, "predict_score"):
            out[SCORE_COLUMNS] = model.predict_score(X)[SCORE_COLUMNS].to_numpy()
        predictions.append(out[PREDICTION_COLUMNS])
        logger.debug(
            "%s matchday %s: %d matches predicted (trained on %d matches)",
            model_key,
            matchday,
            len(matches),
            len(train),
        )
    if not predictions:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(predictions, ignore_index=True)


def predict_matchdays(
    model_key: str,
    matchdays: list[int] | None = None,
    history: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict the given matchdays of the current season.

    Args:
        model_key: Key of the model in ``MODELS``.
        matchdays: Matchday numbers. ``None`` predicts only the last matchday.
        history: Historical matches. Defaults to the downloaded seasons.
        current: Current season with a ``matchday`` column. Defaults to the
            downloaded current season.

    Returns:
        One row per match with ``PREDICTION_COLUMNS``.

    Raises:
        KeyError: If the model is not registered.
        ValueError: If there is no current data or a matchday does not exist.
    """
    resolve_models(model_key)
    history = load_historical(HISTORICAL_DIR) if history is None else history
    current = current_matchdays() if current is None else current
    if current.empty:
        raise ValueError("No current season data. Run `invoke download.current` first.")

    available = sorted(current["matchday"].unique())
    selected = [available[-1]] if matchdays is None else sorted(set(matchdays))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Matchdays {missing} not found. Available: 1-{available[-1]}")

    played = pd.concat([history, current], ignore_index=True).dropna(subset=["FTR"])
    target = current[current["matchday"].isin(selected)]
    predictions = walk_forward(model_key, target, played)
    logger.info("%s: %d matches predicted (matchdays %s)", model_key, len(predictions), selected)
    return predictions


def save_predictions(
    predictions: pd.DataFrame,
    model_key: str,
    replace_all: bool = False,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> Path:
    """Store predictions, merging them with the ones already saved.

    Args:
        predictions: Output of :func:`predict_matchdays`.
        model_key: Model the predictions come from (file name).
        replace_all: Overwrite the file instead of merging.
        predictions_dir: Folder where prediction files are kept.

    Returns:
        Path of the predictions file.
    """
    path = predictions_dir / f"{model_key}.parquet"
    if path.exists() and not replace_all:
        previous = pd.read_parquet(path)
        # New predictions win for the matches they cover
        keys = pd.MultiIndex.from_frame(predictions[["league", "game"]])
        keep = ~pd.MultiIndex.from_frame(previous[["league", "game"]]).isin(keys)
        predictions = pd.concat([previous[keep], predictions], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.sort_values(["date", "league"]).to_parquet(path, index=False)
    logger.info("Saved %d predictions to %s", len(predictions), path)
    return path


def load_predictions(predictions_dir: Path = PREDICTIONS_DIR) -> dict[str, pd.DataFrame]:
    """Load the stored predictions of every registered model.

    Args:
        predictions_dir: Folder where prediction files are kept.

    Returns:
        Mapping from model key to its predictions (models without a file are omitted).
    """
    return {
        key: pd.read_parquet(predictions_dir / f"{key}.parquet")
        for key in MODELS
        if (predictions_dir / f"{key}.parquet").exists()
    }


def run(
    model_key: str = ALL_MODELS,
    matchday: int | None = None,
    all_matchdays: bool = False,
) -> list[Path]:
    """Predict and save: one matchday (default: the last one) or all of them.

    Args:
        model_key: Key of the model in ``MODELS``, or ``"all"`` for every model.
        matchday: Matchday to predict. Ignored when ``all_matchdays`` is set.
        all_matchdays: Predict every matchday of the current season.

    Returns:
        Paths of the predictions files, one per model.
    """
    current = current_matchdays()
    history = load_historical(HISTORICAL_DIR)
    if all_matchdays:
        selected: list[int] | None = sorted(current["matchday"].unique())
    else:
        selected = None if matchday is None else [matchday]
    paths = []
    for key in resolve_models(model_key):
        predictions = predict_matchdays(key, selected, history=history, current=current)
        paths.append(save_predictions(predictions, key, replace_all=all_matchdays))
    return paths


# League codes of the ticket matches the models cannot predict
WOMEN_LEAGUE = "WOMEN"
OTHER_LEAGUE = "OTHER"

# Days after sales close during which a result found for a ticket pairing belongs to it
RESULT_WINDOW = pd.Timedelta(days=4)


def predict_upcoming(
    upcoming: "UpcomingTicket",
    history: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Predict the La Liga and Segunda matches of the next official ticket.

    The official ticket (saved from the betting page) is known days before
    football-data publishes the fixtures and their odds. Its La Liga and
    Segunda matches are predicted with every model trained on all played
    matches; without odds the market reference has no prediction and v3 is
    Dixon-Coles alone. The Liga F matches get the Liga F models (v3F for the
    tickets, ``WOMEN_PLANNER_MODEL``), the national-team matches v1N
    (``NATIONAL_PLANNER_MODEL``) and the other club matches (European cups,
    Copa del Rey, foreign leagues) v1C (``CLUB_PLANNER_MODEL``); the rest get
    the public's picks.

    Args:
        upcoming: The next ticket (``quiniela.data.prizes.load_upcoming``).
        history: Historical matches. Defaults to the downloaded seasons.
        current: Current season matches. Defaults to the downloaded files.

    Returns:
        ``(matches, predictions, info)``: the 14 ticket matches as a matchday
        (``official_position`` gives the ticket order; ``league`` is ``SP1``,
        ``SP2``, ``WOMEN`` or ``OTHER``; results when football-data has them),
        the predictions of every model for the La Liga / Segunda ones, and
        ``info`` with how many are ``ours``, have a football-data ``fixture``,
        ``odds`` and ``results``, how many use the ``public``'s picks and how
        many are ``national``-team matches predicted by v1N and ``clubs``
        matches predicted by v1C, and the
        ``pleno`` al 15 pick (``_pleno_pick``).
    """
    from quiniela.data.prizes import team_names, translate_team
    from quiniela.prediction.market import MarketModel

    history = load_historical(HISTORICAL_DIR) if history is None else history
    current = load_current(CURRENT_DIR) if current is None else current
    played = pd.concat([history, current], ignore_index=True).dropna(subset=["FTR"])
    season = str(current["season"].iloc[-1]) if not current.empty else ""
    names = team_names(current).get(season, {})
    # League of every team: the one of its latest current-season match
    by_date = current.sort_values("date")
    league_of = {
        **dict(zip(by_date["away_team"], by_date["league"], strict=True)),
        **dict(zip(by_date["home_team"], by_date["league"], strict=True)),
    }
    closes = upcoming.closes.tz_localize(None)
    pending = current[current["FTR"].isna()]
    recent = current[current["FTR"].notna() & (current["date"] >= closes - RESULT_WINDOW)]

    liga_f_positions = _liga_f_positions(upcoming, closes)
    rows = []
    for match in upcoming.matches[upcoming.matches["position"] <= 14].itertuples():
        home = translate_team(names, match.lae_home)
        away = translate_team(names, match.lae_away)
        league = league_of.get(home) if home and away else None
        row: dict[str, object] = {
            "position": match.position,
            "home_team": home if league else _plain(match.lae_home),
            "away_team": away if league else _plain(match.lae_away),
            "league": league if league and league == league_of.get(away) else None,
        }
        if row["league"] is None:
            women = "(f)" in f"{match.lae_home} {match.lae_away}".lower()
            # Only the Liga F (an ESPN Liga F fixture) is WOMEN; other women's competitions
            # (e.g. the Champions League) are OTHER, like the men's national teams
            liga_f = women and (liga_f_positions is None or match.position in liga_f_positions)
            row["league"] = WOMEN_LEAGUE if liga_f else OTHER_LEAGUE
            row["home_team"], row["away_team"] = _plain(match.lae_home), _plain(match.lae_away)
        fixture = _pairing(pending, str(row["home_team"]), str(row["away_team"]))
        result = _pairing(recent, str(row["home_team"]), str(row["away_team"]))
        source = result if not result.empty else fixture
        if row["league"] not in (WOMEN_LEAGUE, OTHER_LEAGUE) and not source.empty:
            row.update(source.iloc[0].to_dict())
            row["position"] = match.position
        row["_fixture"] = not fixture.empty or not result.empty
        rows.append(row)

    matches = pd.DataFrame(rows).reindex(columns=list(dict.fromkeys([*current.columns, *rows[0]])))
    matches["season"] = season
    matches["matchday"] = upcoming.jornada
    # League rounds are not known for the official ticket (and mix competitions)
    matches["round"] = None
    matches["official_position"] = matches.pop("position")
    # The public's share of 1, X and 2 of every match (percent): the winners of the
    # ·B prize model (quiniela.prediction.money.PublicPrizeModel)
    public_shares = upcoming.matches.set_index("position")
    for sign, outcome in (("1", "H"), ("X", "D"), ("2", "A")):
        matches[f"_public_{outcome}"] = (
            matches["official_position"].map(public_shares[f"public_{sign}"]).astype(float)
        )
    ours = ~matches["league"].isin([WOMEN_LEAGUE, OTHER_LEAGUE])
    # Without a football-data fixture the match has no date yet: the closing day is used
    known_date = matches["date"].notna()
    matches["date"] = pd.to_datetime(matches["date"]).fillna(closes)
    matches["_kickoff_known"] = known_date
    # Without any football-data fixture the column is all NaN (float): text from here on
    matches["game"] = (
        matches["game"]
        .astype(object)
        .fillna(
            matches["date"].dt.strftime("%Y-%m-%d")
            + " "
            + matches["home_team"]
            + "-"
            + matches["away_team"]
        )
    )
    # League shares of 1, X and 2: the probabilities of matches no model predicts
    shares = played["FTR"].value_counts(normalize=True)
    for outcome in ("H", "D", "A"):
        matches[f"_fill_p_{outcome}"] = float(shares.get(outcome, 1 / 3))

    target = matches[ours]
    predictions = (
        {key: walk_forward(key, target, played) for key in MODELS} if not target.empty else {}
    )
    with_odds = 0
    if not target.empty:
        odds = MarketModel().fit(target).predict_proba(target)
        with_odds = int((~np.isnan(odds).any(axis=1)).sum())
    public = _public_predictions(upcoming, matches[~ours])
    # Liga F matches: the v1F model (public picks); other competitions: the public's picks too
    women = public["game"].isin(matches.loc[matches["league"] == WOMEN_LEAGUE, "game"])
    public.loc[women.to_numpy(), "source"] = "v1F"
    if women.any():
        predictions[PUBLIC_PICKS_MODEL] = public[women.to_numpy()].reset_index(drop=True)
    # v2F, v3F...: the results-based Liga F models (ESPN) on the women's matches of the ticket
    from quiniela.evaluation.women import RESULT_KEYS

    women_matches = matches[matches["league"] == WOMEN_LEAGUE]
    if not women_matches.empty:
        for women_key in RESULT_KEYS:
            rows = _women_results_predictions(upcoming, women_matches, closes, women_key)
            if not rows.empty:
                predictions[women_key] = rows
    # National-team matches: v1N, the Elo of every international (the public without it)
    national = _national_predictions(upcoming, matches[matches["league"] == OTHER_LEAGUE], closes)
    if not national.empty:
        predictions[NATIONAL_PLANNER_MODEL] = national
    # Club matches of other competitions: v1C, the club Elo of ESPN (the public without it)
    others = matches[
        (matches["league"] == OTHER_LEAGUE) & ~matches["game"].isin(national.get("game", []))
    ]
    clubs = _club_predictions(upcoming, others, closes)
    if not clubs.empty:
        predictions[CLUB_PLANNER_MODEL] = clubs
    # The Liga F model of the tickets (v3F); v1F (the public) where it has no prediction
    planner_women = predictions.get(WOMEN_PLANNER_MODEL, pd.DataFrame())
    modelled = [*planner_women.get("game", []), *national.get("game", []), *clubs.get("game", [])]
    # Empty frames are left out: concatenating them would turn the probabilities into objects
    parts = [planner_women, national, clubs, public[~public["game"].isin(modelled)]]
    parts = [part for part in parts if not part.empty]
    fill = pd.concat(parts, ignore_index=True) if parts else public.iloc[:0]
    fill = fill.astype({column: float for column in ("p_H", "p_D", "p_A")})
    if PLANNER_MODEL in predictions:
        # The ticket of v3 + v3F: Dixon-Coles + market where there is data, v3F on the
        # Liga F matches (v1F without it), v1N on the national teams, v1C on the other
        # club matches and the public's picks on the rest
        model_rows = predictions[PLANNER_MODEL].assign(source="model")
        predictions[PLANNER_MODEL] = pd.concat([model_rows, fill], ignore_index=True)
    elif not fill.empty:
        predictions[PLANNER_MODEL] = fill
    info = {
        "ours": int(ours.sum()),
        "fixtures": int(matches.loc[ours, "_fixture"].sum()),
        "odds": with_odds,
        "results": int(matches["FTR"].notna().sum()),
        "public": len(public) - len(national) - len(clubs),
        "national": len(national),
        "clubs": len(clubs),
        "pleno": _pleno_pick(upcoming, names, league_of, pending, recent, played, closes),
    }
    return matches.drop(columns=["_fixture"]), predictions, info


# Pleno al 15 goal categories of each team
PLENO_CATEGORIES = ["0", "1", "2", "M"]


def _pleno_pick(
    upcoming: "UpcomingTicket",
    names: dict[str, str],
    league_of: dict[str, str],
    pending: pd.DataFrame,
    recent: pd.DataFrame,
    played: pd.DataFrame,
    closes: pd.Timestamp,
) -> dict[str, Any] | None:
    """Pick of the Pleno al 15 of the next ticket: goals category (0, 1, 2, M) of each team.

    A La Liga / Segunda match gets the market's most likely pair of categories
    (v2P, ``quiniela.prediction.pleno``: expected goals fitted to its 1X2 and over /
    under 2.5 odds) or, without those odds, Dixon-Coles' (v1P, from its score
    matrix); any other match, the categories the public marks most ("Pronósticos
    más apostados"). The same pick goes on every v3+v3F ticket (the Pleno al 15 is
    a single match: no doubles).

    Returns:
        ``home``, ``away``, ``category`` (M, F, O), ``pick_home``,
        ``pick_away``, ``source`` (``market``, ``model`` or ``public``),
        ``prob`` (the pick's probability, v2P), ``value`` (the best-value pick
        against the public's shares, v2P with the public's picks), the public's
        shares and ``result`` (categories, when football-data has it); or
        ``None`` without a 15th match.
    """
    from quiniela.data.prizes import match_category, translate_team

    fifteen = upcoming.matches[upcoming.matches["position"] == 15]
    if fifteen.empty:
        return None
    match = fifteen.iloc[0]
    home = translate_team(names, match["lae_home"])
    away = translate_team(names, match["lae_away"])
    league = league_of.get(home) if home and away else None
    ours = league is not None and away is not None and league == league_of.get(away)
    out: dict[str, Any] = {
        "home": home if ours else _plain(match["lae_home"]),
        "away": away if ours else _plain(match["lae_away"]),
        "category": match_category(match["lae_home"], match["lae_away"], league if ours else None),
        "public_home": upcoming.pleno_public.get("home") or None,
        "public_away": upcoming.pleno_public.get("away") or None,
        "pick_home": None,
        "pick_away": None,
        "source": None,
        "prob": None,
        "value": None,
        "result": None,
    }
    if ours:
        found = _pairing(recent, str(home), str(away))
        source = found if not found.empty else _pairing(pending, str(home), str(away))
        target = (
            source.iloc[:1].copy()
            if not source.empty
            else pd.DataFrame(
                [{"league": league, "date": closes, "home_team": home, "away_team": away}]
            )
        )
        target["matchday"] = upcoming.jornada
        target["game"] = target.get("game", pd.Series(dtype=object)).fillna(
            f"{closes:%Y-%m-%d} {home}-{away}"
        )
        # v2P: the market's expected goals when the fixture has its 1X2 and over / under odds
        from quiniela.prediction.pleno import market_pleno, value_pick

        market = market_pleno(target)
        if market["pleno_home"].notna().iloc[0]:
            pick = market.iloc[0]
            out.update(
                pick_home=pick["pleno_home"],
                pick_away=pick["pleno_away"],
                source="market",
                prob=round(float(pick["pleno_prob"]), 4),
            )
            if out["public_home"] and out["public_away"]:
                out["value"] = value_pick(
                    np.array(pick["categories"]), out["public_home"], out["public_away"]
                )
        dc = (
            walk_forward(
                "dixon_coles", target.drop(columns=RESULT_COLUMNS, errors="ignore"), played
            )
            if out["source"] is None
            else pd.DataFrame()
        )
        if not dc.empty and dc["pleno_home"].notna().any():
            out.update(
                pick_home=dc["pleno_home"].iloc[0],
                pick_away=dc["pleno_away"].iloc[0],
                source="model",
            )
        if not found.empty and found["FTHG"].notna().any():
            goals = [min(int(found["FTHG"].iloc[0]), 3), min(int(found["FTAG"].iloc[0]), 3)]
            out["result"] = [PLENO_CATEGORIES[g] for g in goals]
    if out["source"] is None and out["public_home"] and out["public_away"]:

        def most(shares: dict[str, float]) -> str:
            return max(PLENO_CATEGORIES, key=lambda c: float(shares.get(c, 0)))

        out.update(
            pick_home=most(out["public_home"]), pick_away=most(out["public_away"]), source="public"
        )
    return out


def _liga_f_positions(upcoming: "UpcomingTicket", closes: pd.Timestamp) -> set[int] | None:
    """Positions of the ticket's women's matches that are Liga F matches in ESPN.

    Args:
        upcoming: The next ticket.
        closes: Sales deadline.

    Returns:
        The positions, or ``None`` without ESPN data (every women's match then
        counts as Liga F).
    """
    from quiniela.data.women import MATCH_DAYS, espn_candidates
    from quiniela.io.espn import load_liga_f_matches

    espn = load_liga_f_matches()
    if espn.empty:
        return None
    names = set(espn["home_team"]) | set(espn["away_team"])
    near = espn[(espn["date"] - closes).abs() <= pd.Timedelta(days=MATCH_DAYS)]
    positions = set()
    for match in upcoming.matches.itertuples():
        homes = espn_candidates(str(match.lae_home), names)
        aways = espn_candidates(str(match.lae_away), names)
        if (
            homes
            and aways
            and (near["home_team"].isin(homes) & near["away_team"].isin(aways)).any()
        ):
            positions.add(int(match.position))
    return positions


def _women_results_predictions(
    upcoming: "UpcomingTicket",
    matches: pd.DataFrame,
    closes: pd.Timestamp,
    key: str = "pi_ratings_f",
) -> pd.DataFrame:
    """Predictions of a results-based Liga F model for the ticket's Liga F matches.

    Each women's match of the ticket is linked to its ESPN fixture (team
    aliases and dates around the sales deadline) and predicted by the model
    (v2F, v3F...) trained on every Liga F match played.

    Args:
        upcoming: The next ticket (LAE team names by position).
        matches: The ticket's Liga F matches (``official_position``, ``game``).
        closes: Sales deadline.
        key: Liga F model (``WOMEN_MODELS``).

    Returns:
        Predictions with ``PREDICTION_COLUMNS`` and ``source``, the model's
        version (``"v2F"``, ``"v3F"``...).
    """
    from quiniela.prediction.registry import WOMEN_MODELS

    from quiniela.data.women import MATCH_DAYS, espn_candidates
    from quiniela.evaluation.women import upcoming_predictions
    from quiniela.io.espn import load_liga_f_matches

    espn = load_liga_f_matches()
    if espn.empty:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "source"])
    # Trained on the matches before the deadline: the prediction stays pre-match once played
    pending = upcoming_predictions(espn, before=closes - pd.Timedelta(days=MATCH_DAYS), key=key)
    names = set(espn["home_team"]) | set(espn["away_team"])
    lae = upcoming.matches.set_index("position")
    rows = []
    for _, match in matches.iterrows():
        position = int(match["official_position"])
        homes = espn_candidates(str(lae.loc[position, "lae_home"]), names)
        aways = espn_candidates(str(lae.loc[position, "lae_away"]), names)
        near = pending[
            pending["home_team"].isin(homes)
            & pending["away_team"].isin(aways)
            & ((pending["date"] - closes).abs() <= pd.Timedelta(days=MATCH_DAYS))
        ]
        if near.empty:
            continue
        fixture = near.iloc[0]
        rows.append(
            {
                **match[
                    ["league", "season", "game", "matchday", "date", "home_team", "away_team"]
                ].to_dict(),
                "p_H": fixture["p_H"],
                "p_D": fixture["p_D"],
                "p_A": fixture["p_A"],
                "prediction": fixture["prediction"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "source"])
    out = pd.DataFrame(rows)
    probs = out[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
    out[CONFIDENCE_COLUMNS] = confidence(probs, out["prediction"]).to_numpy()
    return out.reindex(columns=PREDICTION_COLUMNS).assign(source=WOMEN_MODELS[key].version)


def _national_predictions(
    upcoming: "UpcomingTicket", matches: pd.DataFrame, closes: pd.Timestamp
) -> pd.DataFrame:
    """v1N predictions of the ticket's national-team matches.

    The model is trained on the internationals played before the sales
    deadline, so a match keeps its pre-match prediction once played. The
    home team of the ticket plays at home unless the results file already has
    the match at a neutral ground (it only does once it is played: before,
    the home advantage is assumed, as in most qualifiers and friendlies).

    Args:
        upcoming: The next ticket (LAE team names by position).
        matches: The ticket matches no men's or Liga F model predicts
            (``official_position``, ``game``...).
        closes: Sales deadline.

    Returns:
        Predictions with ``PREDICTION_COLUMNS`` and ``source`` = ``"v1N"``;
        empty without national-team matches or results.
    """
    from quiniela.data.national import MATCH_DAYS, national_team
    from quiniela.evaluation.national import national_model
    from quiniela.io.international import load_results
    from quiniela.prediction.registry import NATIONAL_MODELS

    empty = pd.DataFrame(columns=[*PREDICTION_COLUMNS, "source"])
    lae = upcoming.matches.set_index("position")
    rows = matches.assign(
        _home=[national_team(lae.loc[int(p), "lae_home"]) for p in matches["official_position"]],
        _away=[national_team(lae.loc[int(p), "lae_away"]) for p in matches["official_position"]],
    ).dropna(subset=["_home", "_away"])
    if rows.empty:
        return empty
    results = load_results()
    model = national_model(results, before=closes - pd.Timedelta(days=MATCH_DAYS))
    if model is None:
        return empty
    near = results[(results["date"] - closes).abs() <= pd.Timedelta(days=MATCH_DAYS)]
    neutral = []
    for home, away in zip(rows["_home"], rows["_away"], strict=True):
        found = near[(near["home_team"] == home) & (near["away_team"] == away)]
        neutral.append(bool(found["neutral"].iloc[0]) if not found.empty else False)
    target = pd.DataFrame(
        {"home_team": rows["_home"], "away_team": rows["_away"], "neutral": neutral}
    )
    probs = model.predict_proba(target)
    out = rows.assign(
        p_H=probs[:, 0],
        p_D=probs[:, 1],
        p_A=probs[:, 2],
        prediction=np.array(OUTCOMES)[probs.argmax(axis=1)],
    )
    out[CONFIDENCE_COLUMNS] = confidence(probs, out["prediction"]).to_numpy()
    out = out.reindex(columns=PREDICTION_COLUMNS)
    return out.assign(source=NATIONAL_MODELS[NATIONAL_PLANNER_MODEL].version)


def _club_predictions(
    upcoming: "UpcomingTicket", matches: pd.DataFrame, closes: pd.Timestamp
) -> pd.DataFrame:
    """v1C predictions of the ticket's club matches of other competitions.

    Each match is linked to its ESPN fixture (``quiniela.data.clubs.link_espn``,
    by the similarity of the names around the sales deadline) and predicted by
    the club Elo trained on the ESPN matches played before the deadline, so it
    keeps its pre-match prediction once played.

    Args:
        upcoming: The next ticket (LAE team names by position).
        matches: The ticket matches of other competitions that are not
            national-team ones (``official_position``, ``game``...).
        closes: Sales deadline.

    Returns:
        Predictions with ``PREDICTION_COLUMNS`` and ``source`` = ``"v1C"``;
        empty without club matches or ESPN data.
    """
    from quiniela.data.clubs import is_club_match, link_espn
    from quiniela.evaluation.clubs import club_model
    from quiniela.io.espn_clubs import load_club_matches
    from quiniela.prediction.registry import CLUB_MODELS

    empty = pd.DataFrame(columns=[*PREDICTION_COLUMNS, "source"])
    lae = upcoming.matches.set_index("position")
    positions = [int(p) for p in matches["official_position"]]
    rows = matches.assign(
        lae_home=[lae.loc[p, "lae_home"] for p in positions],
        lae_away=[lae.loc[p, "lae_away"] for p in positions],
        kickoff=pd.NaT,
        draw_date=closes,
    )
    rows = rows[
        [is_club_match(h, a) for h, a in zip(rows["lae_home"], rows["lae_away"], strict=True)]
    ]
    if rows.empty:
        return empty
    espn = load_club_matches()
    if espn.empty:
        return empty
    links = link_espn(rows, espn)
    rows = rows.join(links[["espn_id", "swapped"]]).dropna(subset=["espn_id"])
    model = club_model(espn, before=closes)
    if rows.empty or model is None:
        return empty
    found = espn.set_index("espn_id").loc[rows["espn_id"]]
    swapped = rows["swapped"].to_numpy(dtype=bool)
    target = pd.DataFrame(
        {
            "home_id": np.where(swapped, found["away_id"], found["home_id"]),
            "away_id": np.where(swapped, found["home_id"], found["away_id"]),
            "neutral": found["neutral"].to_numpy(dtype=bool),
        }
    )
    probs = model.predict_proba(target)
    out = rows.assign(
        p_H=probs[:, 0],
        p_D=probs[:, 1],
        p_A=probs[:, 2],
        prediction=np.array(OUTCOMES)[probs.argmax(axis=1)],
    )
    out[CONFIDENCE_COLUMNS] = confidence(probs, out["prediction"]).to_numpy()
    out = out.reindex(columns=PREDICTION_COLUMNS)
    return out.assign(source=CLUB_MODELS[CLUB_PLANNER_MODEL].version)


def _public_predictions(upcoming: "UpcomingTicket", matches: pd.DataFrame) -> pd.DataFrame:
    """v3 predictions of the ticket matches without data, from the public's picks.

    The matches of other competitions (cups) have no model, and the Liga F
    and national-team ones fall back on them when their model cannot predict;
    the share of bettors marking each sign ("Pronósticos más apostados") is
    the best estimate available, normalised to probabilities.

    Args:
        upcoming: The next ticket, with the public's shares.
        matches: The ticket matches no model predicts (``official_position``).

    Returns:
        Predictions with ``PREDICTION_COLUMNS`` and ``source`` = ``"public"``.
    """
    shares = upcoming.matches.set_index("position")[["public_1", "public_X", "public_2"]]
    rows = matches.join(shares, on="official_position").dropna(
        subset=["public_1", "public_X", "public_2"]
    )
    if rows.empty:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "source"])
    probs = rows[["public_1", "public_X", "public_2"]].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    out = rows.assign(
        p_H=probs[:, 0],
        p_D=probs[:, 1],
        p_A=probs[:, 2],
        prediction=np.array(OUTCOMES)[probs.argmax(axis=1)],
    )
    out[CONFIDENCE_COLUMNS] = confidence(probs, out["prediction"]).to_numpy()
    out = out.reindex(columns=PREDICTION_COLUMNS)
    return out.assign(source="public")


def _pairing(matches: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Matches between ``home`` and ``away`` (at ``home``'s ground)."""
    return matches[(matches["home_team"] == home) & (matches["away_team"] == away)]


def _plain(lae_name: str) -> str:
    """LAE team name without the men's / women's mark (``"Eibar (F)"`` -> ``"Eibar"``)."""
    return (
        lae_name.replace("(M)", "").replace("(F)", "").replace("(m)", "").replace("(f)", "").strip()
    )
