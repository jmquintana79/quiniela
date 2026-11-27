"""Evaluation of the club models (v1C), apart from the men's, Liga F and national-team ones.

- **Every ESPN club match** from ``FIRST_YEAR`` on (``quiniela.io.espn_clubs``):
  v1C (``quiniela.prediction.elo_clubs``) predicts each match with the
  ratings before it, and its 1X2 regression is refitted at the start of every
  year on the earlier matches. The reference, the **result shares** of the
  matches of the regression window, is scored the same way.
- **The club matches of the official tickets** outside La Liga / Segunda
  (``quiniela.data.clubs``): the same predictions, in the ticket's order of
  the teams; the **public's picks** too on the tickets whose betting page was
  archived. They are summarised like the national teams
  (``quiniela.evaluation.national.summarize_national``).
"""

from typing import Any

import numpy as np
import pandas as pd

from quiniela.evaluation.national import PREDICTION_COLUMNS, summarize_national
from quiniela.prediction.elo_clubs import (
    ENTRY_QUANTILE,
    HOME_ADVANTAGE,
    K_FACTOR,
    WINDOW_YEARS,
    _neutral,
    club_ratings,
)
from quiniela.prediction.elo_national import OUTCOMES, fit_ordinal, ordinal_probabilities
from quiniela.prediction.registry import CLUB_MODELS

ELO_KEY = "elo_clubs"

# First year scored (2009 and 2010 settle the ratings)
FIRST_YEAR = 2011

SHARES_KEY = "club_shares"
PUBLIC_KEY = "club_public"
REFERENCE_SPECS: list[dict[str, Any]] = [
    {
        "key": SHARES_KEY,
        "name": "Result shares",
        "version": "",
        "reference": True,
        "description": "Shares of 1, X and 2 of the club matches of the earlier years; its sign is "
        "always the home win.",
    },
    {
        "key": PUBLIC_KEY,
        "name": "Public picks",
        "version": "",
        "reference": True,
        "description": "Share of quiniela bettors marking each sign, normalised: what the tickets "
        "used for these club matches until v0.20 (only the archived tickets have it).",
    },
]

CLUB_COLUMNS = [*PREDICTION_COLUMNS, "competition", "espn_id"]


def _rows(matches: pd.DataFrame, key: str, probs: np.ndarray) -> pd.DataFrame:
    """Prediction rows of one model for some matches."""
    return matches.assign(
        model=key,
        p_H=probs[:, 0],
        p_D=probs[:, 1],
        p_A=probs[:, 2],
        prediction=np.array(OUTCOMES)[probs.argmax(axis=1)],
    ).reindex(columns=CLUB_COLUMNS)


def club_backtest(espn: pd.DataFrame, first_year: int = FIRST_YEAR) -> pd.DataFrame:
    """Walk-forward predictions of v1C and the result shares on every played ESPN club match.

    Args:
        espn: ESPN club matches (``quiniela.io.espn_clubs.load_club_matches``).
        first_year: First year predicted (earlier matches only train).

    Returns:
        ``CLUB_COLUMNS`` for every played match from ``first_year`` on.
    """
    played = espn.dropna(subset=["FTR"]).sort_values("date", kind="stable").reset_index(drop=True)
    if played.empty:
        return pd.DataFrame(columns=CLUB_COLUMNS)
    before, _ = club_ratings(played, K_FACTOR, HOME_ADVANTAGE, ENTRY_QUANTILE)
    difference = (before[:, 0] - before[:, 1]) / 400
    home = (~_neutral(played)).astype(float)
    codes = played["FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy(dtype=int)
    years = played["date"].dt.year.to_numpy()
    matches = played.assign(
        year=years,
        game=played["espn_id"],
        tournament=played["competition"],
    )
    frames = []
    for year in range(first_year, int(years.max()) + 1):
        target = years == year
        train = (years < year) & (years >= year - WINDOW_YEARS)
        if not target.any() or not train.any():
            continue
        coef = fit_ordinal(difference[train], home[train], codes[train])
        frames.append(
            _rows(
                matches[target],
                ELO_KEY,
                ordinal_probabilities(coef, difference[target], home[target]),
            )
        )
        shares = np.bincount(codes[train], minlength=3) / train.sum()
        frames.append(_rows(matches[target], SHARES_KEY, np.tile(shares, (target.sum(), 1))))
    if not frames:
        return pd.DataFrame(columns=CLUB_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def ticket_predictions(
    linked: pd.DataFrame,
    backtest: pd.DataFrame,
    public: dict[tuple[str, int], np.ndarray] | None = None,
) -> pd.DataFrame:
    """The backtest predictions of the club matches of the official tickets.

    Args:
        linked: Ticket club matches with their ESPN match (``quiniela.data.clubs.ticket_clubs``).
        backtest: Output of :func:`club_backtest`.
        public: Public's shares of every archived ticket (``load_public_shares``).

    Returns:
        ``CLUB_COLUMNS`` plus ``season``, ``jornada`` and ``position``, in the
        ticket's order of the teams (1 and 2 swapped when ESPN lists them the
        other way round), with the ticket's result, and the public's picks as
        ``PUBLIC_KEY`` where the ticket was archived.
    """
    columns = [*CLUB_COLUMNS, "season", "jornada", "position"]
    if linked.empty or backtest.empty:
        return pd.DataFrame(columns=columns)
    tickets = linked[["season", "jornada", "position", "espn_id", "swapped", "FTR", "game", "date"]]
    tickets = tickets.assign(
        home_team=linked["lae_home"],
        away_team=linked["lae_away"],
        competition=linked["competition"],
    )
    predicted = backtest.drop(columns=["FTR", "game", "home_team", "away_team", "date"])
    joined = tickets.merge(predicted.drop(columns=["competition"]), on="espn_id", how="inner")
    frames = []
    if not joined.empty:
        probs = joined[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
        swapped = joined["swapped"].to_numpy(dtype=bool)
        probs[swapped] = probs[swapped][:, ::-1]
        joined = joined.assign(p_H=probs[:, 0], p_D=probs[:, 1], p_A=probs[:, 2])
        for key, rows in joined.groupby("model"):
            out = _rows(rows, str(key), rows[["p_H", "p_D", "p_A"]].to_numpy(dtype=float))
            frames.append(
                out.assign(**{c: rows[c].to_numpy() for c in ("season", "jornada", "position")})
            )
    for record in linked.to_dict("records") if public else []:
        shares = public.get((str(record["season"]), int(record["jornada"]))) if public else None
        if shares is None or pd.isna(record["FTR"]):
            continue
        probs = np.asarray(shares[int(record["position"]) - 1], dtype=float)[None, :]
        row = pd.DataFrame([record]).assign(
            year=pd.Timestamp(record["date"]).year,
            home_team=record["lae_home"],
            away_team=record["lae_away"],
        )
        out = _rows(row, PUBLIC_KEY, probs / probs.sum())
        frames.append(
            out.assign(
                season=record["season"], jornada=record["jornada"], position=record["position"]
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    rows = pd.concat(frames, ignore_index=True)[columns]
    return rows[rows["FTR"].isin(OUTCOMES)].reset_index(drop=True)


def summarize_clubs(backtest: pd.DataFrame, tickets: pd.DataFrame) -> dict[str, Any]:
    """Metrics of the club models on every ESPN match and on the tickets.

    Args:
        backtest: Output of :func:`club_backtest`.
        tickets: Output of :func:`ticket_predictions`.

    Returns:
        As ``quiniela.evaluation.national.summarize_national``: ``all``,
        ``tickets``, ``public`` and ``by_year``.
    """
    return summarize_national(backtest, tickets, public_key=PUBLIC_KEY)


def club_model(espn: pd.DataFrame, before: pd.Timestamp | None = None) -> Any:
    """v1C fitted on the club matches played before a date.

    Args:
        espn: ESPN club matches (played and fixtures).
        before: Only the matches before it train the model. Defaults to every played match.

    Returns:
        The fitted model, or ``None`` without played matches.
    """
    played = espn.dropna(subset=["FTR"])
    train = played if before is None else played[played["date"] < before]
    if train.empty:
        return None
    return CLUB_MODELS[ELO_KEY].factory().fit(train.sort_values("date", kind="stable"))
