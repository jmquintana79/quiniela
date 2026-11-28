"""Evaluation of the national-team models, apart from the men's and Liga F backtests.

- **Every international** from ``FIRST_YEAR`` on (``quiniela.io.international``):
  v1N (``quiniela.prediction.elo_national``) predicts each match with the
  ratings before it, and its 1X2 regression is refitted at the start of
  every year on the earlier matches (a year is a few international windows:
  refitting it more often barely changes it). The ratings only use earlier
  results, so nothing is known after the match. The reference, the **result
  shares** of 1, X and 2 of the matches of the regression window, is scored
  the same way.
- **The national-team matches of the official tickets**
  (``quiniela.data.national``): the same predictions, in the ticket's order of
  the teams; the **public's picks** are scored too on the tickets whose
  betting page was archived (``quiniela.data.women.load_public_shares``).
"""

from typing import Any

import numpy as np
import pandas as pd

from quiniela.evaluation.metrics import contingency_table, evaluate
from quiniela.prediction.elo_national import (
    HOME_ADVANTAGE,
    OUTCOMES,
    WINDOW_YEARS,
    _neutral,
    elo_ratings,
    fit_ordinal,
    ordinal_probabilities,
)
from quiniela.prediction.registry import NATIONAL_MODELS

ELO_KEY = "elo_national"

# First year of the backtest (the tickets saved from LAE start in 2010-11)
FIRST_YEAR = 2010

# References of the national-team evaluation
SHARES_KEY = "national_shares"
PUBLIC_KEY = "national_public"
REFERENCE_SPECS: list[dict[str, Any]] = [
    {
        "key": SHARES_KEY,
        "name": "Proporciones",
        "version": "",
        "reference": True,
        "description": "Proporciones de 1, X y 2 de los partidos internacionales de los años anteriores "
        "(lo que usaban los boletos sin lo que marca el público); su signo es siempre que gana el local.",
    },
    {
        "key": PUBLIC_KEY,
        "name": "Público",
        "version": "",
        "reference": True,
        "description": "Porcentaje de apostantes de la quiniela que marca cada signo, normalizado: lo que "
        "usaban los boletos en los partidos de selecciones hasta la v0.19 (solo lo tienen los boletos "
        "archivados).",
    },
]

PREDICTION_COLUMNS = [
    "year",
    "date",
    "game",
    "home_team",
    "away_team",
    "tournament",
    "neutral",
    "FTR",
    "model",
    "p_H",
    "p_D",
    "p_A",
    "prediction",
]


def _rows(matches: pd.DataFrame, key: str, probs: np.ndarray) -> pd.DataFrame:
    """Prediction rows of one model for some matches."""
    return matches.assign(
        model=key,
        p_H=probs[:, 0],
        p_D=probs[:, 1],
        p_A=probs[:, 2],
        prediction=np.array(OUTCOMES)[probs.argmax(axis=1)],
    ).reindex(columns=PREDICTION_COLUMNS)


def national_backtest(results: pd.DataFrame, first_year: int = FIRST_YEAR) -> pd.DataFrame:
    """Walk-forward predictions of v1N and the result shares on every international.

    Args:
        results: Played internationals in date order (``quiniela.io.international``).
        first_year: First year predicted (earlier matches only train).

    Returns:
        ``PREDICTION_COLUMNS`` for every match from ``first_year`` on.
    """
    if results.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    played = results.reset_index(drop=True)
    before, _ = elo_ratings(played, HOME_ADVANTAGE)
    difference = (before[:, 0] - before[:, 1]) / 400
    home = (~_neutral(played)).astype(float)
    codes = played["FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy(dtype=int)
    years = played["date"].dt.year.to_numpy()
    matches = played.assign(
        year=years,
        game=played["date"].dt.strftime("%Y-%m-%d")
        + " "
        + played["home_team"]
        + "-"
        + played["away_team"],
    )
    frames = []
    for year in range(first_year, int(years.max()) + 1):
        target = years == year
        train = (years < year) & (years >= year - WINDOW_YEARS)
        if not target.any() or not train.any():
            continue
        coef = fit_ordinal(difference[train], home[train], codes[train])
        probs = ordinal_probabilities(coef, difference[target], home[target])
        frames.append(_rows(matches[target], ELO_KEY, probs))
        shares = np.bincount(codes[train], minlength=3) / train.sum()
        frames.append(_rows(matches[target], SHARES_KEY, np.tile(shares, (target.sum(), 1))))
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def ticket_predictions(
    national: pd.DataFrame,
    backtest: pd.DataFrame,
    public: dict[tuple[str, int], np.ndarray] | None = None,
) -> pd.DataFrame:
    """The backtest predictions of the national-team matches of the official tickets.

    Args:
        national: Ticket matches linked to their result (``quiniela.data.national.ticket_national``).
        backtest: Output of :func:`national_backtest`.
        public: Public's shares of every archived ticket (``load_public_shares``).

    Returns:
        ``PREDICTION_COLUMNS`` plus ``season``, ``jornada`` and ``position``,
        in the ticket's order of the teams (probabilities of 1 and 2 swapped
        when the results file lists them the other way round), with the
        public's picks as ``PUBLIC_KEY`` where the ticket was archived.
    """
    columns = [*PREDICTION_COLUMNS, "season", "jornada", "position"]
    if national.empty or backtest.empty:
        return pd.DataFrame(columns=columns)
    frames = []
    tickets = national.assign(
        _home=np.where(national["swapped"], national["away_team"], national["home_team"]),
        _away=np.where(national["swapped"], national["home_team"], national["away_team"]),
    )
    joined = _join(tickets, backtest)
    if not joined.empty:
        swapped = joined["swapped"].to_numpy(dtype=bool)
        probs = joined[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
        probs[swapped] = probs[swapped][:, ::-1]
        for key, rows in joined.assign(p_H=probs[:, 0], p_D=probs[:, 1], p_A=probs[:, 2]).groupby(
            "model"
        ):
            frames.append(
                _rows(rows, str(key), rows[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)).assign(
                    season=rows["season"].to_numpy(),
                    jornada=rows["jornada"].to_numpy(),
                    position=rows["position"].to_numpy(),
                )
            )
    for record in national.to_dict("records") if public else []:
        shares = public.get((str(record["season"]), int(record["jornada"]))) if public else None
        if shares is None:
            continue
        probs = np.asarray(shares[int(record["position"]) - 1], dtype=float)[None, :]
        row = pd.DataFrame([record]).assign(year=pd.Timestamp(record["date"]).year)
        frames.append(
            _rows(row, PUBLIC_KEY, probs / probs.sum()).assign(
                season=record["season"], jornada=record["jornada"], position=record["position"]
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def _join(tickets: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Ticket matches joined to the backtest rows of the same match (results-file order)."""
    right = backtest.drop(columns=["game", "FTR", "year", "neutral", "tournament"]).rename(
        columns={"home_team": "_home", "away_team": "_away"}
    )
    return (
        tickets.drop(columns=["year"], errors="ignore")
        .merge(right, on=["date", "_home", "_away"], how="inner")
        .assign(year=lambda rows: rows["date"].dt.year)
    )


def summarize_national(
    backtest: pd.DataFrame, tickets: pd.DataFrame, public_key: str = PUBLIC_KEY
) -> dict[str, Any]:
    """Metrics of the national-team models on every international and on the tickets.

    The club models are summarised the same way (``quiniela.evaluation.clubs``).

    Args:
        backtest: Output of :func:`national_backtest`.
        tickets: Output of :func:`ticket_predictions`.
        public_key: Model key of the public's picks in ``tickets``.

    Returns:
        ``all`` (every international), ``tickets`` (the ticket matches) and
        ``public`` (the ticket matches with public picks, every model on the
        same ones), each keyed by model with the metrics of ``evaluate`` and
        a ``contingency`` table; ``by_year`` gives the hit rate and log-loss of
        every model on every international by year.
    """

    def scores(rows: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, group in rows.groupby("model"):
            probs = group[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
            out[str(key)] = {
                **evaluate(probs, group["FTR"], group["prediction"]),
                "contingency": contingency_table(group["prediction"], group["FTR"]),
            }
        return out

    with_public = tickets.loc[tickets["model"] == public_key, "game"]
    by_year: dict[str, dict[str, Any]] = {}
    for (year, key), group in backtest.groupby(["year", "model"]):
        probs = group[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
        metrics = evaluate(probs, group["FTR"], group["prediction"])
        by_year.setdefault(str(year), {})[str(key)] = {
            "matches": metrics["matches"],
            "hit_rate": metrics.get("hit_rate"),
            "log_loss": metrics.get("log_loss"),
        }
    return {
        "all": scores(backtest),
        "tickets": scores(tickets[tickets["model"] != public_key]),
        "public": scores(tickets[tickets["game"].isin(with_public)]),
        "by_year": by_year,
    }


def national_model(results: pd.DataFrame, before: pd.Timestamp | None = None) -> Any:
    """v1N fitted on the internationals played before a date.

    Args:
        results: Played internationals (``quiniela.io.international.load_results``).
        before: Only the matches before it train the model (e.g. a ticket's
            sales deadline: its matches keep their pre-match prediction once
            played). Defaults to every match.

    Returns:
        The fitted model, or ``None`` without results.
    """
    train = results if before is None else results[results["date"] < before]
    if train.empty:
        return None
    return NATIONAL_MODELS[ELO_KEY].factory().fit(train)
