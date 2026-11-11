"""Build static HTML dashboards from the downloaded match data.

Each dashboard is a single self-contained HTML file: the data is computed here
with pandas, serialised to JSON and embedded in an HTML template, so the file
opens with a double click (no server, no network access needed).

Dashboards:

- ``current.html``: season in progress (upcoming fixtures with market
  probabilities, latest results and standings).
- ``historical.html``: completed seasons (1X2 split and goals per season,
  plus standings and results of the selected season).
- ``predictions.html``: every matchday of the current season with the real
  result, one column per model and the quiniela ticket outcome.
- ``evaluation.html``: backtest metrics of every model over past seasons,
  one column per model, with the market as reference.
"""

import json
import logging
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quiniela.config import CURRENT_DIR, DASHBOARDS_DIR, HISTORICAL_DIR
from quiniela.data.loading import pending_fixtures
from quiniela.io.download import LEAGUES

logger = logging.getLogger(__name__)

# football-data.co.uk league code -> display name
LEAGUE_NAMES: dict[str, str] = {
    LEAGUES["ESP-La Liga"]: "La Liga",
    LEAGUES["ESP-Segunda"]: "Segunda",
}

# Spain moved from 2 to 3 points per win in the 1995-96 season
THREE_POINTS_FROM = 1995

# Number of recent matches shown as "form" in the standings
FORM_LENGTH = 5

# Columns kept for every match (the rest of the ~200 columns are betting odds)
MATCH_COLUMNS = ["date", "home_team", "away_team", "FTHG", "FTAG", "FTR"]

# Odds used for the implied probabilities, in order of preference
ODDS_COLUMNS: list[tuple[str, str, str]] = [
    ("AvgH", "AvgD", "AvgA"),  # market average
    ("B365H", "B365D", "B365A"),  # Bet365, available for most fixtures
]

# Placeholders replaced in the templates
STYLE_PLACEHOLDER = "/*__STYLE__*/"
SCRIPT_PLACEHOLDER = "/*__SCRIPT__*/"
DATA_PLACEHOLDER = "/*__DATA__*/null"


# %% DATA PREPARATION


def standings(matches: pd.DataFrame, points_per_win: int = 3) -> pd.DataFrame:
    """Compute a league table from match results.

    Teams are ranked by points, goal difference and goals scored. Head-to-head
    tiebreakers and point deductions are not applied, so positions can differ
    slightly from the official table.

    Args:
        matches: Played matches with ``home_team``, ``away_team``, ``FTHG``,
            ``FTAG`` and ``date`` columns.
        points_per_win: Points awarded for a win.

    Returns:
        One row per team with ``team``, ``played``, ``won``, ``drawn``,
        ``lost``, ``gf``, ``ga``, ``gd``, ``points`` and ``form`` (results of
        the last matches, oldest first, as a string such as ``"WDLWW"``).
    """
    played = matches.dropna(subset=["FTHG", "FTAG"])
    # One row per team and match, from that team's point of view
    home = pd.DataFrame(
        {
            "team": played["home_team"],
            "date": played["date"],
            "gf": played["FTHG"],
            "ga": played["FTAG"],
        }
    )
    away = pd.DataFrame(
        {
            "team": played["away_team"],
            "date": played["date"],
            "gf": played["FTAG"],
            "ga": played["FTHG"],
        }
    )
    rows = pd.concat([home, away], ignore_index=True)
    rows["result"] = "D"
    rows.loc[rows["gf"] > rows["ga"], "result"] = "W"
    rows.loc[rows["gf"] < rows["ga"], "result"] = "L"

    table = rows.groupby("team").agg(
        played=("result", "size"),
        won=("result", lambda r: int((r == "W").sum())),
        drawn=("result", lambda r: int((r == "D").sum())),
        lost=("result", lambda r: int((r == "L").sum())),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
    )
    table["gd"] = table["gf"] - table["ga"]
    table["points"] = table["won"] * points_per_win + table["drawn"]
    table["form"] = (
        rows.sort_values("date")
        .groupby("team")["result"]
        .agg(lambda r: "".join(r.tail(FORM_LENGTH)))
    )
    table = table.reset_index().sort_values(
        ["points", "gd", "gf", "team"], ascending=[False, False, False, True]
    )
    return table.astype({c: int for c in ["gf", "ga", "gd", "points"]}).reset_index(drop=True)


def implied_probabilities(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Convert 1X2 odds into probabilities, removing the bookmaker margin.

    The market average is used when available, otherwise Bet365.

    Args:
        fixtures: Upcoming matches with odds columns.

    Returns:
        DataFrame aligned with ``fixtures`` with columns ``p1``, ``px`` and
        ``p2`` (NaN when no odds are published for a match).
    """
    probs = pd.DataFrame(index=fixtures.index, columns=["p1", "px", "p2"], dtype=float)
    for columns in ODDS_COLUMNS:
        if not set(columns).issubset(fixtures.columns):
            continue
        inverse = 1 / fixtures[list(columns)].astype(float)
        # Normalise so the three outcomes add up to 1
        normalised = inverse.div(inverse.sum(axis=1), axis=0)
        normalised.columns = probs.columns
        probs = probs.fillna(normalised)
    return probs


def season_summary(matches: pd.DataFrame) -> dict[str, float | int]:
    """Summarise the outcomes of a set of matches.

    Args:
        matches: Played matches with ``FTR``, ``FTHG`` and ``FTAG`` columns.

    Returns:
        Number of matches, share of home wins / draws / away wins and average
        goals per match.
    """
    outcomes = matches["FTR"].value_counts(normalize=True)
    return {
        "matches": len(matches),
        "home": round(float(outcomes.get("H", 0.0)), 4),
        "draw": round(float(outcomes.get("D", 0.0)), 4),
        "away": round(float(outcomes.get("A", 0.0)), 4),
        "goals": round(float((matches["FTHG"] + matches["FTAG"]).mean()), 3),
    }


def _season_start_year(path: Path) -> int:
    """Read the season start year from a historical file name (``2015-16.parquet``)."""
    return int(path.stem.split("-")[0])


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-friendly records (dates as ISO strings, NaN as null)."""
    out = df.copy()
    for column in out.select_dtypes(include="datetime").columns:
        out[column] = out[column].dt.strftime("%Y-%m-%d %H:%M")
    records: list[dict[str, Any]] = json.loads(out.to_json(orient="records"))
    return records


def _matches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the match columns used by the dashboards, sorted by date."""
    return df.reset_index(drop=True)[MATCH_COLUMNS].sort_values("date").reset_index(drop=True)


def current_payload(current_dir: Path = CURRENT_DIR) -> dict[str, Any]:
    """Collect the data shown in the current season dashboard.

    Args:
        current_dir: Folder with ``<code>/results.parquet`` and ``<code>/fixtures.parquet``.

    Returns:
        JSON-serialisable dictionary with one entry per league.
    """
    leagues = []
    for code, name in LEAGUE_NAMES.items():
        results_path = current_dir / code / "results.parquet"
        if not results_path.exists():
            logger.warning("No current data for %s (%s missing)", name, results_path)
            continue
        results = _matches(pd.read_parquet(results_path))
        table = standings(results)

        fixtures_path = current_dir / code / "fixtures.parquet"
        fixtures = pd.read_parquet(fixtures_path) if fixtures_path.exists() else pd.DataFrame()
        # Odds change until kick-off: the page says when this snapshot was taken
        odds_downloaded = (
            datetime.fromtimestamp(fixtures_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            if fixtures_path.exists()
            else None
        )
        upcoming: list[dict[str, Any]] = []
        if not fixtures.empty:
            # The source can still list matches already played: those are not "next"
            fixtures = pending_fixtures(fixtures.reset_index(drop=True), results)
            fixtures = fixtures.sort_values("date")
            upcoming_df = fixtures[["date", "home_team", "away_team"]].join(
                implied_probabilities(fixtures).round(4)
            )
            upcoming = _records(upcoming_df)

        # A matchday has one match per pair of teams
        latest = results.tail(len(table) // 2).iloc[::-1]
        leagues.append(
            {
                "code": code,
                "name": name,
                "updated": results["date"].max().strftime("%Y-%m-%d"),
                "odds_downloaded": odds_downloaded,
                "fixtures": upcoming,
                "latest": _records(latest),
                "standings": _records(table),
            }
        )
    return {"leagues": leagues}


def historical_payload(historical_dir: Path = HISTORICAL_DIR) -> dict[str, Any]:
    """Collect the data shown in the historical dashboard.

    The per-season summary is precomputed for every season. Matches and
    standings are embedded too, but the page only renders the selected season.

    Args:
        historical_dir: Folder with ``<code>/<season>.parquet`` files.

    Returns:
        JSON-serialisable dictionary with one entry per league.
    """
    leagues = []
    for code, name in LEAGUE_NAMES.items():
        paths = sorted((historical_dir / code).glob("*.parquet"), key=_season_start_year)
        if not paths:
            logger.warning("No historical data for %s in %s", name, historical_dir / code)
            continue
        seasons = []
        for path in paths:
            start_year = _season_start_year(path)
            matches = _matches(pd.read_parquet(path))
            points = 3 if start_year >= THREE_POINTS_FROM else 2
            seasons.append(
                {
                    "season": path.stem,
                    "summary": season_summary(matches),
                    "standings": _records(standings(matches, points).drop(columns="form")),
                    "matches": _records(matches),
                }
            )
        leagues.append({"code": code, "name": name, "seasons": seasons})
    return {"leagues": leagues}


def _score(hits: int, misses: int) -> dict[str, Any]:
    """Hits, misses and hit rate of a set of predictions."""
    played = hits + misses
    return {"hits": hits, "misses": misses, "rate": round(hits / played, 4) if played else None}


def predictions_payload(
    current: pd.DataFrame | None = None,
    predictions: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Collect the data shown in the predictions dashboard.

    Every matchday of the current season is listed with the real result (when
    played) and one prediction column per model, plus each model's hits and
    the outcome of its 14-match quiniela ticket.

    Args:
        current: Current season with ``round`` and ``matchday`` columns.
            Defaults to the downloaded data.
        predictions: Stored predictions per model key. Defaults to the saved files.

    Returns:
        JSON-serialisable dictionary with the models, matchdays and season totals.
    """
    # Imported here: only this dashboard needs the prediction package
    from quiniela.prediction.registry import MODELS
    from quiniela.prediction.pipeline import current_matchdays, load_predictions
    from quiniela.prediction.ticket import evaluate_ticket, ticket_positions

    current = current_matchdays() if current is None else current
    predictions = load_predictions() if predictions is None else predictions
    keys = [key for key in MODELS if key in predictions]
    models = _model_specs(keys)

    # One column per model: predicted sign, its probability and, for models
    # that predict goals, the exact score and Pleno al 15 picks
    matches = current.copy()
    for key in keys:
        p = predictions[key].set_index(["league", "game"])
        aligned = p.reindex(pd.MultiIndex.from_frame(matches[["league", "game"]]))
        signs = aligned["prediction"]
        matches[f"{key}_sign"] = signs.to_numpy()
        # Probability of the predicted sign (a model may not always pick the largest one)
        prob = pd.Series(np.nan, index=aligned.index)
        for outcome in ("H", "D", "A"):
            chosen = (signs == outcome).to_numpy()
            prob[chosen] = aligned.loc[chosen, f"p_{outcome}"]
        matches[f"{key}_prob"] = prob.round(4).to_numpy()
        if "score_home" in aligned.columns and aligned["score_home"].notna().any():
            scored = aligned["score_home"].notna().to_numpy()
            score = pd.Series(None, index=aligned.index, dtype=object)
            pleno = pd.Series(None, index=aligned.index, dtype=object)
            score[scored] = (
                aligned.loc[scored, "score_home"].astype(int).astype(str)
                + "-"
                + aligned.loc[scored, "score_away"].astype(int).astype(str)
            )
            pleno[scored] = (
                aligned.loc[scored, "pleno_home"] + "-" + aligned.loc[scored, "pleno_away"]
            )
            matches[f"{key}_score"] = score.to_numpy()
            matches[f"{key}_pleno"] = pleno.to_numpy()

    totals = {
        key: {"hits": 0, "misses": 0, "missing": 0, "prizes": 0, "tickets": 0} for key in keys
    }
    matchdays = []
    for number, group in matches.groupby("matchday"):
        group = group.assign(ticket=ticket_positions(group))
        group = group.sort_values(["ticket", "league", "date"], na_position="last")
        played = group["FTR"].notna()
        in_ticket = group["ticket"].notna()

        summary: dict[str, Any] = {}
        for key in keys:
            predicted = group[f"{key}_sign"].notna()
            hit = played & predicted & (group[f"{key}_sign"] == group["FTR"])
            miss = played & predicted & ~hit
            ticket = evaluate_ticket(
                size=int(in_ticket.sum()),
                hits=int((hit & in_ticket).sum()),
                pending=int((in_ticket & ~played).sum()),
            )
            summary[key] = {
                **_score(int(hit.sum()), int(miss.sum())),
                "pending": int((predicted & ~played).sum()),
                # Matches the model could not predict: a third category next to
                # hits and misses, left out of the hit rate
                "missing": int((~predicted).sum()),
                # On the ticket they count as wrong lines (no sign marked)
                "missing_in_ticket": int((~predicted & in_ticket).sum()),
                "ticket": ticket.to_dict(),
            }
            totals[key]["hits"] += int(hit.sum())
            totals[key]["misses"] += int(miss.sum())
            totals[key]["missing"] += int((~predicted).sum())
            if ticket.status in ("prize", "no_prize"):
                totals[key]["tickets"] += 1
                totals[key]["prizes"] += ticket.status == "prize"

        columns = ["ticket", "league", "round", "date", "home_team", "away_team"]
        columns += ["FTHG", "FTAG", "FTR"]
        columns += [f"{k}_{field}" for k in keys for field in ("sign", "prob")]
        columns += [c for k in keys for c in (f"{k}_score", f"{k}_pleno") if c in group.columns]
        matchdays.append(
            {
                "number": int(number),
                "start": group["date"].min().strftime("%Y-%m-%d"),
                "end": group["date"].max().strftime("%Y-%m-%d"),
                "played": int(played.sum()),
                "matches": _records(group[columns]),
                "summary": summary,
            }
        )

    return {
        "models": models,
        "leagues": LEAGUE_NAMES,
        "matchdays": matchdays,
        "totals": {
            k: {
                **_score(t["hits"], t["misses"]),
                "missing": t["missing"],
                "prizes": t["prizes"],
                "tickets": t["tickets"],
            }
            for k, t in totals.items()
        },
    }


def _model_specs(keys: list[str]) -> list[dict[str, Any]]:
    """Display information of the given registered models, in registry order."""
    from quiniela.prediction.registry import MODELS

    return [
        {
            "key": key,
            "name": MODELS[key].name,
            "version": MODELS[key].version,
            "reference": MODELS[key].reference,
            "description": MODELS[key].description,
        }
        for key in MODELS
        if key in keys
    ]


# Metrics shown in the evaluation dashboard: key, label, which direction is
# better, display format, and the explanation and formula shown in the page
EVALUATION_METRICS: list[dict[str, str]] = [
    {
        "key": "hit_rate",
        "label": "Hit rate",
        "better": "higher",
        "format": "pct",
        "description": "Share of matches whose predicted sign (1, X or 2) was the real result.",
        "formula": "hits / N",
    },
    {
        "key": "log_loss",
        "label": "Log-loss",
        "better": "lower",
        "format": "num",
        "description": "Average surprise: minus the log of the probability given to the real "
        "result. Punishes confident mistakes hard. Always predicting 1/3 each scores 1.099.",
        "formula": "−(1/N) Σᵢ log pᵢ,real",
    },
    {
        "key": "brier",
        "label": "Brier score",
        "better": "lower",
        "format": "num",
        "description": "Average squared distance between the three probabilities and what "
        "happened (1 for the real result, 0 for the others). From 0 (perfect) to 2.",
        "formula": "(1/N) Σᵢ Σₖ (pᵢₖ − oᵢₖ)²",
    },
    {
        "key": "rps",
        "label": "RPS",
        "better": "lower",
        "format": "num",
        "description": "Ranked probability score: like Brier but on cumulative probabilities, "
        "so 1 → X → 2 are ordered and predicting a draw is a smaller mistake than "
        "predicting the opposite result. From 0 to 1.",
        "formula": "(1/N) Σᵢ ½ Σₖ₌₁² (Σⱼ≤ₖ pᵢⱼ − Σⱼ≤ₖ oᵢⱼ)²",
    },
    {
        "key": "draw_picks",
        "label": "X picks",
        "better": "info",
        "format": "pct",
        "description": "Share of matches where the model predicted a draw, to compare with "
        "the real share of draws below.",
        "formula": "",
    },
    {
        "key": "draw_rate",
        "label": "Real draws",
        "better": "info",
        "format": "pct",
        "description": "Share of matches that ended in a draw.",
        "formula": "",
    },
    {
        "key": "score_hit_rate",
        "label": "Exact score",
        "better": "higher",
        "format": "pct",
        "description": "Share of matches whose most likely score was the real one (models "
        "that predict goals).",
        "formula": "",
    },
    {
        "key": "pleno_hit_rate",
        "label": "Pleno al 15",
        "better": "higher",
        "format": "pct",
        "description": "Share of matches where both teams' goals category (0, 1, 2, M) was "
        "right, as in the Pleno al 15.",
        "formula": "",
    },
    {
        "key": "prizes",
        "label": "Prize tickets",
        "better": "higher",
        "format": "int",
        "description": "Matchdays whose 14-match ticket got 10 or more hits (5th category or "
        "better), out of the complete tickets. Only for both leagues together.",
        "formula": "",
    },
    {
        "key": "matches",
        "label": "Matches",
        "better": "info",
        "format": "int",
        "description": "Matches predicted (matches without a prediction, e.g. no odds for the "
        "market, are left out of every metric).",
        "formula": "",
    },
]


def _clean(record: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN with None and numpy numbers with plain Python ones (JSON-friendly)."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float | np.floating) and np.isnan(value):
            out[key] = None
        elif isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = round(float(value), 5)
        elif isinstance(value, float):
            out[key] = round(value, 5)
        else:
            out[key] = value
    return out


def evaluation_payload(backtest: pd.DataFrame | None = None) -> dict[str, Any]:
    """Collect the data shown in the evaluation (backtest) dashboard.

    Args:
        backtest: Backtest predictions with real results. Defaults to the saved backtest.

    Returns:
        JSON-serialisable dictionary with the models, metric definitions,
        metrics for every league/season combination and calibration tables.
    """
    # Imported here: only this dashboard needs the evaluation package
    from quiniela.evaluation.backtest import load_backtest, summarize
    from quiniela.evaluation.metrics import calibration_table

    backtest = load_backtest() if backtest is None else backtest
    if backtest.empty:
        return {"models": [], "metrics": EVALUATION_METRICS, "seasons": [], "results": {}}

    models = _model_specs(list(backtest["model"].unique()))
    seasons = sorted(backtest["season"].unique())

    # Metrics keyed by "<league>|<season>" ("all" for both leagues / every season)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for by in ([], ["season"], ["league"], ["season", "league"]):
        for row in summarize(backtest, by).to_dict(orient="records"):
            key = f"{row.get('league', 'all')}|{row.get('season', 'all')}"
            results.setdefault(key, {})[row["model"]] = _clean(row)

    calibration: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for league in ["all", *sorted(backtest["league"].unique())]:
        subset = backtest if league == "all" else backtest[backtest["league"] == league]
        for model in subset["model"].unique():
            rows = subset[subset["model"] == model].dropna(subset=["p_H", "p_D", "p_A"])
            if rows.empty:
                continue
            table = calibration_table(rows[["p_H", "p_D", "p_A"]].to_numpy(), rows["FTR"])
            calibration.setdefault(league, {})[model] = [
                _clean(r) for r in table.to_dict(orient="records")
            ]

    return {
        "models": models,
        "metrics": EVALUATION_METRICS,
        "leagues": {"all": "Both leagues", **LEAGUE_NAMES},
        "seasons": seasons,
        "matches": int(backtest[backtest["model"] == backtest["model"].iloc[0]].shape[0]),
        "results": results,
        "calibration": calibration,
    }


# %% RENDERING


def _template(name: str) -> str:
    """Read a file from the bundled templates folder."""
    return resources.files("quiniela.visualization").joinpath("templates", name).read_text("utf-8")


def render(template: str, payload: dict[str, Any], output: Path) -> Path:
    """Embed the shared stylesheet, helpers and the data into a template and write it.

    Args:
        template: Template file name inside ``visualization/templates``.
        payload: Data exposed to the page as ``DATA``.
        output: Destination HTML file.

    Returns:
        The path written.
    """
    # "</" would close the <script> tag if a value contained it
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (
        _template(template)
        .replace(STYLE_PLACEHOLDER, _template("base.css"))
        .replace(SCRIPT_PLACEHOLDER, _template("base.js"))
        # Data last, so placeholder-like text inside the data is never replaced
        .replace(DATA_PLACEHOLDER, data)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s (%.1f KB)", output, output.stat().st_size / 1024)
    return output


def build_dashboards(output_dir: Path = DASHBOARDS_DIR) -> list[Path]:
    """Build every dashboard from the current data folders.

    Args:
        output_dir: Folder where the HTML files are written.

    Returns:
        Paths of the dashboards written.
    """
    return [
        render("current.html", current_payload(), output_dir / "current.html"),
        render("historical.html", historical_payload(), output_dir / "historical.html"),
        render("predictions.html", predictions_payload(), output_dir / "predictions.html"),
        render("evaluation.html", evaluation_payload(), output_dir / "evaluation.html"),
    ]
