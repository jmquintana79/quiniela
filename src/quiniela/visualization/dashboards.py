"""Build static HTML dashboards from the downloaded match data.

Each dashboard is a single self-contained HTML file: the data is computed here
with pandas, serialised to JSON and embedded in an HTML template, so the file
opens with a double click (no server, no network access needed).

Dashboards:

- ``current.html``: season in progress (upcoming fixtures with market
  probabilities, latest results and standings).
- ``historical.html``: completed seasons (1X2 split and goals per season,
  plus standings and results of the selected season).
"""

import json
import logging
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from quiniela.config import CURRENT_DIR, DASHBOARDS_DIR, HISTORICAL_DIR
from quiniela.io.download import LEAGUES

logger = logging.getLogger(__name__)

# football-data.co.uk league code -> display name
LEAGUE_NAMES: dict[str, str] = {code: league for league, code in LEAGUES.items()}

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
        upcoming: list[dict[str, Any]] = []
        if not fixtures.empty:
            fixtures = fixtures.reset_index(drop=True).sort_values("date")
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
    ]
