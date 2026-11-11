"""Bring everything up to date in one step: data, predictions and dashboards.

Meant to be run at the start of every matchday (or after weeks without
running anything). In order:

1. **Missing completed seasons**: every completed season (up to the one before
   the current season) that is not in ``data/historical/`` is downloaded, per
   league. Covers the change of season (the season that just ended joins the
   history), a first installation and gaps. Seasons already downloaded are not
   requested again.
2. **Current season**: results played so far and upcoming fixtures, always
   downloaded (the source does not say whether its files changed; they are a
   few small CSV files).
3. **Predictions**: every matchday of the current season with every model,
   walk-forward. Covers skipped weeks and replaces the previous season's
   predictions when a new season starts.
4. **Backtest** (optional): every model over past seasons, when a model
   changed or a season ended.
5. **Dashboards**: rebuilt from the local data.

Errors:

- If downloading the current season fails, the update stops before touching
  predictions and dashboards (so old and new data are never mixed), except in
  the off-season (``OFF_SEASON_MONTHS``), when the new season's files may not
  be published yet: predictions are skipped and the dashboards are rebuilt.
- If there is no current-season match to predict, predictions are skipped
  with a warning.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from quiniela.config import HISTORICAL_DIR
from quiniela.evaluation import backtest
from quiniela.io import download
from quiniela.prediction import pipeline
from quiniela.visualization import dashboards

logger = logging.getLogger(__name__)

# Months between seasons, when the new season's files may not exist yet
OFF_SEASON_MONTHS = (7, 8)


@dataclass
class UpdateReport:
    """What an update did.

    Attributes:
        historical: Season files downloaded into the history.
        current: Current-season files downloaded.
        predictions: Prediction files written (one per model).
        backtest: Backtest file written, if the backtest was run.
        dashboards: Dashboards rebuilt.
        warnings: Steps skipped and why.
    """

    historical: list[Path] = field(default_factory=list)
    current: list[Path] = field(default_factory=list)
    predictions: list[Path] = field(default_factory=list)
    backtest: Path | None = None
    dashboards: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One line per step, for the command line."""
        lines = [
            f"Historical seasons downloaded: {len(self.historical)} files",
            f"Current season downloaded: {len(self.current)} files",
            f"Predictions: {len(self.predictions)} models",
            f"Backtest: {'updated' if self.backtest else 'not run'}",
            f"Dashboards: {len(self.dashboards)} rebuilt",
        ]
        lines += [f"Warning: {warning}" for warning in self.warnings]
        return "\n".join(lines)


def missing_seasons(
    last_completed: int, historical_dir: Path = HISTORICAL_DIR
) -> dict[str, list[int]]:
    """Completed seasons not downloaded yet, per league.

    Args:
        last_completed: Start year of the last completed season.
        historical_dir: Folder with ``<code>/<season>.parquet`` files.

    Returns:
        Mapping from soccerdata league id to the missing season start years
        (leagues with nothing missing are omitted).
    """
    missing: dict[str, list[int]] = {}
    for league, code in download.LEAGUES.items():
        years = [
            year
            for year in range(download.FIRST_SEASON[code], last_completed + 1)
            if not (historical_dir / code / f"{download.season_label(year)}.parquet").exists()
        ]
        if years:
            missing[league] = years
    return missing


def _runs(years: list[int]) -> list[tuple[int, int]]:
    """Group sorted years into consecutive ranges (``[1, 2, 4]`` -> ``[(1, 2), (4, 4)]``)."""
    runs: list[tuple[int, int]] = []
    for year in years:
        if runs and year == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], year)
        else:
            runs.append((year, year))
    return runs


def run_update(
    run_backtest: bool = False,
    today: date | None = None,
    historical_dir: Path = HISTORICAL_DIR,
) -> UpdateReport:
    """Update the data, predictions and dashboards (see the module docstring).

    Args:
        run_backtest: Also re-run the backtest of every model.
        today: Reference date (season and off-season detection). Defaults to today.
        historical_dir: Folder with the completed seasons.

    Returns:
        What was done.

    Raises:
        Exception: The error of the current-season download, outside the off-season.
    """
    today = today or date.today()
    report = UpdateReport()

    # 1. Completed seasons that are not downloaded yet
    last_completed = download.current_season(today) - 1
    missing = missing_seasons(last_completed, historical_dir)
    if not missing:
        logger.info("History up to date (last completed season %s)", last_completed)
    for league, years in missing.items():
        for start, end in _runs(years):
            logger.info("Downloading missing %s seasons %s to %s", league, start, end)
            report.historical += download.download_historical(start, end, leagues=[league])

    # 2. Current season, always refreshed
    try:
        report.current = download.download_current()
    except Exception as error:
        if today.month not in OFF_SEASON_MONTHS:
            raise
        report.warnings.append(
            f"Current season not downloaded ({error}); in the off-season its files may not be "
            "published yet. Predictions skipped."
        )

    # 3. Predictions of every matchday with every model
    if report.current:
        try:
            report.predictions = pipeline.run(pipeline.ALL_MODELS, all_matchdays=True)
        except ValueError as error:
            report.warnings.append(f"Predictions skipped: {error}")

    # 4. Optional backtest of every model over past seasons
    if run_backtest:
        report.backtest = backtest.save_backtest(backtest.run_backtest())

    # 5. Dashboards from the local data
    report.dashboards = dashboards.build_dashboards()

    for warning in report.warnings:
        logger.warning(warning)
    return report
