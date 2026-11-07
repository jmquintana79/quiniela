"""Data download tasks for pyinvoke.

Examples::

    invoke download.historical --start 2015 --end 2025
    invoke download.historical --start 1993
    invoke download.current
"""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# Every task runs the project CLI inside the Poetry virtual environment
CLI = "poetry run quiniela download"

# %% TASKS


@task(
    help={
        "start": "Start year of the first season to download (2015 = 2015-16).",
        "end": "Start year of the last season (inclusive). Defaults to the last completed one.",
    }
)
def historical(ctx: Context, start: int, end: int | None = None) -> None:
    """Download completed seasons into data/historical."""
    # Invoke only infers types from default values, so CLI input arrives as str
    command = f"{CLI} historical --start {int(start)}"
    if end is not None:
        command += f" --end {int(end)}"
    ctx.run(command)


@task
def current(ctx: Context) -> None:
    """Download the season in progress and upcoming fixtures into data/current."""
    ctx.run(f"{CLI} current")
