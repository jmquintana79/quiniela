"""Backtest tasks for pyinvoke: evaluate every model over past seasons.

Examples::

    invoke backtest.run                        # every model, 2015-16 to the last season
    invoke backtest.run --start 2020 --end 2024
    invoke backtest.run --model dixon_coles
    invoke backtest.run dashboards.build       # evaluate and refresh the dashboards
"""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# Every task runs the project CLI inside the Poetry virtual environment
CLI = "poetry run quiniela backtest"

# %% TASKS


@task(
    help={
        "start": "Start year of the first season (default: 2015, i.e. 2015-16).",
        "end": "Start year of the last season (inclusive). Defaults to the last downloaded one.",
        "model": "Model key, or 'all' for every registered model (default).",
    }
)
def run(ctx: Context, start: int = 2015, end: int | None = None, model: str = "all") -> None:
    """Predict every matchday of past seasons with the models and print their metrics."""
    command = f"{CLI} --start {int(start)} --model {model}"
    if end is not None:
        # Invoke only infers types from default values, so CLI input arrives as str
        command += f" --end {int(end)}"
    ctx.run(command)
