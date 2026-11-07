"""Prediction tasks for pyinvoke.

Examples::

    invoke predict.matchday                 # last matchday of the current season
    invoke predict.matchday --number 12
    invoke predict.all
    invoke predict.all dashboards.build     # predict and refresh the dashboards
"""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# Every task runs the project CLI inside the Poetry virtual environment
CLI = "poetry run quiniela predict"

# %% TASKS


@task(
    help={
        "number": "Matchday of the current season to predict. Defaults to the last one.",
        "model": "Model key (default: head_to_head).",
    }
)
def matchday(ctx: Context, number: int | None = None, model: str = "head_to_head") -> None:
    """Predict one matchday of the current season (the last one by default)."""
    command = f"{CLI} --model {model}"
    if number is not None:
        # Invoke only infers types from default values, so CLI input arrives as str
        command += f" --matchday {int(number)}"
    ctx.run(command)


# Exposed as "predict.all"; the function name avoids shadowing the builtin
@task(name="all", help={"model": "Model key (default: head_to_head)."})
def all_matchdays(ctx: Context, model: str = "head_to_head") -> None:
    """Predict every matchday of the current season."""
    ctx.run(f"{CLI} --model {model} --all")
