"""Update task for pyinvoke: bring data, predictions and dashboards up to date.

Examples::

    invoke update               # at the start of every matchday
    invoke update --backtest    # also re-run the backtest (after changing a model)
"""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# %% TASKS


@task(help={"backtest": "Also re-run the backtest of every model (about two minutes)."})
def update(ctx: Context, backtest: bool = False) -> None:
    """Download missing seasons and the current one, predict every matchday, rebuild dashboards."""
    ctx.run(f"poetry run quiniela update{' --backtest' if backtest else ''}")
