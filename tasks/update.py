"""Update task for pyinvoke: bring data, predictions and dashboards up to date.

Examples::

    invoke update               # at the start of every matchday
    invoke update --backtest    # also re-run the backtest (after changing a model)
    invoke update --no-browser  # never open the LAE pages (missing LAE data is a warning)
"""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# %% TASKS


@task(
    help={
        "backtest": "Also re-run the backtest of every model (about ten minutes).",
        "browser": "Open the LAE pages when the next ticket or a draw is missing (default on).",
    }
)
def update(ctx: Context, backtest: bool = False, browser: bool = True) -> None:
    """Download what is missing (LAE with one click in the browser), predict, rebuild dashboards."""
    flags = (" --backtest" if backtest else "") + ("" if browser else " --no-browser")
    # pty: the update waits for Enter while the LAE pages are open
    ctx.run(f"poetry run quiniela update{flags}", pty=True)
