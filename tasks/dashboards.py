"""Dashboard tasks for pyinvoke.

Examples::

    invoke dashboards.build
    invoke dashboards.open
    invoke download.current dashboards.build  # refresh data and dashboards
"""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# Generated dashboards, relative to the project root
DASHBOARDS = (
    "dashboards/current.html",
    "dashboards/historical.html",
    "dashboards/predictions.html",
)

# %% TASKS


@task
def build(ctx: Context) -> None:
    """Build the HTML dashboards in dashboards/ from the data folders."""
    ctx.run("poetry run quiniela dashboards")


# Exposed as "dashboards.open"; the function name avoids shadowing the builtin
@task(
    name="open",
    help={"name": "Dashboard to open: current, historical, predictions or all (default)."},
)
def open_dashboards(ctx: Context, name: str = "all") -> None:
    """Open the dashboards in the default browser (macOS)."""
    paths = DASHBOARDS if name == "all" else (f"dashboards/{name}.html",)
    ctx.run(f"open {' '.join(paths)}")
