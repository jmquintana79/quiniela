"""Development tasks for pyinvoke: tests, linting, formatting, type checks and API docs.

Examples::

    invoke dev.all                # format, lint, type-check, test and build the docs
    invoke dev.test
    invoke dev.lint
    invoke dev.format             # rewrite files
    invoke dev.format --check     # only report files that would change
    invoke dev.typecheck
    invoke dev.docs               # HTML API docs in docs/api/
    invoke dev.docs --serve       # live docs server in the browser
"""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# Every task runs inside the Poetry virtual environment
RUN = "poetry run"

# Folders checked by the linters
SOURCES = "src tests tasks"

# Folders type-checked by mypy (tests are left out: fixtures and asserts add noise)
TYPED_SOURCES = "src tasks"

# Package documented by pdoc and the folder where the HTML docs are written
DOCS_PACKAGE = "quiniela"
DOCS_DIR = "docs/api"

# Docstrings follow the Google style (Args:, Returns:, Raises:)
DOCS_FORMAT = "google"

# %% TASKS


@task
def test(ctx: Context) -> None:
    """Run the test suite (no network access needed)."""
    ctx.run(f"{RUN} pytest")


@task
def lint(ctx: Context) -> None:
    """Check the code with ruff (without fixing anything)."""
    ctx.run(f"{RUN} ruff check --no-fix {SOURCES}")


# Exposed as "dev.format"; the function name avoids shadowing the builtin
@task(name="format", help={"check": "Only report files that would be reformatted."})
def format_code(ctx: Context, check: bool = False) -> None:
    """Format the code with ruff."""
    ctx.run(f"{RUN} ruff format {'--check ' if check else ''}{SOURCES}")


@task
def typecheck(ctx: Context) -> None:
    """Type-check the code with mypy (strict mode, see pyproject.toml)."""
    ctx.run(f"{RUN} mypy {TYPED_SOURCES}")


@task(help={"serve": "Start a live docs server and open it in the browser instead."})
def docs(ctx: Context, serve: bool = False) -> None:
    """Generate the HTML API documentation of src/ with pdoc."""
    output = "" if serve else f" --output-directory {DOCS_DIR}"
    # pdoc imports the package; allow the subprocesses some libraries start on import
    ctx.run(
        f"{RUN} pdoc --docformat {DOCS_FORMAT}{output} {DOCS_PACKAGE}",
        env={"PDOC_ALLOW_EXEC": "1"},
    )


# Exposed as "dev.all"; runs the tasks below in order and stops at the first failure
@task(name="all", pre=[format_code, lint, typecheck, test, docs])
def all_checks(ctx: Context) -> None:
    """Format, lint, type-check, test and build the API docs."""
