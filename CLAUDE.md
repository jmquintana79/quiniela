# CLAUDE.md

Guidance for Claude Code in this repository. Read `README.md` (architecture, models, metrics,
releases) and `TODO.todo` (pending work, in Spanish) before starting a task.

## Project

Python ML project that predicts the 1X2 results of Spanish football (La Liga `SP1` and
Segunda `SP2`) for the quiniela. Data from football-data.co.uk through `soccerdata`; every
model is evaluated with a walk-forward backtest against the betting market.

- Package: `src/quiniela/` · tasks: `tasks/` (invoke) · tests: `tests/` (pytest).
- Python `~3.11`, Poetry 1.8, virtual environment in `.venv` (`source .venv/bin/activate`).
- Data (`data/`), dashboards (`dashboards/`) and API docs (`docs/api/`) are generated and
  not versioned.

## Working with the user

- The user writes in Spanish: answer in Spanish.
- When the user says "solo responde", "responde sin hacer nada" or similar, answer the
  question only: do not edit files or run changes.
- Commit, push and create tags only when the user asks.

## Code conventions

- Everything in the code and the README is in **English**; `TODO.todo` is in Spanish.
- Type hints everywhere (mypy `strict`), Google-style docstrings (`Args:`, `Returns:`,
  `Raises:`) and comments explaining the *why*. Match the style of the surrounding code.
- Models follow the **scikit-learn estimator API** (`fit`, `predict`, `predict_proba`,
  `classes_` in `H`, `D`, `A` order), subclassing `ClassifierMixin, BaseEstimator` with
  `# type: ignore[misc]` on the class line.
- Each model version lives in **its own module** in `src/quiniela/prediction/` and is
  registered in `quiniela.prediction.registry.MODELS` (with `version`, and `reference=True`
  only for baselines). Registered models are picked up automatically by the predictions, the
  backtest and the dashboards (new columns).
- Models receive the whole match table as `X` (league, date, teams, odds and, when fitting,
  goals); the target's goals and result are removed before `predict`. Never use information
  unavailable before the match (no closing odds, no look-ahead).
- v4 (`quiniela.prediction.planner`) is a decision layer, not a registered model: it places
  doubles and triples on the v3 ticket. The dashboards show it as the pseudo-model `planned`;
  the backtest replays it in `ticket_strategies` (saved to `data/backtest/tickets.parquet`).
- New submodules of packages with `__all__` (`quiniela.config`, `quiniela.prediction`,
  `quiniela.evaluation`) must be added to `__all__` so pdoc documents them.

## Formulas (LaTeX)

- README: display formulas in ` ```math ` blocks, inline `$...$`.
- Docstrings (pdoc with `--math`): raw strings `r"""..."""`; `$...$` / `$$...$$`;
  `\lbrace ... \rbrace` for literal braces; `\\\\` for row breaks in `cases`; no formula line
  starting with `-` or `+` (Markdown reads it as a list).

## Commands

```bash
invoke update               # download what is missing, predict, rebuild dashboards
invoke update --backtest    # also re-run the backtest (after changing a model)
invoke backtest.run         # every model over 2015-16 to the last season (~4 min)
invoke dev.all              # format, lint, mypy, pytest, API docs: run before committing
```

Tests must not need network access (mock downloads, use small synthetic data from
`tests/helpers.py`).

## Releasing a model version

1. Run the backtest and update the README: **Predictive models** (results tables, a
   subsection with the formulas in LaTeX and observations), **Evaluation** if metrics change,
   **Roadmap** and `TODO.todo` (mark done items).
2. Bump the version in `pyproject.toml` and `src/quiniela/__init__.py`.
3. Add a row to **Releases** in the README.
4. `invoke dev.all`, commit with a Conventional Commits message (`feat:`, `fix:`, `docs:`,
   `refactor:`), annotated tag `vX.Y.Z` (`git tag -a`), push the commit and the tag.

## Gotchas

- The shell is bash 3.2: keep commands simple (no long inline scripts with backticks; write
  one-off scripts to the scratchpad and run them).
- football-data.co.uk's fixtures file can lag behind and still list played matches;
  `load_current` deduplicates fixtures by league, home and away team.
- Some old source CSVs have rows longer than their header; `quiniela.io.download` replaces
  soccerdata's parser to keep them.
- Segunda is registered in soccerdata from `quiniela.io.download` (not built in).
- The ClubElo API (soccerdata `ClubElo`) returned 502 when last tried.
- Poetry 1.8.3 needs `pkginfo>=1.12,<2` in its own environment to read recent packages.
