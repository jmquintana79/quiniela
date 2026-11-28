# CLAUDE.md

Guidance for Claude Code in this repository. Read `README.md` (architecture, models, metrics,
releases) and `TODO.todo` (pending work, in Spanish) before starting a task. Before proposing
new ideas (brainstorming), also read `DISCARDED.todo` (discarded ideas with their reason)
so they are not discussed again; add every idea the user discards there.

## Project

Python ML project that predicts the 1X2 results of Spanish football (La Liga `SP1` and
Segunda `SP2`) for the quiniela. Data from football-data.co.uk through `soccerdata`; every
model is evaluated with a walk-forward backtest against the betting market.

- Package: `src/quiniela/` · tasks: `tasks/` (invoke) · tests: `tests/` (pytest).
- Python `~3.11`, Poetry 1.8, virtual environment in `.venv` (`source .venv/bin/activate`).
- Data (`data/`) and API docs (`docs/api/`) are generated and not versioned. The dashboards
  (`dashboards/`) are generated and versioned, except `historical.html` (too heavy) and
  `bookmarklet.html`: commit them after rebuilding when the user asks for a commit.

## Working with the user

- The user writes in Spanish: answer in Spanish.
- When the user says "solo responde", "responde sin hacer nada" or similar, answer the
  question only: do not edit files or run changes.
- Commit, push and create tags only when the user asks.

## Code conventions

- Everything in the code and the README is in **English**; `TODO.todo` is in Spanish, and so is
  **every text the dashboards show** (since v0.22.1): the templates, `base.js` (glossary included) and
  the display strings that reach them from Python (model, ticket, metric and strategy names and
  descriptions, messages, legends). Code, docstrings and comments stay in English; the dashboard
  file names and `<title>`s too. Numbers and dates use the Spanish format (`num0`, `es-ES`).
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
- **Notation: model · ticket.** Men's models are numbered (v1 head-to-head, v2 Dixon-Coles,
  v3 DC + market, v4 pi-ratings...); **Liga F models have their own counter with an F**
  (v1F public picks, `quiniela.prediction.public_picks`; v2F pi-ratings on the Liga F results of
  ESPN, `quiniela.io.espn`; v3F the same learned from goals and shots on target,
  `quiniela.prediction.pi_ratings_shots_f`; all in `WOMEN_MODELS`), with
  their own history (`quiniela.data.women`: `data/women/liga_f.parquet` and the archived
  next tickets in `data/lae/public/`) and evaluation (`quiniela.evaluation.women`). The **national
  teams** have their own counter with an N: v1N World Football Elo of every international since
  1872 (`quiniela.prediction.elo_national`, `NATIONAL_MODELS`, data from martj42 in
  `quiniela.io.international`, ticket matches in `quiniela.data.national`, evaluation in
  `quiniela.evaluation.national`); the **other club competitions** (European cups, Copa del Rey,
  foreign leagues) with a C: v1C club Elo of every ESPN match of 38 competitions since 2009
  (`quiniela.prediction.elo_clubs`, `CLUB_MODELS`, data in `quiniela.io.espn_clubs`, ticket matches
  linked by name similarity in `quiniela.data.clubs`, evaluation in `quiniela.evaluation.clubs`). The **Pleno
  al 15** has its own counter with a P: v1P Dixon-Coles' most likely goals categories, v2P the
  market's (`quiniela.prediction.pleno`, the tickets' pick since v0.19). Every
  dashboard shows men's and Liga F figures apart, after a table of contents (`data-toc`
  sections, `tableOfContents()` in `base.js`). Tickets are decision layers, not registered
  models, written as the models whose probabilities they use plus a letter: **·D** direct
  doubles and triples (`quiniela.prediction.planner`; **·D2** the same with 2 bets), **·R** with the official reductions
  (`quiniela.prediction.reduced`), **·B** big win and **·E** best expected prize
  (`quiniela.prediction.money`). They use `PLANNER_MODEL` (v3) on the men's matches and
  `WOMEN_PLANNER_MODEL` (v3F, with v1F as fallback) on the Liga F ones and
  `NATIONAL_PLANNER_MODEL` (v1N, since v0.20) on the national-team ones and `CLUB_PLANNER_MODEL`
  (v1C, since v0.21) on the other club ones: v3+v3F·D, ·D2, ·R, ·B and
  ·E (v3+v2F·D... in v0.12-v0.17, v3+v1F·D... in v0.10-v0.11, v3·D... in v0.9,
  v4, v4R and v5 until v0.8.0). Dashboard pseudo-models: `v3d2`, `v3d`, `v3r`,
  `v3b`; backtest strategies: `v3d` and `v3r` in `ticket_strategies`
  (`data/backtest/tickets.parquet`), and `v3d2`, `v3d`, `v3r`, `v3e`, `v3e_full`, `v3e_always`, `v3b`
  in the euro backtest `quiniela.evaluation.money` (`data/backtest/money.parquet`, from the
  official draws in `data/lae/`, `quiniela.data.prizes`). A reduced ticket's hits and prizes
  come from its real columns (`TicketPlan.columns`), never from counting covered matches.
- The next official ticket (`data/lae/upcoming.json`, saved with the Quiniela bookmark
  `tools/lae_bookmarklet.js` from `invoke update`, `quiniela.io.lae`) drives `current.html` (`quiniela.prediction.pipeline.predict_upcoming`); v3 uses its
  public picks for the matches no model covers. Matches carry a category (M, F, O, U:
  `quiniela.data.prizes.match_category`) and the official jornada they were on.
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
invoke backtest.run         # every model over 2015-16 to the last season, plus euros (~10 min)
invoke dev.all              # format, lint, mypy, pytest, API docs: run before committing
```

Tests must not need network access (mock downloads, use small synthetic data from
`tests/helpers.py`).
`tests/conftest.py` empties the national-team results and ESPN club matches for every test (a
test that needs them sets its own; the loaders' own tests call `monkeypatch.undo()`).

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
- The ClubElo API (soccerdata `ClubElo`, api.clubelo.com) is retired (no answer, November 2026):
  the ClubElo idea is discarded, see the end of `TODO.todo`.
- The LAE results service answers browsers only (scripts get "Access Denied"): its JSON is
  saved by the user's own browser (Quiniela bookmark; `invoke update` opens the pages and picks
  the files up from Downloads, see the README). Do not try to get around the protection.
  Its `temporada` field, not the request dates, gives the season of a draw.
- Poetry 1.8.3 needs `pkginfo>=1.12,<2` in its own environment to read recent packages.
