# Quiniela

Machine learning project to predict the results (1X2) of Spanish football matches for the
*quiniela*, covering La Liga (Primera División) and Segunda División.

The project downloads historical and current-season match data, predicts the matchdays of
the current season with a first benchmark model and shows everything in HTML dashboards.

## Data source

Match data comes from [football-data.co.uk](https://www.football-data.co.uk), downloaded
through the [soccerdata](https://github.com/probberechts/soccerdata) library:

| League | Code | First season |
|---|---|---|
| La Liga | `SP1` | 1993-94 |
| Segunda División | `SP2` | 1996-97 |

Every match includes the full-time and half-time score, match statistics (shots, corners,
cards) and pre-match odds from several bookmakers. Column meanings are described in
football-data.co.uk's [notes](https://www.football-data.co.uk/notes.txt).

The current season file is updated by the source about twice a week, and the upcoming
fixtures (the next matchday to predict) are published with their odds.

## Setup

Requirements: Python 3.11 and [Poetry](https://python-poetry.org) (1.8 or later).

```bash
poetry install --with dev,tests
source .venv/bin/activate
```

The virtual environment is created inside the project, in `.venv` (see `poetry.toml`).

## Usage

Every action is available as an [invoke](https://www.pyinvoke.org) task. List them with:

```bash
invoke --list
```

### Download data

Seasons are identified by their **start year**: `2015` means the 2015-16 season.

```bash
# Completed seasons, from --start to --end (inclusive)
invoke download.historical --start 1993 --end 2025

# Without --end, up to the last completed season
invoke download.historical --start 2015

# Season in progress: results played so far and upcoming fixtures
invoke download.current
```

Run `download.current` every week to refresh the current season. Completed seasons are
cached, so downloading them again does not hit the network.

### Predictions

```bash
invoke predict.matchday               # last matchday of the current season
invoke predict.matchday --number 12   # a given matchday
invoke predict.all                    # every matchday of the current season
```

Predictions are stored in `data/predictions/<model>.parquet`. For every matchday the model is
trained only on matches played before it (all historical seasons plus the earlier matches of
the current season), so past matchdays are evaluated without look-ahead.

**Matchdays.** football-data.co.uk has no round numbers, so they are inferred
(`quiniela.data.matchdays`): each league's rounds come from the teams' match counts and dates,
and a quiniela matchday pairs a La Liga round with the Segunda round played the same week.
Rescheduled matches join the matchday they are played in, so a round can show 9 or 12 La Liga
matches.

**Models** (`quiniela.prediction`) follow the scikit-learn API (`fit`, `predict`,
`predict_proba`) and are registered in `quiniela.prediction.registry.MODELS`. Each version is
described in [Predictive models](#predictive-models).

### Dashboards

```bash
invoke dashboards.build   # build dashboards/*.html from the downloaded data
invoke dashboards.open    # open them in the browser (macOS)

# Weekly refresh: new data, new predictions, rebuilt dashboards
invoke download.current predict.matchday dashboards.build
```

- `dashboards/current.html`: next matchday with the 1X2 probabilities implied by the
  market odds, current standings with recent form, and latest results.
- `dashboards/historical.html`: home win, draw and away win shares and goals per match for
  every season, plus the final standings and results of the selected season (filterable by
  team).
- `dashboards/predictions.html`: every matchday of the current season with the real result
  and one column per model (green when right, red when wrong), the hits and misses of each
  model, and whether its 14-match quiniela ticket would have won a prize (14 hits 1st, 13 2nd,
  12 3rd, 11 4th, 10 5th category; the Pleno al 15 is not simulated). The ticket is
  approximated with the matchday's La Liga matches followed by Segunda matches, up to 14.

Both are self-contained HTML files (data embedded, no external libraries) that open with a
double click, also offline.

Standings are computed from the results: 2 points per win before 1995-96 and 3 points
afterwards. Head-to-head tiebreakers and point deductions are not applied, so a position can
occasionally differ from the official table.

### Command line

The tasks wrap the `quiniela` command, which can also be used directly:

```bash
quiniela download historical --start 2015 --end 2025
quiniela download current
quiniela predict --matchday 12   # or --all; --model selects the model
quiniela dashboards
```

## Predictive models

Every model version is registered in `quiniela.prediction.registry.MODELS` under a key, and
gets its own column in the predictions dashboard. All versions are evaluated the same way: each
matchday of the current season is predicted with a model trained only on earlier matches, and
the predicted sign (1, X or 2) is compared with the real result.

Reference points for every version:

- **Always "1"**: the hit rate of predicting a home win for every match (the share of home
  wins). A useful model has to beat it.
- **Quiniela prizes**: matchdays whose 14-match ticket reaches 10 or more hits (see
  [Dashboards](#dashboards)).

| Version | Key | Status | Hit rate | Always "1" | Prizes |
|---|---|---|---|---|---|
| v1 · Head-to-head | `head_to_head` | **Current benchmark** | 43.8% | 45.8% | 1 of 10 tickets (5th category) |

*Season 2026-27, matchdays 1-13: 240 played matches (results up to 2 November 2026).*

### v1 · Head-to-head benchmark (`head_to_head`)

[`quiniela/prediction/head_to_head.py`](src/quiniela/prediction/head_to_head.py)

**Idea.** The probability of each result is the conditional probability of that result given
the fixture: how often the same home team beat, drew with or lost to the same away team **at
the same ground** in the past. For example, for Real Madrid vs Barcelona at the Bernabéu it
only looks at past Real Madrid (home) vs Barcelona (away) matches.

**Formula.** For a match between home team $h$ and away team $a$ in league $\ell$:

```math
P(r \mid h, a) = \frac{n_{h,a}(r) + \alpha \, \pi_\ell(r)}{n_{h,a} + \alpha},
\qquad r \in \{1, X, 2\}
```

where:

- $n_{h,a}(r)$ is the number of past matches of $h$ at home against $a$ that ended with
  result $r$, and $n_{h,a} = \sum_r n_{h,a}(r)$ is the number of those meetings.
- $\pi_\ell(r)$ is the share of result $r$ in league $\ell$ over the training data (the
  *prior*, e.g. about 46% home wins). The global share is used for a league not seen in
  training.
- $\alpha \ge 0$ is the smoothing strength (additive / Laplace smoothing), measured in
  "virtual matches" drawn from the prior. Default: $\alpha = 2$.

The predicted sign is the most likely result, with ties resolved in the order 1, X, 2:

```math
\hat{r} = \underset{r \in \{1, X, 2\}}{\arg\max} \; P(r \mid h, a)
```

**Smoothing.** With few meetings the raw frequencies are extreme (one past win gives 100%);
$\alpha$ pulls them towards the league prior:

- teams that never met ($n_{h,a} = 0$) get exactly the prior $\pi_\ell$;
- with a long history ($n_{h,a} \gg \alpha$) the observed frequencies dominate;
- $\alpha = 0$ gives the raw frequencies (and the prior when $n_{h,a} = 0$).

**Training data.** Every historical season (La Liga since 1993-94, Segunda since 1996-97) plus
the current season's matches played before the matchday being predicted.

**Results.**

| League | Matches | Hit rate | Always "1" |
|---|---|---|---|
| La Liga | 109 | 50.5% | 45.9% |
| Segunda | 131 | 38.2% | 45.8% |
| **All** | **240** | **43.8%** | **45.8%** |

**Observations.**

- It predicts a home win in 70% of the matches, while only 46% end that way, and it rarely
  predicts draws (14% of predictions against 23% of real results). The prior and short
  histories push most fixtures towards "1".
- It beats "always 1" in La Liga, where teams have long head-to-head histories, but not in
  Segunda, where squads change from season to season and many fixtures have few meetings.
- It ignores current form, injuries and team strength: a meeting from 1995 weighs the same as
  last season's.

## Project structure

```text
├── dashboards/              # generated HTML dashboards (not versioned)
├── data/                    # downloaded data (not versioned)
│   ├── historical/<code>/<season>.parquet   # e.g. SP1/2015-16.parquet
│   ├── current/<code>/results.parquet       # season in progress: played matches
│   ├── current/<code>/fixtures.parquet      # season in progress: upcoming matches
│   ├── predictions/<model>.parquet          # predictions for the current season
│   └── cache/                               # raw CSV files cached by soccerdata
├── src/quiniela/
│   ├── config/paths.py      # filesystem paths used by the project
│   ├── io/download.py       # historical and current season downloads
│   ├── data/                # match loaders and matchday inference
│   ├── prediction/          # model registry, models, prediction pipeline, ticket rules
│   ├── visualization/       # dashboard builder and HTML templates
│   └── scripts.py           # `quiniela` command line entry point
├── docs/api/                # generated API docs (not versioned)
├── tasks/                   # invoke tasks (download, predict, dashboards, dev)
└── tests/
```

The data folder can be moved elsewhere with the `QUINIELA_DATA_DIR` environment variable.

## Notes on the data

- **Segunda División** is not part of soccerdata's built-in leagues. It is registered in
  `quiniela.io.download`, so no configuration file in `~/soccerdata` is needed.
- **Malformed rows.** Some old source files contain rows with more fields than their header
  (e.g. La Liga 2004-05, Segunda 2002-03). soccerdata's parser silently drops those rows,
  so the project replaces it with one that keeps them: every downloaded season has the same
  number of matches as its source file.

## Development

Every development command has an invoke task in the `dev` namespace
([tasks/dev.py](tasks/dev.py)):

| Task | Runs | What it does |
|---|---|---|
| `invoke dev.test` | `pytest` | Test suite (no network access needed). |
| `invoke dev.lint` | `ruff check --no-fix src tests tasks` | Lint, without changing files. |
| `invoke dev.format` | `ruff format src tests tasks` | Format the code (`--check` only reports what would change). |
| `invoke dev.typecheck` | `mypy src tasks` | Strict type checking (settings in `pyproject.toml`). |
| `invoke dev.docs` | `pdoc --docformat google --math --output-directory docs/api quiniela` | Generate the API docs (`--serve` starts a live server instead). |
| `invoke dev.all` | all of the above | Format, lint, type-check, test and build the docs, stopping at the first failure. |

Run `invoke dev.all` before committing.

### API documentation

The API reference of `src/quiniela` is generated with [pdoc](https://pdoc.dev) from the
docstrings, which follow the
[Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
(`Args:`, `Returns:`, `Raises:`).

Docstrings can include LaTeX formulas, rendered with MathJax (`--math`): `$...$` inline and
`$$...$$` for display formulas (see `quiniela.prediction.head_to_head`). Write those
docstrings as raw strings (`r"""..."""`) so backslashes are kept, and use
`\lbrace ... \rbrace` for literal braces, because Markdown removes the backslash of `\{`.
MathJax is loaded from a CDN, so viewing the formulas needs an internet connection.

```bash
invoke dev.docs           # writes docs/api/ (not versioned)
open docs/api/index.html  # macOS

invoke dev.docs --serve   # live server that rebuilds on every change
```

pdoc documents the public members of every module. In packages that define `__all__`
(`quiniela.config`, `quiniela.prediction`), only the names listed there appear, so a new
submodule has to be added to `__all__` to show up in the docs.

### Commits

[commitizen](https://commitizen-tools.github.io/commitizen/) is configured in
`pyproject.toml` for [Conventional Commits](https://www.conventionalcommits.org) messages
and version bumps (`cz commit`, `cz bump`).

## Background

This repository started in 2021 as a scraper for a Spanish football results dataset (see
`docs/references.md`). It has been rebuilt as a Poetry-managed project on top of
football-data.co.uk, which provides historical data and weekly updates from a single source.
