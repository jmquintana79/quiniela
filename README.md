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
`predict_proba`) and are registered in `quiniela.prediction.MODELS`:

| Key | Model |
|---|---|
| `head_to_head` | Past results between the same teams at the same ground, smoothed towards the league's home/draw/away shares (benchmark). |

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
│   ├── prediction/          # models, prediction pipeline and quiniela ticket rules
│   ├── visualization/       # dashboard builder and HTML templates
│   └── scripts.py           # `quiniela` command line entry point
├── tasks/                   # invoke tasks (download, predict, dashboards)
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

```bash
pytest                # tests (no network access needed)
ruff check src tests tasks
ruff format src tests tasks
mypy src tasks
```

[commitizen](https://commitizen-tools.github.io/commitizen/) is configured in
`pyproject.toml` for [Conventional Commits](https://www.conventionalcommits.org) messages
and version bumps (`cz commit`, `cz bump`).

## Background

This repository started in 2021 as a scraper for a Spanish football results dataset (see
`docs/references.md`). It has been rebuilt as a Poetry-managed project on top of
football-data.co.uk, which provides historical data and weekly updates from a single source.
