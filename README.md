# Quiniela

Machine learning project to predict the results (1X2) of Spanish football matches for the
*quiniela*, covering La Liga (Primera División) and Segunda División.

The project downloads historical and current-season match data, predicts every matchday with
several model versions (all following the scikit-learn API), evaluates them on past seasons
with a walk-forward backtest against the betting market, and shows everything in HTML
dashboards. See [Predictive models](#predictive-models) and [Releases](#releases).

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
invoke predict.matchday                 # last matchday of the current season, every model
invoke predict.matchday --number 12     # a given matchday
invoke predict.all                      # every matchday of the current season
invoke predict.all --model dixon_coles  # a single model (key from the registry)
```

Every registered model is run by default. Predictions are stored in
`data/predictions/<model>.parquet`. For every matchday the model is trained only on matches
played before it (all historical seasons plus the earlier matches of the current season), so
past matchdays are evaluated without look-ahead.

**Matchdays.** football-data.co.uk has no round numbers, so they are inferred
(`quiniela.data.matchdays`): each league's rounds come from the teams' match counts and dates,
and a quiniela matchday pairs a La Liga round with the Segunda round played the same week.
Rescheduled matches join the matchday they are played in, so a round can show 9 or 12 La Liga
matches.

**Models** (`quiniela.prediction`) follow the scikit-learn API (`fit`, `predict`,
`predict_proba`) and are registered in `quiniela.prediction.registry.MODELS`. Each version is
described in [Predictive models](#predictive-models).

### Backtest

```bash
invoke backtest.run                          # every model, 2015-16 to the last season
invoke backtest.run --start 2020 --end 2024  # a range of seasons (start years)
invoke backtest.run --model dixon_coles      # a single model
```

Every registered model predicts every matchday of the selected past seasons, walk-forward
(trained only on matches before each matchday), exactly as during a live season. The default
range, 2015-16 to 2025-26, has 9,262 matches, all with bookmaker odds for the market reference;
it runs in about two minutes using every CPU core. Predictions go to
`data/backtest/predictions.parquet` and the metrics are printed and shown in the evaluation
dashboard. See [Evaluation](#evaluation).

### Dashboards

```bash
invoke dashboards.build   # build dashboards/*.html from the downloaded data
invoke dashboards.open    # open them in the browser (macOS)

# Weekly refresh: new data, new predictions, rebuilt dashboards
invoke download.current predict.matchday dashboards.build

# After adding or changing a model: re-run the backtest too
invoke backtest.run predict.all dashboards.build
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
  Models that predict goals show their exact score and Pleno al 15 picks in the tooltip.
- `dashboards/evaluation.html`: backtest metrics of every model, one column per model in
  registry order (benchmark first, market last as reference), by league and season, plus
  calibration charts and an explanation of each metric.

All are self-contained HTML files (data embedded, no external libraries) that open with a
double click, also offline. New models appear as new columns automatically.

Standings are computed from the results: 2 points per win before 1995-96 and 3 points
afterwards. Head-to-head tiebreakers and point deductions are not applied, so a position can
occasionally differ from the official table.

### Command line

The tasks wrap the `quiniela` command, which can also be used directly:

```bash
quiniela download historical --start 2015 --end 2025
quiniela download current
quiniela predict --matchday 12   # or --all; --model selects one model (default: all)
quiniela backtest --start 2015   # --end, --model
quiniela dashboards
```

## Evaluation

Accuracy alone is a crude measure (it only looks at the most likely sign, and a few hundred
matches leave a margin of about ±3 points), so every model is evaluated on **probabilities**,
over many seasons, and against two references:

- **Always "1"**: predicting a home win in every match (45.3% hit rate in the backtest). The
  floor: a model that does not beat it adds nothing.
- **Market**: the bookmakers' own forecast, explained below. The ceiling.

### The market reference

"Market" is **not a model of ours**: it is the forecast hidden in the betting odds that
football-data.co.uk publishes with every match
([`quiniela/prediction/market.py`](src/quiniela/prediction/market.py)).

**From odds to probabilities.** A decimal odd $o$ pays $o$ times the stake, so it implies a
probability of $1/o$ (odds of 2.00 mean 50%). The three implied probabilities of a match add up
to slightly more than 100%; the excess is the **bookmaker's margin**. Normalising them removes
it:

```math
p_k = \frac{1/o_k}{1/o_1 + 1/o_X + 1/o_2}, \qquad k \in \{1, X, 2\}
```

For example, odds of 2.10 / 3.40 / 3.60 imply 47.6% + 29.4% + 27.8% = 104.8% (a 4.8% margin),
which normalise to 45.4% / 28.1% / 26.5%.

**Which odds.** Pre-match odds (not closing odds, so no information from the last hours before
kick-off is used). For each match, the first bookmaker with the three odds available, in this
order: market average across bookmakers (`Avg`, since 2019-20), Betbrain average (`BbAv`,
2005-2019), Pinnacle (`PS`), then single bookmakers available in older seasons (`B365`, `BW`,
`IW`, `WH`, `LB`). A match without any odds gets no prediction (21 of the 9,262 backtest
matches) and is left out of the market's metrics.

**Why it is the ceiling.** Odds summarise everything the betting market knows before the
match: team form, injuries, line-ups, news and the money of many bettors, and bookmakers lose
money if they are wrong. Getting close to the market means a model captures most of the
available information; beating it consistently is very hard. It is shown in every table as a
**reference**, not a candidate: it never counts as the best model (bold) and it is not used to
train any model.

### Metrics

With $N$ matches, $p_{ik}$ the probability given to outcome $k$ (1, X or 2) in match $i$, and
$o_{ik} = 1$ if that outcome happened (0 otherwise):

| Metric | Formula | Better | Meaning |
|---|---|---|---|
| Hit rate | $\frac{1}{N}\sum_i \mathbb{1}[\hat r_i = r_i]$ | higher | Share of right signs, the quiniela's own measure. |
| Log-loss | $-\frac{1}{N}\sum_i \log p_{i,r_i}$ | lower | Average surprise; heavily punishes confident mistakes. Uniform 1/3 scores $\log 3 = 1.099$. |
| Brier score | $\frac{1}{N}\sum_i \sum_k (p_{ik} - o_{ik})^2$ | lower | Squared distance between forecast and outcome (0 to 2). |
| RPS | $\frac{1}{N}\sum_i \frac{1}{2}\sum_{k=1}^{2}\left(\sum_{j \le k} p_{ij} - \sum_{j \le k} o_{ij}\right)^2$ | lower | Ranked probability score: Brier on cumulative probabilities, so 1, X, 2 are ordered and a draw is a smaller miss than the opposite result (0 to 1). |
| Calibration | predicted probability vs observed frequency, in 10% bins | on the diagonal | When the model says 60%, does it happen 60% of the times? |

Also reported:

| Metric | Meaning |
|---|---|
| X picks | Share of matches where the predicted sign is X (draw). See below. |
| Real draws | Share of matches that actually ended in a draw (28.3% in the backtest), to compare with X picks. |
| Exact score | Share of matches whose most likely score was the real one (models that predict goals). |
| Pleno al 15 | Share of matches where both teams' goal categories (0, 1, 2, M = 3+) were right. |
| Prize tickets | Matchdays whose 14-match ticket got 10 or more hits (5th category or better), out of the matchdays with a full ticket. |

### X picks

**X picks** is the share of matches in which a model's predicted sign is a draw:

```math
\text{X picks} = \frac{1}{N}\sum_i \mathbb{1}[\hat r_i = X]
```

It is not a quality metric (there is no "better" direction): it shows how a model handles
draws, next to **real draws**, the share of matches that actually ended level.

Every model picks far fewer X than there are draws (Dixon-Coles 1.2% and the market 0.5%,
against 28.3% real draws). This is expected, not a bug. The predicted sign is the **single most
likely** outcome, and a draw is rarely the most likely one even when it is quite probable. With
probabilities 1 = 40%, X = 32%, 2 = 28%, the draw is a real possibility, but "1" is still the
best single bet. Draws happen often, but they are spread across many matches where they are
never the favourite.

Picking more X does not help by itself either: the head-to-head benchmark picks X in 15.1% of
the matches and still has the lowest hit rate, because many of those draws do not happen. For
a well-calibrated model, choosing the most likely sign already maximises the expected number of
hits. A model that predicted X more often would only gain hits if it identified draws better,
for instance through better probabilities.

The metrics are implemented in
[`quiniela/evaluation/metrics.py`](src/quiniela/evaluation/metrics.py) and the backtest in
[`quiniela/evaluation/backtest.py`](src/quiniela/evaluation/backtest.py).

## Predictive models

Every model version lives in its own module, follows the scikit-learn estimator API (`fit`,
`predict`, `predict_proba`, `classes_` in H, D, A order) and is registered in
`quiniela.prediction.registry.MODELS`. Registered models are run by every prediction, backtest
and dashboard build, and each one is a new column in the dashboards, so versions can be
compared side by side.

**Backtest results** (walk-forward, seasons 2015-16 to 2025-26, 9,262 matches, both leagues;
bold = best candidate model):

| Version | Key | Status | Hit rate | Log-loss | Brier | RPS | X picks | Prize tickets |
|---|---|---|---|---|---|---|---|---|
| v1 · Head-to-head | `head_to_head` | Benchmark | 43.3% | 1.1155 | 0.6689 | 0.2261 | 15.1% | 14 / 376 |
| v2 · Dixon-Coles | `dixon_coles` | **Current** | **48.7%** | **1.0215** | **0.6130** | **0.2059** | 1.2% | **27 / 376** |
| Market | `market` | Reference | 50.0% | 1.0046 | 0.6013 | 0.2007 | 0.5% | 40 / 376 |

*Real draws: 28.3%. Always "1": 45.3%. Exact score and Pleno al 15 hit rates of v2: 14.1% and
14.5%.*

By league (hit rate):

| League | Always "1" | v1 | v2 | Market |
|---|---|---|---|---|
| La Liga | 45.7% | 47.0% | 52.9% | 53.9% |
| Segunda | 45.0% | 40.2% | 45.3% | 46.8% |

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
$\alpha$ pulls them towards the league prior: teams that never met get exactly the prior, and
with a long history the observed frequencies dominate.

**Training data.** Every earlier season (La Liga since 1993-94, Segunda since 1996-97) plus the
current season's matches played before the matchday being predicted.

**Observations.**

- Over 9,262 matches it is **below "always 1"** (43.3% vs 45.3%), a clear gap: with the first
  240 matches of 2026-27 the two could not be told apart.
- It is badly calibrated (see the evaluation dashboard): short head-to-head histories make its
  probabilities too extreme in both directions.
- It ignores current form and team strength: a meeting from 1995 weighs the same as last
  season's, and squads change every year.

### v2 · Dixon-Coles (`dixon_coles`)

[`quiniela/prediction/dixon_coles.py`](src/quiniela/prediction/dixon_coles.py) · based on
Dixon & Coles (1997), *Modelling association football scores and inefficiencies in the
football betting market*.

**Idea.** Instead of looking at past meetings, every team gets an **attack** and a **defence**
strength estimated from all its recent matches, adjusted for the strength of the opponents.
Goals are Poisson counts driven by those strengths, so the model predicts the whole score
distribution, from which the 1X2 probabilities, the exact score and the Pleno al 15 follow.

**Expected goals.** For a match of league $\ell$ between home team $h$ and away team $a$:

```math
\lambda = \exp(\mu_\ell + \eta_\ell + \alpha_h - \delta_a), \qquad
\nu = \exp(\mu_\ell + \alpha_a - \delta_h)
```

- $\lambda$, $\nu$: expected home and away goals.
- $\alpha_t$, $\delta_t$: attack and defence of team $t$ (higher is better). A team keeps the
  same strengths when it moves between La Liga and Segunda.
- $\mu_\ell$: goal level of the league (league adjustment: Segunda scores fewer goals).
- $\eta_\ell$: home advantage of the league.

**Score probabilities.** Independent Poisson goals with the Dixon-Coles correction $\tau$ for
low scores, governed by $\rho$:

```math
P(X = x, Y = y) = \tau_{\lambda,\nu}(x, y)\,
\frac{\lambda^x e^{-\lambda}}{x!}\,\frac{\nu^y e^{-\nu}}{y!},
\qquad
\tau_{\lambda,\nu}(x, y) =
\begin{cases}
1 - \lambda\nu\rho & x = y = 0 \\
1 + \lambda\rho & x = 0,\ y = 1 \\
1 + \nu\rho & x = 1,\ y = 0 \\
1 - \rho & x = y = 1 \\
1 & \text{otherwise}
\end{cases}
```

**Fitting with time decay.** The parameters maximise the time-weighted log-likelihood of past
matches, with a Gaussian prior (ridge penalty) on the strengths:

```math
\max_{\alpha, \delta, \mu, \eta, \rho} \;
\sum_k e^{-\xi t_k} \log P(x_k, y_k)
\;-\; \frac{1}{2\sigma^2} \sum_t \left(\alpha_t^2 + \delta_t^2\right)
```

- $t_k$: age of match $k$ in days. With $\xi = 0.0018$ a match loses half its weight in about
  a year ($\ln 2 / \xi \approx 385$ days); only the last four years are used.
- **Start of the season:** there is no special case. On matchday 1 all the weight comes from
  previous seasons; as the season advances its matches take over continuously.
- $\sigma = 0.5$: the prior keeps teams with few matches (newly promoted, new names) close to
  average. Unseen teams get average strength.

The optimisation uses L-BFGS with analytic gradients (about 0.1 s per fit).

**Predictions.** From the score matrix $P(x, y)$, $x, y \le 10$:

```math
P(1) = \sum_{x > y} P(x, y), \qquad P(X) = \sum_{x = y} P(x, y), \qquad P(2) = \sum_{x < y} P(x, y)
```

The predicted sign is the most likely one; the exact score is $\arg\max_{x,y} P(x, y)$; the
Pleno al 15 pick is the most likely pair of goal categories (0, 1, 2, M = 3+) after adding up
the matrix by category.

**Observations.**

- +5.4 points of hit rate over v1 and much better log-loss, Brier and RPS. It is well
  calibrated and closes most of the gap to the market (1.3 points of hit rate).
- It almost never picks X (1.2% of matches): even with good probabilities, a draw is rarely
  the single most likely outcome. The market behaves the same way (0.5%).
- The fitted $\rho$ is close to 0 in this data: the low-score correction barely changes the
  probabilities.

### Next steps

- **Pi-ratings** (Constantinou & Fenton, 2013): separate home and away ratings learned from
  goal differences.
- **Shots on target** (`HST`, `AST`) as a less noisy measure of performance than goals.
- **Learned combination** with the market odds (stacking).
- **Pleno al 15 and contrarian picks**: use the exact-score predictions and, if the bettors'
  sign percentages can be obtained, pick against the crowd in close matches.

## Releases

Each model version is released with a git tag in commitizen's `v<version>` format, using the
version in `pyproject.toml`:

| Tag | Models | Notes |
|---|---|---|
| [`v0.2.1`](https://github.com/jmquintana79/quiniela/releases/tag/v0.2.1) | v1 head-to-head, **v2 Dixon-Coles**, market reference | Calibrator and draw rule removed: in the backtest they did not improve the hit rate or the probabilities of v2. |
| [`v0.2.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.2.0) | v1 head-to-head, **v2 Dixon-Coles**, v2 + calibrator, market reference | Walk-forward backtest, probabilistic metrics, evaluation dashboard. |
| [`v0.1.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.1.0) | v1 head-to-head | Project structure, data download and first benchmark, evaluated on the current season only. |

## Project structure

```text
├── dashboards/              # generated HTML dashboards (not versioned)
├── data/                    # downloaded data (not versioned)
│   ├── historical/<code>/<season>.parquet   # e.g. SP1/2015-16.parquet
│   ├── current/<code>/results.parquet       # season in progress: played matches
│   ├── current/<code>/fixtures.parquet      # season in progress: upcoming matches
│   ├── predictions/<model>.parquet          # predictions for the current season
│   ├── backtest/predictions.parquet         # walk-forward predictions of past seasons
│   └── cache/                               # raw CSV files cached by soccerdata
├── src/quiniela/
│   ├── config/paths.py      # filesystem paths used by the project
│   ├── io/download.py       # historical and current season downloads
│   ├── data/                # match loaders and matchday inference
│   ├── prediction/          # model registry, models, pipeline, ticket rules
│   ├── evaluation/          # metrics and the multi-season backtest
│   ├── visualization/       # dashboard builder and HTML templates
│   └── scripts.py           # `quiniela` command line entry point
├── docs/api/                # generated API docs (not versioned)
├── tasks/                   # invoke tasks (download, predict, backtest, dashboards, dev)
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
`$$...$$` for display formulas (see `quiniela.prediction.dixon_coles`). Write those
docstrings as raw strings (`r"""..."""`) so backslashes are kept. Markdown runs before
MathJax, so:

- use `\lbrace ... \rbrace` for literal braces (Markdown removes the backslash of `\{`);
- write row breaks of `cases` or `aligned` as `\\\\` (Markdown turns `\\` into `\`);
- do not start a line of a formula with `-` or `+`, or Markdown reads it as a list item.

MathJax is loaded from a CDN, so viewing the formulas needs an internet connection.

```bash
invoke dev.docs           # writes docs/api/ (not versioned)
open docs/api/index.html  # macOS

invoke dev.docs --serve   # live server that rebuilds on every change
```

pdoc documents the public members of every module. In packages that define `__all__`
(`quiniela.config`, `quiniela.prediction`, `quiniela.evaluation`), only the names listed
there appear, so a new
submodule has to be added to `__all__` to show up in the docs.

### Commits

[commitizen](https://commitizen-tools.github.io/commitizen/) is configured in
`pyproject.toml` for [Conventional Commits](https://www.conventionalcommits.org) messages
and version bumps (`cz commit`, `cz bump`).

## Background

This repository started in 2021 as a scraper for a Spanish football results dataset (see
`docs/references.md`). It has been rebuilt as a Poetry-managed project on top of
football-data.co.uk, which provides historical data and weekly updates from a single source.
