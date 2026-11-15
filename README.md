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

### Official draws: real tickets, winners and prizes

The real quiniela tickets, takings, winners and prize of every category come from the results
service of Loterías y Apuestas del Estado (LAE), the JSON its website uses, one request per
season ([`quiniela/data/prizes.py`](src/quiniela/data/prizes.py)):

```text
https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAQU&celebrados=true&fechaInicioInclusiva=20250801&fechaFinInclusiva=20260731
```

Each draw has the 15 matches of the real ticket (teams, kick-off, sign and score), the bets and
takings, the winners and prize per winner of every category and the jackpot announced for the
Pleno al 15. The history is complete from 2010-11 (about 55 to 76 draws per season).

**Saving it: one click, from `invoke update`.** The service only answers web browsers (scripts
get "Access Denied", and the project does not try to get around it), so the files are saved by
your own browser:

1. **Once**: open `dashboards/bookmarklet.html` (written by `invoke update` or
   `invoke dashboards.build`) and drag its **Quiniela** button to the bookmarks bar. It is
   [`tools/lae_bookmarklet.js`](tools/lae_bookmarklet.js).
2. **Every `invoke update`** checks what is missing (`quiniela.io.lae`): the next ticket (none
   saved, or the saved one already closed) and the season's draws (no file, or the draw of a
   closed ticket missing a day and a half after its deadline, when its matches are over). If
   something is missing, it **opens the LAE pages in your browser** and waits: **click the
   Quiniela bookmark** on each page. On the results page (this URL, one per season, 1 August to
   31 July) it saves the season's draws; on the betting page, the next ticket.
3. `invoke update` **picks the files up from your Downloads folder** (`quiniela_lae_<season>.json`
   and `quiniela_upcoming_j<jornada>.json`), moves them to `data/lae/` (`lae_<season>.json`,
   `upcoming.json`) and carries on: predictions, Liga F history, dashboards. Press Enter to go on
   without them; it stops waiting after five minutes.

`invoke update --no-browser` (or running it without a terminal) never opens the browser and
only warns about what is missing. Older seasons were saved the same way, once; any file name
works in `data/lae/` and repeated draws are dropped.

**What it shows** (checked on every draw since 2015-16):

| Rule | Value |
|---|---|
| Price of a bet (column) | 0.75 € (the minimum ticket is two columns, 1.50 €) |
| 14 hits (1st) | 16% of the takings |
| 13, 12 and 11 hits (2nd to 4th) | 7.5% each |
| 10 hits (5th) | 9%, not paid when it would be below 1.50 € (5 to 27 draws per season) |
| Pleno al 15 (14 hits + 15th match) | 7.5% plus the announced jackpot (weekend draws) |

Every saved draw is listed in [`prizes.html`](#dashboards): its jackpot and, for each category,
the prize per winner and the number of winners, with the distribution of the prize per winner of
each category (median 112,158 € for 14 + Pleno al 15, 51,115 € for 14, 1,999 € for 13, 135 € for
12, 18 € for 11 and 5.71 € for 10 hits, 2010-11 to 2026-27).

**The next ticket.** The matches of the next ticket, its sales deadline and the public's share
of each sign ("Pronósticos más apostados") are on the official betting page,
`https://juegos.loteriasyapuestas.es/jugar/la-quiniela/apuesta`, days before football-data
publishes the fixtures. `invoke update` opens it when the saved ticket is missing or closed, and
one click on the Quiniela bookmark saves it (see above); every ticket saved is also archived by
season and jornada in `data/lae/public/` (the history of the public's picks, see
[Liga F](#liga-f-womens-league)). `current.html` then shows that ticket (see
[Dashboards](#dashboards)). The public's shares are v3's prediction for the matches no model
covers (see [v3](#v3--dixon-coles--market-market_blend)).

Not every ticket is La Liga and Segunda: of 755 draws between 2015-16 and 2025-26, about 330
have their 14 matches in those leagues and some 70 more have 10 to 13. The rest are midweek
draws (Champions League, Copa del Rey, national teams) and summer draws (Nordic leagues, World
Cup). Since 2026-27 weekend tickets include four women's league matches. Team names are
matched with football-data's through an alias table (e.g. `At. Madrid` → `Ath Madrid`) and
the date of the match.

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

### Update everything (recommended)

At the start of every matchday, or after weeks without running anything:

```bash
invoke update               # data, predictions and dashboards
invoke update --backtest    # also re-run the backtest (after changing a model)
```

It runs, in order ([`quiniela/update.py`](src/quiniela/update.py)):

| Step | What it does | Avoids repeated work |
|---|---|---|
| 1. Missing seasons | Downloads every completed season (per league) that is not in `data/historical/`. Covers the change of season (the season that just ended joins the history), a first installation and gaps. | Seasons already downloaded are skipped; soccerdata also caches their raw files. |
| 2. Current season | Downloads the results played so far and the upcoming fixtures. | Always downloaded: the source does not tell whether its files changed (a few small CSV files). |
| 3. Liga F | Downloads the missing completed Liga F seasons and refreshes the current one (results and fixtures) from ESPN ([Liga F](#liga-f-womens-league)). A failure is a warning. | Completed seasons are skipped; months are cached in `data/cache/espn/`. |
| 4. LAE | When the next ticket or a played draw is missing, opens the LAE pages in your browser; one click on the Quiniela bookmark saves each one ([official draws](#official-draws-real-tickets-winners-and-prizes)). | Only what is missing; `--no-browser` only warns. |
| 5. Predictions | Predicts every matchday of the current season with every model, walk-forward. Covers skipped weeks and replaces last season's predictions when a new one starts. | Local data only, a few seconds. |
| 6. Backtest | Only with `--backtest`: every model over past seasons, plus the real tickets in euros when the LAE draws are saved (about ten minutes). | Not run by default. |
| 7. Dashboards | Rebuilds the five dashboards, with the Liga F history and evaluation (v2F walk-forward, about half a minute). | Local data only. |

Special cases:

- **No connection or source down**: if the current season cannot be downloaded, the update
  stops before touching predictions and dashboards, so old and new data are never mixed.
- **Off-season (July and August)**: the new season's files may not be published yet. The
  failed download is then reported as a warning, predictions are skipped and the dashboards
  are rebuilt.
- **No matches to predict**: predictions are skipped with a warning.

The steps can also be run one by one with the tasks below.

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
`data/predictions/<model>.parquet`, with the probabilities, the predicted sign and its
[confidence and margin](#confidence-of-a-prediction). For every matchday the model is trained only on matches
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
range, 2015-16 to 2025-26, has 9,262 matches, all but 21 with bookmaker odds; the models take
about four minutes using every CPU core (v3 most of it). Predictions go to
`data/backtest/predictions.parquet` and the metrics are printed and shown in the evaluation
dashboard. Each matchday's 14-match ticket is also replayed with doubles and triples (no
doubles, random, by eye, optimal and [v3+v1F·D](#v3v1fd--direct-ticket-planner)); those tickets go to
`data/backtest/tickets.parquet` and their prizes are printed after the metrics. With the
[official draws](#official-draws-real-tickets-winners-and-prizes) saved in `data/lae/`, the
real tickets are then played in euros ([v3+v1F·E and v3+v1F·B](#v3v1fe-and-v3v1fb--money-tickets-money), about five more minutes)
into `data/backtest/money.parquet` (the big-win mode makes this the slowest part: some fifteen
minutes or more). See [Evaluation](#evaluation).

### Dashboards

```bash
invoke dashboards.build   # build dashboards/*.html from the downloaded data
invoke dashboards.open    # open them in the browser (macOS)

# Weekly refresh: new data, new predictions, rebuilt dashboards
invoke update

# After adding or changing a model: re-run the backtest too
invoke update --backtest
```

Five dashboards, each answering one question:

| Dashboard | Question | Content |
|---|---|---|
| `current.html` | What happens this week? | Current matchday (with confidence, prize chances, the v3+v1F·D and v3+v1F·R tickets and the v3+v1F·B big-win ticket) and standings. |
| `season.html` | How are the models doing this season? | Every matchday of the current season. |
| `historical.html` | What happened in past seasons? | Results, standings and outcome shares since 1993-94. |
| `evaluation.html` | Which model is better? | Backtest metrics over past seasons, confidence tiers, tickets with doubles and triples and real tickets in euros. |
| `prizes.html` | What did each draw pay? | Every official draw since 2010-11: jackpot, winners and prize per winner of each category. |

**In every dashboard**, a **table of contents** after the header links to each section, and the
models' figures (metrics, hits, hit rates) come **twice, apart**: one block for the **men's
leagues** (La Liga and Segunda, the v1-v4 models) and one for the **Liga F** (the v1F and v2F
models and their reference), each with the history it has (see [Liga F](#liga-f-womens-league)):

| Dashboard | Men's leagues | Liga F |
|---|---|---|
| `current.html` | Footer hit rate of the men's matches (and other competitions) | **Public picks v1F** and **Pi-ratings v2F** columns and their own footer hit rate |
| `season.html` | Season totals and matchdays | Season block: v1F, v2F and reference hits, every Liga F ticket match with the v1F and v2F picks |
| `evaluation.html` | Metrics, hit rate by result, contingency tables, confidence, calibration | The same tables for v1F, v2F and the reference, **v1F against v2F** on the common matches, the data by season and every ticket match |
| `historical.html` | Outcomes, standings and results | Outcomes and goals by season (ESPN) and every Liga F ticket match |

The tickets (v3+v1F·D, ·R, ·B) mix both leagues and have their own block.

- `dashboards/current.html`: the **current matchday**. When the next official ticket is saved
  (`data/lae/upcoming.json`, see [the next ticket](#official-draws-real-tickets-winners-and-prizes))
  and its sales closed less than four days ago, the current matchday is **that ticket**, in its
  official order: the sales deadline (Madrid time, in large type) at the very top, its La Liga and
  Segunda matches predicted by every model (trained on all played matches; the other matches,
  women's league or national teams, have no prediction and count with the league shares of 1,
  X and 2 for doubles and triples), no colours until there are results, and a notice saying
  whether **football-data's fixtures are available**: until they are (Friday afternoon for
  weekend matches), there are no odds and v3 predicts with Dixon-Coles only. Otherwise (no saved
  ticket, or an old one) the current matchday comes from the football-data matches. The
  official ticket also shows its **Pleno al 15**: one pick of the goals categories (0, 1, 2, M)
  of each team, the same on the v3, v3+v1F·D, v3+v1F·R and v3+v1F·B tickets: Dixon-Coles' most likely pair when it is
  a La Liga / Segunda match, otherwise the categories the public marks most. The v3+v1F·D, v3+v1F·R and v3+v1F·B
  banners give the **price of their ticket** (0.75 € × 2^doubles × 3^triples bets, or the bets
  of the reduction times its direct doubles and triples for v3+v1F·R), the
  table's footer gives every model's hits on the 14 ticket matches, and under the Pleno al 15 a
  table gives the price of each ticket (14 matches and the Pleno, which adds no bets). Before the
  draw, Loterías does not publish the date of the women's and other matches (only football-data
  has the La Liga / Segunda kick-offs), so they show none until the draw is saved. Without a
  saved ticket, the current matchday comes from the football-data matches: its
  matches (kick-off, teams, result when played) with one column per model in registry order. The **market** reference is not
  shown here: v3 follows the odds closely, so it would repeat v3's signs; it stays in
  `evaluation.html`, where it is the benchmark. Each cell shows only the predicted sign, green with ✓ when right and red
  with ✗ when wrong, so the table has room for more models; hovering it shows the probability
  (and, for models that predict goals, the exact score and Pleno al 15 picks). The footer gives
  each model's hit rate on the played
  matches. A **Confidence** column shows how predictable each match is according to v3 (the
  probability of its sign, green high, amber medium, red low; the margin over the second sign
  on hover), and the table can be sorted in ticket order or **most predictable first**. The
  **Direct ticket v3+v1F·D** column shows the v3 signs with the doubles (`1X`) and triples (`1X2`)
  chosen by the [ticket planner](#v3v1fd--direct-ticket-planner), the **Reduced ticket v3+v1F·R**
  column the same budget with an [official reduction](#v3v1fr--reduced-ticket-reduced) (its matches
  marked R), side by side, and the **Big-win ticket v3+v1F·B**
  column the **big-win ticket** of [v3+v1F·E and v3+v1F·B](#v3v1fe-and-v3v1fb--money-tickets-money) (only with the LAE draws
  saved). Above the table, a status banner (upcoming, in progress or completed) and the
  14-match quiniela ticket outcome of each model, with the **chance of each prize before the
  matchday** (10+ to 14 hits); the v3+v1F·B banner adds a **chart of what can happen with its ticket**
  (chance of each payout range, from nothing back to 100,000 € or more), its cost, the chance of
  10,000 € or more and how much comes back on average, so the decision to play is yours; once the
  matchday is played, the v3, v3+v1F·D, v3+v1F·R and v3+v1F·B banners give the **euros spent and won** (real prizes of
  the official draw when its ticket matches ours, otherwise an estimate with the prize model).
  Below, the standings of each league with
  recent form. The current matchday is the first one with matches still to play; when every
  published match has been played (e.g. during an international break, before the source
  publishes the next fixtures) the last one is shown and marked as completed. A note says when
  the odds used by v3 were downloaded (they keep moving until kick-off).
- `dashboards/season.html`: the **current season**: totals per model and for the v3+v1F·D, v3+v1F·R and
  v3+v1F·B tickets (hit rate, prize tickets and, for v3, v3+v1F·D, v3+v1F·R and v3+v1F·B, euros spent and won; without
  the market), a table of every matchday with each model's hits and quiniela outcome (and, for
  v3, v3+v1F·D, v3+v1F·R and v3+v1F·B, the prize won in euros and the ticket's cost), and the detail of any matchday (same table as `current.html`). The ticket is approximated with the
  matchday's La Liga matches followed by Segunda matches, up to 14 (14 hits 1st, 13 2nd,
  12 3rd, 11 4th, 10 5th category; the Pleno al 15 is not simulated). Models that predict
  goals show their exact score and Pleno al 15 picks in the tooltip.
- `dashboards/historical.html`: home win, draw and away win shares and goals per match for
  every season, plus the final standings and results of the selected season (filterable by
  team), each match with its category and whether it was on an official ticket; and the
  **official quiniela tickets** of the season, any jornada, with all their matches (men's La
  Liga and Segunda, women's football and other competitions) and results.
- `dashboards/evaluation.html`: first the backtest **metrics** of every model (with
  **macro-F1** as an informative row), then the **hit rate by result** (1, X, 2; a double or
  triple of a ticket is a hit for every result it covers), then the **contingency tables**
  (observed result × predicted sign) of every model and of the v3 tickets (observed result ×
  marked signs, on the official tickets), and then the rest. The metrics have one column per
  model in registry order (benchmark first, market last as reference), by league and season,
  with a **v3+v1F·D** and a **v3+v1F·R** column after the models that show only their hit rate and prizes (they play
  v3's probabilities, so the probability metrics do not apply), plus the hit rate by confidence
  tier, the prizes of the tickets played with doubles and triples (no
  doubles, random, by eye, optimal, v3+v1F·D and v3+v1F·R), the **real tickets in euros** (single column,
  v3+v1F·D, v3+v1F·R, v3+v1F·E, v3+v1F·E every draw and v3+v1F·B: spent, won, net, return and best prize), the **official ticket matches by
  category** (how many were M, F, O or U each season, and v3's hit rate on the M ones),
  calibration charts and an explanation of each metric.
- `dashboards/prizes.html`: every official draw saved in `data/lae/`, newest first, one row
  per draw with its season, jornada, date and whether it had a jackpot (and how much), and
  for each category (14 + Pleno al 15, 14, 13, 12, 11 and 10 hits) the prize each winner got
  and the number of winners (a 10-hit prize under 1.50 € shows as *not paid*). Below the table,
  linked from the top, one density chart per category of the prize per winner over every paid
  draw, all on the same logarithmic euro axis (1 € to 15 M €), with the median and the range of the middle half.

**Match categories and the official ticket.** Every match table (current, season and
historical) shows two columns, with a legend:

| Column | Values |
|---|---|
| Cat. | **M** (male): men's La Liga or Segunda, in football-data, the matches the models predict · **F** (female): women's football, marked (F) on the official ticket · **O** (other): another competition (national teams, cups, foreign or lower leagues) · **U** (unknown): not enough information |
| Quiniela | **Q N**: the match was on the official ticket of jornada N |

football-data matches are M; the official tickets saved from LAE add the F and O ones. LAE only
marks women's teams (since 2023-24) and men's teams (since 2026-27), so an older unmatched match
counts as O. `current.html` only shows v3 and the tickets built on it (v3+v1F·D, v3+v1F·R and v3+v1F·B); every
model is in `season.html`, which also lists the official tickets of the season with what every
model marked: v1, v2 and v3 their sign on the La Liga / Segunda matches, v3+v1F·D, v3+v1F·R and v3+v1F·B their
doubles and triples on the 14 matches (R marks the matches of v3+v1F·R's reduction) (on the v3 probabilities, the league shares of 1, X and 2 where no
model predicts: the public's picks of past tickets were not saved), the Pleno al 15 pick
(Dixon-Coles) and each ticket's hits and price.

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
quiniela update                  # --backtest
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
**reference**, not a candidate: it never counts as the best model (bold). Since v3 the odds are
also an input of a model (see [v3](#v3--dixon-coles--market-market_blend)); v2 stays
independent of them.

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
| Hit rate by result | For each real result $k$ (1, X, 2): $\frac{\#\{i : r_i = k,\; r_i \in \hat S_i\}}{\#\{i : r_i = k\}}$, with $\hat S_i$ the sign or signs marked. A double or triple of a ticket is a hit for every result it covers. See below. |
| Macro-F1 | $\frac{1}{3}\sum_k \frac{2\,\text{hits}_k}{\text{predicted}_k + \text{real}_k}$: the mean over 1, X and 2 of the F1 score of the predicted sign. **Informative only** (never bold, never used to choose a model). See below. |
| X picks | Share of matches where the predicted sign is X (draw). See below. |
| Real draws | Share of matches that actually ended in a draw (28.3% in the backtest), to compare with X picks. |
| Exact score | Share of matches whose most likely score was the real one (models that predict goals). |
| Pleno al 15 | Share of matches where both teams' goal categories (0, 1, 2, M = 3+) were right. |
| Prize tickets | Matchdays whose 14-match ticket got 10 or more hits (5th category or better), out of the matchdays with a full ticket. |

### Hit rate by result

How often each model gets the matches of every real result right (both leagues, 2015-16 to
2025-26). For **v3+v1F·D** and **v3+v1F·R** a match is a hit when its result is among the marked signs,
so a triple (1X2) is a hit whether it ends 1, X or 2; outside the ticket they are v3's sign:

| Observed result | v1 | v2 | v3 | v4 | v3+v1F·D | v3+v1F·R | Market |
|---|---|---|---|---|---|---|---|
| **1** (45.3% of the matches) | 72.7% | 88.4% | 87.0% | 89.8% | 89.8% | **90.8%** | 87.0% |
| **X** (28.3%) | 15.7% | 1.5% | 3.7% | 0.1% | 21.0% | **32.4%** | 0.7% |
| **2** (26.4%) | 22.3% | 31.2% | 36.2% | 31.2% | 42.9% | **46.1%** | 39.5% |
| All results | 43.3% | 48.7% | 50.0% | 49.0% | 57.9% | 62.5% | 50.0% |

- **The draw is where the tickets win.** Single-sign models get almost no draws (v3 3.7%, the
  market 0.7%): a draw is rarely the most likely result. The doubles 1X and the triples of the
  tickets turn many of them into hits: 21% for v3+v1F·D and 32% for v3+v1F·R, which covers more matches
  with the same 48 bets.
- **Away wins are the other gap.** Models get only a third of them (v3 36%); v3 and the market
  do better than the goal models (v2 and v4 31%) because the odds know when the away team is the
  stronger one.
- **Home wins are easy for everyone** (87-90%) except v1, which spreads its signs and pays for
  it here.

It is informative, like macro-F1: the overall hit rate and the probability metrics choose the
model.

### Macro-F1 and contingency tables

The **contingency table** of a model counts the matches of every **observed** result (rows:
what really happened) against the **predicted** sign (columns: what the model marked). Its
diagonal are the hits. Over the 9,262 backtest matches (2015-16 to 2025-26, both leagues):

| Observed ↓ · Predicted → | v1 (1 / X / 2) | v2 (1 / X / 2) | v3 (1 / X / 2) | v4 (1 / X / 2) | Market (1 / X / 2) |
|---|---|---|---|---|---|
| **1** (4,197) | 3,051 / 636 / 510 | 3,711 / 42 / 444 | 3,650 / 82 / 465 | 3,771 / 2 / 424 | 3,640 / 12 / 534 |
| **X** (2,618) | 1,817 / 410 / 391 | 2,123 / 39 / 456 | 2,010 / 97 / 511 | 2,172 / 2 / 444 | 1,997 / 18 / 603 |
| **2** (2,447) | 1,548 / 354 / 545 | 1,651 / 33 / 763 | 1,478 / 84 / 885 | 1,681 / 3 / 763 | 1,458 / 16 / 963 |

*The market is counted on the 9,241 matches with odds.*

They show why the draw is the quiniela's hard sign: every model except v1 almost never marks
X, so most of the 2,618 real draws fall in the 1 column (a home win was always more likely),
and the 2 column catches most of the rest.

**Macro-F1** summarises a contingency table giving the three signs the same weight: for each
sign, $F1_k = 2\,\text{hits}_k / (\text{predicted}_k + \text{real}_k)$ (a sign never
predicted scores 0), and their mean. It is **informative**: a quiniela pays every hit the same,
so hit rate and the probability metrics choose the model; macro-F1 only shows how evenly a
model spreads its signs.

| | v1 | v2 | v3 | v4 | Market |
|---|---|---|---|---|---|
| Hit rate | 43.3% | 48.7% | 50.0% | 49.0% | 50.0% |
| Macro-F1 | 0.353 | 0.345 | **0.374** | 0.338 | 0.361 |

v3 has the highest macro-F1 too: it marks more draws than the market (2.8% against 0.5%) and
gets a share of them right. v1 marks many more X (15%) but misses most of them, and v4, which
almost never marks X, has the lowest.

The evaluation dashboard shows these tables for every model (with the league and season
filters) and, for the **v3 tickets**, the same table on the official tickets of the
[euro backtest](#v3v1fe-and-v3v1fb--money-tickets-money) with the signs each ticket marked as columns
(1, X, 2, the doubles 1X, 12, X2 and the triple 1X2):

| Ticket | Observed | 1 | X | 2 | 1X | 12 | X2 | 1X2 | Covered |
|---|---|---|---|---|---|---|---|---|---|
| One column (v3 signs) | 1 | **2,232** | 39 | 296 | **0** | **0** | 0 | **0** | 49.6% |
|  | X | 1,265 | **54** | 331 | **0** | 0 | **0** | **0** |  |
|  | 2 | 946 | 42 | **591** | 0 | **0** | **0** | **0** |  |
| v3+v1F·D | 1 | **1,664** | 0 | 147 | **423** | **123** | 69 | **141** | 63.1% |
|  | X | 827 | **2** | 168 | **314** | 102 | **90** | **147** |  |
|  | 2 | 550 | 0 | **369** | 273 | **152** | **108** | **127** |  |
| v3+v1F·R | 1 | **1,359** | 0 | 87 | **609** | **82** | 81 | **349** | 71.2% |
|  | X | 590 | **1** | 113 | **444** | 69 | **98** | **335** |  |
|  | 2 | 374 | 0 | **273** | 356 | **96** | **124** | **356** |  |
| v3+v1F·B | 1 | **1,537** | 10 | 82 | **623** | **164** | 150 | **1** | 62.9% |
|  | X | 735 | **22** | 127 | **457** | 157 | **150** | **2** |  |
|  | 2 | 499 | 14 | **288** | 374 | **209** | **195** | **0** |  |

*414 official tickets, 5,796 matches (14 per ticket, also the women's and other matches,
planned with the league shares of 1, X and 2). Bold: the observed result is among the marked
signs. Covered: share of the matches whose result was marked; for v3+v1F·R a covered result in a
reduced match is not always a hit of its best column.*

The tickets put their doubles mostly as 1X, on the open matches the single column would give to
the home side, so most draws move from a miss in the 1 column to a hit in the 1X column; v3+v1F·R
adds triples (1X2) and covers 71% of the results against 63% for v3+v1F·D, with the same 48 bets.

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

### Confidence of a prediction

Every prediction comes with two measures of how predictable the match is, computed from its
1X2 probabilities alone, so they exist for every model
([`quiniela/prediction/confidence.py`](src/quiniela/prediction/confidence.py)). With the
probabilities of a match sorted as $p_{(1)} \ge p_{(2)} \ge p_{(3)}$:

| Measure | Formula | Meaning |
|---|---|---|
| Confidence | $p_{(1)}$ | Probability of the predicted sign. For a calibrated model it is literally the chance of hitting the match with one sign. |
| Margin | $p_{(1)} - p_{(2)}$ | How clear the favourite is over the second option (45 / 40 / 15 has a small margin). |

Matches are grouped in three tiers: **high** (60% or more), **medium** (45% to 60%) and **low**
(below 45%). A confidence is only useful if it keeps its promise: the hit rate of each tier
should be close to its average confidence. In the backtest (hit rate, share of matches):

| Tier | v1 | v2 | v3 | Market |
|---|---|---|---|---|
| High (≥ 60%) | 51.5% · 23% | 73.2% · 11% | 70.8% · 18% | 74.5% · 14% |
| Medium (45-60%) | 40.8% · 44% | 50.8% · 41% | 51.6% · 38% | 53.3% · 38% |
| Low (< 45%) | 40.7% · 32% | 41.2% · 48% | 40.1% · 44% | 40.3% · 48% |

For v3 the average confidence of the high tier is 69.9% and it hits 70.8% of those matches:
its confidence can be read at face value. v1's cannot: it says 68% and hits 51%, because its
probabilities are too extreme. Those probabilities, and so the confidence, drive the
doubles and triples of [v3+v1F·D](#v3v1fd--direct-ticket-planner).

The metrics are implemented in
[`quiniela/evaluation/metrics.py`](src/quiniela/evaluation/metrics.py) and the backtest in
[`quiniela/evaluation/backtest.py`](src/quiniela/evaluation/backtest.py).

## Predictive models

Every model version lives in its own module, follows the scikit-learn estimator API (`fit`,
`predict`, `predict_proba`, `classes_` in H, D, A order) and is registered in
`quiniela.prediction.registry.MODELS`. Registered models are run by every prediction, backtest
and dashboard build, and each one is a new column in the dashboards, so versions can be
compared side by side.

**Notation: model · ticket.** Men's models are numbered (v1, v2, v3, v4...) and **Liga F
(women's league) models have their own counter with an F** (v1F, v2F...), with their own data and
evaluation (see [Liga F](#liga-f-womens-league)). The ways of playing the probabilities on the
quiniela ticket are not models but decision layers, written as the models they use plus a letter,
so new models and new tickets can be added independently. A ticket mixes both leagues, so it uses
**v3 on the men's matches and v1F on the Liga F ones**:

| Ticket | Letter | Module | Current | v0.9.x | Before v0.9.0 |
|---|---|---|---|---|---|
| Direct doubles and triples, best chance of 11+ hits within 48 bets | **·D** | `planner` | **v3+v1F·D** | v3·D | v4, v4D |
| The same budget, also with the official reductions | **·R** | `reduced` | **v3+v1F·R** | v3·R | v4R |
| Big win: best chance of 10,000 € or more within 50 € | **·B** | `money` | **v3+v1F·B** | v3·B | v5 (big-win mode) |
| Best expected prize in euros, played only when it pays (backtest only) | **·E** | `money` | **v3+v1F·E** | v3·E | v5 (prudent) |

The tickets plan on v3's probabilities (`PLANNER_MODEL`) for the men's matches and on v1F's
(`WOMEN_PLANNER_MODEL`) for the Liga F ones; the other competitions (national teams, cups) use
the public's picks too, and every match without them the shares of 1, X and 2. This is what the
tickets already did; v0.10.0 names it. Release notes keep the names of their time.

**Backtest results** (walk-forward, seasons 2015-16 to 2025-26, 9,262 matches, both leagues;
bold = best candidate model):

| Version | Key | Status | Hit rate | Log-loss | Brier | RPS | X picks | Prize tickets |
|---|---|---|---|---|---|---|---|---|
| v1 · Head-to-head | `head_to_head` | Benchmark | 43.3% | 1.1155 | 0.6689 | 0.2261 | 15.1% | 14 / 376 |
| v2 · Dixon-Coles | `dixon_coles` | Independent | 48.7% | 1.0215 | 0.6130 | 0.2059 | 1.2% | 27 / 376 |
| v3 · DC + Market | `market_blend` | **Current** | **50.0%** | **1.0028** | **0.6002** | **0.2003** | 2.8% | **35 / 376** |
| v4 · Pi-ratings | `pi_ratings` | Independent | 49.0% | 1.0178 | 0.6104 | 0.2048 | 0.1% | 30 / 376 |
| v3+v1F·D · Direct ticket | `planner` | Way of playing v3 (≤ 48 bets) | 57.9% | – | – | – | – | 156 / 376 |
| v3+v1F·R · Reduced ticket | `reduced` | Way of playing v3 (≤ 48 bets) | 62.5% | – | – | – | – | **191 / 376** |
| Market | `market` | Reference | 50.0% | 1.0046 | 0.6013 | 0.2007 | 0.5% | 40 / 376 |

*v3+v1F·D and v3+v1F·R are not probability models: they mark the v3 signs plus doubles and triples on the
ticket, so a match is a hit when its result is among the marked signs, and their probability
metrics would be v3's. They cost up to 48 bets a matchday, against 1 for the others. v3+v1F·R's
prize tickets count the hits of its best column (a reduction does not play every combination).*

*Real draws: 28.3%. Always "1": 45.3%. Exact score and Pleno al 15 hit rates of v2: 14.1% and
14.5%. The market is evaluated on the 9,241 matches with odds; v3 on all 9,262 (Dixon-Coles
alone in the other 21).*

By league (hit rate / log-loss):

| League | Always "1" | v1 | v2 | v3 | v4 | v3+v1F·D (hit rate) | v3+v1F·R (hit rate) | Market |
|---|---|---|---|---|---|---|---|---|
| La Liga | 45.7% | 47.0% / 1.0738 | 52.9% / 0.9784 | 53.5% / 0.9622 | 52.8% / 0.9779 | 63.3% | 70.3% | 53.9% / 0.9637 |
| Segunda | 45.0% | 40.2% / 1.1499 | 45.3% / 1.0570 | 47.1% / 1.0362 | 45.9% / 1.0507 | 53.5% | 56.0% | 46.8% / 1.0384 |

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

### v3 · Dixon-Coles + market (`market_blend`)

[`quiniela/prediction/market_blend.py`](src/quiniela/prediction/market_blend.py)

**Idea.** Dixon-Coles only sees past goals; the odds also reflect injuries, line-ups, news and
the money of many bettors. v3 learns how much to trust each forecast with a **log-linear pool**,
a multinomial logistic regression with one weight per forecast:

```math
z_k = a \log p^{\mathrm{DC}}_k + b \log p^{\mathrm{M}}_k + c_k, \qquad
P(k) = \frac{e^{z_k}}{\sum_{k'} e^{z_{k'}}}, \qquad k \in \{1, X, 2\}
```

or, equivalently, $P(k) \propto (p^{\mathrm{DC}}_k)^a \, (p^{\mathrm{M}}_k)^b \, e^{c_k}$, where:

- $p^{\mathrm{DC}}_k$: probability of outcome $k$ according to Dixon-Coles (v2).
- $p^{\mathrm{M}}_k$: market probability (pre-match odds with the margin removed, as in the
  [market reference](#the-market-reference)); never closing odds.
- $a$, $b$: weights of each forecast. $a + b > 1$ sharpens the probabilities, $a + b < 1$
  flattens them.
- $c_k$: intercept per outcome ($c_1 = 0$), correcting a systematic bias towards draws or away
  wins.

The four parameters maximise the log-likelihood of the training results (L-BFGS with
analytic gradients).

**Out-of-sample training.** The pool must learn from forecasts made as in a real matchday: a
Dixon-Coles fitted on the same matches it is judged on would look better than it is. So, at
every fit, Dixon-Coles is run walk-forward over the **last five years** of the training data,
refitted every week on the matches before that week and used to predict the week. Those
out-of-sample probabilities and the odds of the same matches (about 4,200) train the pool.
Every weekly Dixon-Coles depends only on the matches before it, so it is cached and reused by
the later matchdays of the backtest.

**Without odds.** A match without odds (e.g. a matchday whose odds are not published yet) gets
the Dixon-Coles probabilities unchanged, exactly v2.

**Without data: the public's picks** (since v0.6.0). The official ticket also has matches no
model covers: women's football and other competitions (national teams, cups...). For them v3
uses the share of bettors marking each sign on the betting page ("Pronósticos más apostados",
saved in `data/lae/upcoming.json`), normalised:

```math
p_k = \frac{s_k}{s_1 + s_X + s_2}, \qquad k \in \{1, X, 2\}
```

So v3 is: **Dixon-Coles + market** where there are odds, **Dixon-Coles** where there are none
yet, and **the public's picks** where there is no data at all. v3+v1F·D, v3+v1F·R and v3+v1F·B plan their doubles and
triples on these probabilities, so all 14 matches of the ticket count with an estimate (before,
the matches without data got the league shares of 1, X and 2). The public's shares only exist for
the ticket being sold (the betting page shows them; the results service does not keep them), so
past tickets keep the league shares for those matches and the backtests do not change: the rule
applies from the first ticket saved with the Quiniela bookmark on.

**In practice.** v2 only does the work when there are no odds: v3 then uses v2 as it is. When
the odds are available, the forecast is essentially the market's, slightly corrected (see the
weights in the observations below): Dixon-Coles adds nothing the market does not already know.

**When to run it: the odds make the difference.** v3 is only better than v2 when the odds of
the matchday are there; without them it is v2. football-data.co.uk collects the odds of
weekend matches on **Friday afternoon** and of midweek matches on **Tuesday afternoon**, so the
moment `invoke update` runs decides which model is really predicting. A typical week, for a
matchday played Friday to Monday:

| Day | What happens | v3 predicts with |
|---|---|---|
| Monday-Tuesday | The previous matchday ends; LAE publishes the next ticket. | Dixon-Coles only (no odds yet) |
| Tuesday-Thursday | football-data may list the fixtures, still without odds for them. | Dixon-Coles only |
| **Friday afternoon** | football-data adds the odds of the weekend matches. | **Dixon-Coles + market** |
| Friday night | First match; the ticket closes before it. | — |

So the useful window is from Friday afternoon to the first match: a few hours when there is a
Friday match. Midweek matchdays (Copa, Champions League, midweek league rounds) leave even
less, from Tuesday afternoon to the matches on Tuesday or Wednesday night. In practice: run
`invoke update` on Friday afternoon (Tuesday afternoon for midweek matchdays), check in
`current.html` that the market odds are there (v3 differs from v2), and fill the ticket. The
exact closing time of the ticket and the exact time football-data publishes have not been
checked; confirm them before relying on a tight window.

**Choice of the combination.** Measured on the same out-of-sample forecasts (2015-16 to
2025-26, log-loss):

| Combination | Parameters | Hit rate | Log-loss |
|---|---|---|---|
| Full multinomial regression on the six log-probabilities, 3 years | 21 | 49.7% | 1.0049 |
| Same, 5 years | 21 | 49.9% | 1.0043 |
| Same, one regression per league, 3 years | 2 × 21 | 49.7% | 1.0071 |
| Log-linear pool, 3 years | 4 | 49.9% | 1.0035 |
| **Log-linear pool, 5 years** | 4 | 50.0% | **1.0031** |

With a few thousand matches, fewer parameters generalise better.

**Observations.**

- The best log-loss, Brier and RPS of every model, **also better than the market** (1.0028 vs
  1.0046 log-loss) and in each of the 11 seasons. The hit rate is the same as the market's
  (50.0%), +1.3 points over v2; in Segunda it beats the market (47.1% vs 46.8%), in La Liga
  it is slightly below (53.5% vs 53.9%).
- It wins fewer prize tickets than the market (35 vs 40 out of 376), but over 376 matchdays a
  difference of 5 tickets is well within luck.
- The fitted weights are stable over time: $b \approx 1.15$ to $1.25$ and
  $a \approx -0.07$ to $-0.15$ (from 2015 to 2024). Most of the gain is a **recalibration of the
  market**: $b > 1$ means the normalised odds are slightly too cautious (favourites win a bit
  more often than they imply). Dixon-Coles gets a small negative weight: given the odds, it adds
  no information of its own (the market already contains what past goals say), so its weight
  only corrects for the overlap between both inputs.
- It picks X more often (2.8% vs 1.2% for v2 and 0.5% for the market), mostly through the draw
  intercept, positive in recent seasons ($c_X \approx 0.02$ to $0.05$), with no loss of hit
  rate.
- It loses independence: a model that uses the odds cannot claim to beat the market on its own.
  That is why v2 stays as the independent model.

### v4 · Pi-ratings (`pi_ratings`)

[`quiniela/prediction/pi_ratings.py`](src/quiniela/prediction/pi_ratings.py)

**Idea.** A rating system made for football (Constantinou & Fenton, 2013): every team has
**two ratings, one at home and one away**, learned **match by match from the goal difference**
(a 3-0 says more than a 1-0, with diminishing returns). Like Dixon-Coles it uses only results,
but it adapts after every match instead of weighting a window of past matches, and each team
has its own home advantage. It is an independent model, like v2.

**Ratings and predicted goal difference.** A rating $r$ means an expected goal difference
against an average team of

```math
\psi(r) = \operatorname{sign}(r)\left(b^{|r|/c} - 1\right), \qquad b = 10,\; c = 3
```

so the predicted goal difference of home team $h$ against away team $a$ is
$\hat g = \psi(R^H_h) - \psi(R^A_a)$.

**Learning.** After the match, with real goal difference $g$, the error $e = g - \hat g$ is
weighted with diminishing returns and moves both teams, the rating of the ground they played at
by the learning rate $\lambda$ and their other rating by a share $\gamma$ of that change:

```math
\phi(e) = \operatorname{sign}(e)\, c \log_b(1 + |e|)
```

```math
\Delta R^H_h = \lambda\,\phi(e), \quad \Delta R^A_h = \gamma\,\Delta R^H_h, \qquad
\Delta R^A_a = -\lambda\,\phi(e), \quad \Delta R^H_a = \gamma\,\Delta R^A_a
```

Every match since 1993-94 is replayed in date order, so each one is predicted with the
ratings before it.

**Promoted and relegated teams.** La Liga and Segunda teams never meet in the league, so their
ratings are relative to their own league. A team entering a league from below (or for the first
time) takes the mean ratings of the three lowest-rated teams of that league's previous season
(the ones it replaces); one entering from above, of the three highest-rated ones.

**From goal difference to 1X2.** An ordered logistic regression per league, fitted on the last
five seasons of training matches (their pre-match $\hat g$):

```math
P(A) = \sigma(\theta_1 - \beta \hat g), \qquad P(A \cup D) = \sigma(\theta_2 - \beta \hat g),
\qquad P(H) = 1 - P(A \cup D)
```

**Parameters.** $\lambda = 0.035$ (the paper's) and $\gamma = 0.9$ (0.7 in the paper), chosen on
seasons **before** the backtest (log-loss of 2010-11 to 2014-15, mapping fitted on 2005-06 to
2009-10): a larger learning rate overreacts and a smaller one lags. $\gamma = 1$ scores about
the same (-0.0007) but turns the two ratings of a team into one, so the home/away split itself
adds little.

**Backtest** (same walk-forward, 2015-16 to 2025-26, 9,262 matches; see the table at the top of
this section):

| | v2 Dixon-Coles | **v4 Pi-ratings** | v3 DC + Market | Market |
|---|---|---|---|---|
| Hit rate | 48.7% | **49.0%** | 50.0% | 50.0% |
| Log-loss | 1.0215 | **1.0178** | 1.0028 | 1.0046 |
| Brier | 0.6130 | **0.6104** | 0.6002 | 0.6013 |
| RPS | 0.2059 | **0.2048** | 0.2003 | 0.2007 |
| Prize tickets (single column) | 27 / 376 | **30 / 376** | 35 / 376 | 40 / 376 |

**Does it add anything to v3?** The same log-linear pool as v3, with the pi-ratings as one more
term, $P(k) \propto p_{DC}^{a}\, p_{\pi}^{d}\, p_{M}^{b}\, e^{c_k}$, fitted on the backtest's
out-of-sample forecasts of one half of the seasons and scored on the other:

| Pool (trained → scored) | 2015-20 → 2020-26 | 2020-26 → 2015-20 |
|---|---|---|
| DC + market (v3) | **1.0038** | **1.0025** |
| DC + pi-ratings + market | 1.0045 | 1.0031 |
| Pi-ratings + market | 1.0045 | 1.0026 |

**Observations.**

- **The best independent model.** v4 beats Dixon-Coles on every probability metric over the
  11 seasons (log-loss -0.0037, Brier -0.0026, RPS -0.0011) and wins three more single-ticket
  prizes, in both leagues. Learning from every match, with diminishing returns for big wins,
  works slightly better than Dixon-Coles' time-weighted strengths.
- **It does not beat the market.** It stays 0.013 behind it in log-loss, about the same gap as
  Dixon-Coles: the bookmakers know everything results tell, and more.
- **Nothing on top of the market.** In the pool its weight is close to zero or negative and the
  out-of-sample log-loss gets slightly worse, exactly as with Dixon-Coles in v3. So v3 does not
  change and the tickets (v3+v1F·D, v3+v1F·R, v3+v1F·B) keep planning on v3. v4 matters when there are no
  odds: it would be the better fallback than Dixon-Coles, a change left for a later version.
- **It almost never picks a draw** (0.1% of the matches, like Dixon-Coles' 1.2%): no
  difference of ratings is small enough to make the draw the most likely result.

### Consensus of the models (tested, not used)

**Idea.** Combine v1 to v4 by their **consensus**: if the models disagree on a match, maybe it
is less predictable than v3's probability says, and the doubles and triples should go there.
Tested on the 9,262 backtest matches (2015-16 to 2025-26) with the walk-forward predictions of
every model; nothing in the code uses it.

**1. Hit rate at the same confidence.** Matches grouped by how many of v2 and v4 mark v3's sign:

| v2 and v4 agreeing with v3 | Matches | v3's mean confidence | v3's real hit rate | Gap |
|---|---|---|---|---|
| None | 927 (10%) | 38.6% | 39.5% | +0.9 points |
| One | 632 (7%) | 40.8% | 38.3% | -2.5 points |
| Both | 7,703 (83%) | 51.6% | 52.2% | +0.6 points |

v3 is calibrated in every group: at the same confidence it hits as often with or without
consensus. Split again by confidence tier (below 40%, 40-45%, ... 70% or more) the gaps go up
and down with no pattern and almost none beyond chance; counting v1 too changes nothing.

**2. Correcting the probabilities by consensus.** A temperature per consensus group,
$P(k) \propto p_{v3}(k)^{\tau_g}$ ($\tau_g < 1$ opens the probabilities of group $g$), fitted on
half of the seasons and scored on the other half:

| Trained → scored | v3 as it is | $\tau$ by consensus group |
|---|---|---|
| 2015-20 → 2020-26 | **1.0036** | 1.0039 |
| 2020-26 → 2015-20 | **1.0019** | 1.0028 |

The correction makes the log-loss worse both ways, and the fitted $\tau_g$ change direction from
one half to the other: they only fit noise.

**3. Who is right when they disagree.** In the 927 matches where v2 and v4 both mark another
sign than v3:

| Model | Right |
|---|---|
| v3 | **39.5%** |
| v4 | 31.2% |
| v1 | 30.5% |
| v2 | 30.2% |

The goal models mostly back the home side there (v2 marks 1 in 70% of them), while v3, following
the market, mostly marks 2 (52%) or X (26%), and is right much more often. A disagreement does
not mean an unpredictable match: it means the goal models are missing what the odds know.

**Conclusion.** The consensus adds nothing beyond v3's probabilities, as expected: v3 already
uses the disagreement between Dixon-Coles and the market, against Dixon-Coles (its negative
weight in the pool), and adding v4 to that pool did not help either (see
[v4](#v4--pi-ratings-pi_ratings)). The tickets keep planning on v3's probabilities.

### v3+v1F·D · Direct ticket (`planner`)

[`quiniela/prediction/planner.py`](src/quiniela/prediction/planner.py)

**Idea.** Not a new probability model but a **decision layer** on top of v3: the quiniela
allows covering more than one sign of a match. A **double** (two signs) doubles the number of
bets and a **triple** (1, X and 2) triples it, so $d$ doubles and $t$ triples cost
$2^d \cdot 3^t$ bets. With a fixed budget, the question is which matches get them. v3+v1F·D uses
the v3 probabilities to place them where they raise the chance of a prize the most.

**Coverage of a match.** With the v3 probabilities sorted as
$p_{i(1)} \ge p_{i(2)} \ge p_{i(3)}$, the probability of hitting match $i$ is:

```math
q_i = \begin{cases}
p_{i(1)} & \text{single sign} \\
p_{i(1)} + p_{i(2)} & \text{double (the two most likely signs)} \\
1 & \text{triple}
\end{cases}
```

so a double gains $p_{i(2)}$ and a triple $1 - p_{i(1)}$.

**Chance of each prize.** Treating the 14 matches as independent, the number of hits $H$
follows a Poisson-binomial distribution, computed exactly by adding one match at a time:

```math
P_i(h) = P_{i-1}(h)\,(1 - q_i) + P_{i-1}(h - 1)\,q_i, \qquad P_0(0) = 1
```

This gives the chance of every category before the matchday: $P(H = 14)$ for the 1st,
$P(H \ge 13)$ for the 2nd or better, ..., $P(H \ge 10)$ for any prize. The dashboards show it
under every ticket, also for single-sign tickets ($q_i = p_{i(1)}$).

**Optimal ticket.** Given a budget, every way of placing the doubles and triples on the 14
matches is evaluated and the one that maximises the target is kept:

```math
\max_{\text{placement}} \; P(H \ge k), \qquad k = 11 \text{ (default)}, \; 10 \text{ (any prize) or } 14 \text{ (the top one)}
```

Ties go to the higher chance of 14 hits, then to the higher expected hits $\sum_i q_i$, and
finally to the cheaper ticket. The budget can be given as exact doubles and triples, or as a
maximum number of bets: every mix that fits is tried (with 48 bets: 5 doubles; 4 doubles +
1 triple; 2 doubles + 2 triples; 3 triples...). The search is exhaustive (about 22,000
placements for 48 bets, milliseconds).

**The budget is a cap, not a target.** The best mix does not have to spend it all: if a cheaper
one gives a higher chance of a prize, it wins (2 doubles + 2 triples cost 36 bets, less than
48).

**Dynamic, matchday by matchday.** Only the budget and the target are fixed. How many doubles
and triples to use, and on which matches, is decided again every matchday from that matchday's
v3 probabilities.

**Why the mix changes.** A double gains the second sign and a triple everything but the
favourite. In an open match (40 / 35 / 25) a double raises the chance of a hit from 40% to 75%
(+35 points) and a triple to 100% (+60). Going from 2 doubles + 2 triples to 4 doubles +
1 triple turns one triple into a double and adds two doubles:

- with **two very open matches** (favourite below about 45%), the second triple is worth a lot,
  and the two extra doubles would go to clearer matches whose second sign adds little (a
  65 / 22 / 13 match gains +22): **2 doubles + 2 triples** wins;
- with **one very open match** and several middling ones (55 / 30 / 15 gains +30 with a
  double), the extra doubles pay more: **4 doubles + 1 triple** wins, the usual case.

**v3+v1F·D** (direct, called v3+v1F·D until v0.7.2) is the default configuration of the planner: the v3 signs, a **48-bet budget** with the best mix of
doubles and triples, and the target **11+ hits**. Outside the ticket it is the v3 sign. Until
v0.4.0 the target was 10+ hits (any prize), but the [official draws](#official-draws-real-tickets-winners-and-prizes)
show that the 10-hit prize is often not paid (below 1.50 €), so securing it is a poor goal.

**Backtest.** Every matchday of 2015-16 to 2025-26 with a full ticket (376, the approximate
ticket: La Liga then Segunda), with the v3 signs, played in seven ways. Every strategy but the
first spends the same budget, 4 doubles and 1 triple (48 bets); v3+v1F·D also chooses the mix within
48 bets; random placements are averaged over 20 draws. Prizes are counted here; the
[euros](#v3v1fe-and-v3v1fb--money-tickets-money) come with v3+v1F·E and v3+v1F·B, on the real tickets:

| Strategy | Bets | Prizes (10+) | 14 | 13 | 12 | 11 | 10 | Mean hits | Planned chance of a prize | Real prize rate |
|---|---|---|---|---|---|---|---|---|---|---|
| No doubles or triples | 1 | 35 / 376 | 0 | 1 | 1 | 12 | 21 | 7.20 | 10.3% | 9.3% |
| Random matches | 48 | 130.2 / 376 | 0.3 | 3.4 | 13.3 | 40.8 | 72.4 | 8.80 | 33.2% | 34.6% |
| By eye (least predictable matches) | 48 | 158 / 376 | 0 | 10 | 20 | 55 | 73 | 9.18 | 41.0% | 42.0% |
| Optimal, aiming at 14 hits | 48 | 154 / 376 | 0 | 8 | 21 | 53 | 72 | 9.13 | 41.2% | 41.0% |
| Optimal, aiming at 10+ hits (v3+v1F·D until v0.4.0) | 48 | 159 / 376 | 0 | 6 | 26 | 55 | 72 | 9.15 | 41.3% | 42.3% |
| Optimal, aiming at 11+ hits | 48 | 156 / 376 | 0 | 8 | 24 | 55 | 69 | 9.16 | 41.3% | 41.5% |
| **v3+v1F·D** (best direct mix within 48 bets, 11+) | 47.9 | 156 / 376 | 0 | 8 | 24 | 54 | 70 | 9.16 | 41.3% | 41.5% |
| **v3+v1F·R** (best direct or reduced ticket within 48 bets, 11+, see [v3+v1F·R](#v3v1fr--reduced-ticket-reduced)) | 48 | **191 / 376** | 0 | 6 | 36 | 65 | 84 | 9.56 | 50.0% | 50.8% |

Aiming at 11+ trades a few 10-hit tickets (70 against 76) for a few more of 11 or more (86
against 84): the prizes that are always paid.

**Observations.**

- **Hit rate.** v3+v1F·D hits about 58% of all the matches, against 50.0% for v3 (+8 points), and
  65.4% of the ticket matches (9.16 of 14 on average, against 7.20 for the single v3 ticket):
  the evaluation dashboard shows it as its own column. Outside the ticket v3+v1F·D is v3, so the
  whole gain comes from the 14 ticket matches.
- Doubles and triples placed with the v3 probabilities win **4.5 times more prizes** than the
  single ticket (156 against 35) and **20% more than placing the same bets at random** (156
  against 130).
- The planned chance of a prize matches what happened (41.3% planned, 41.5% real; 10.3%
  against 9.3% for the single ticket): the chances shown in the dashboards can be trusted.
- Most of the gain comes from **where** the bets go, not from the optimisation itself: the
  "by eye" rule (triples on the least predictable matches, doubles on the next ones) gets
  158 prizes, as many as v3+v1F·D (156). That difference is within luck; the planner's value is doing
  it consistently and choosing the mix (with 11+ as target, 4 doubles + 1 triple in 374 of 376
  matchdays).
- Aiming at 14 hits does not pay off with this budget: no strategy got the 1st category in
  376 matchdays (random placements, 0.25 on average), and the 14-hit target loses prizes in
  the lower categories.
- The prizes are counted, not their money: the lower categories, the ones doubles and triples
  win most, pay the least. In euros (real tickets and prizes, see [v3+v1F·E and v3+v1F·B](#v3v1fe-and-v3v1fb--money-tickets-money))
  v3+v1F·D loses about two thirds of what it spends.

### v3+v1F·R · Reduced ticket (`reduced`)

[`quiniela/prediction/reduced.py`](src/quiniela/prediction/reduced.py)

**Idea.** v3+v1F·D plays **direct** tickets: every combination of its doubles and triples
($2^d \cdot 3^t$ bets). The official ticket also offers six **reductions**: the same doubles
and triples covered with only some of the combinations, chosen so that every possible result is
close to at least one column played. With the same 48 bets a reduction covers **more matches**,
at the price of losing the guaranteed 14: if every covered sign is right, the best column has at
most one miss (13 hits guaranteed; 12 for the 5th and 6th reductions). v3+v1F·R is v3+v1F·D with the
reductions as extra candidates; v3+v1F·D and v3+v1F·R are shown side by side.

| Reduction | Doubles | Triples | Direct bets | Bets | Guaranteed | Chance of 14 if every covered sign is right |
|---|---|---|---|---|---|---|
| 1st | 0 | 4 | 81 | 9 (6.75 €) | 13 | 1/9 |
| 2nd | 7 | 0 | 128 | 16 (12 €) | 13 | 1/8 |
| 3rd | 3 | 3 | 216 | 24 (18 €) | 13 | 1/9 |
| 4th | 6 | 2 | 576 | 64 (48 €) | 13 | 1/9 |
| 5th | 0 | 8 | 6,561 | 81 (60.75 €) | 12 | – |
| 6th | 11 | 0 | 2,048 | 132 (99 €) | 12 | – |

The other matches can still be played direct: each extra double multiplies the bets by 2 and
each triple by 3 (e.g. the 3rd reduction plus one direct double: 48 bets).

**Columns.** Loterías does not publish the column tables online, so each reduction is a
**covering code** with the official size and guarantee: the ternary Hamming code for the 1st
(9 columns, perfect: every one of the 81 results is at most one sign away from a column), the
binary Hamming code (7, 4) for the 2nd (16 of 128, perfect), their product for the 5th, and codes
found by local search for the 3rd, 4th and 6th; the tests check every guarantee. All of them play
the column of the most likely signs. Equivalent codes give the same guarantee; the chance of 14
can differ slightly from the official columns.

**Hit distribution.** The direct matches follow the Poisson-binomial of v3+v1F·D. In the $R$
matches of the reduction, each sign is named by its rank $0, 1, 2$ (most to least likely; in a
double rank 2 is the sign left out), every result $o \in \lbrace 0, 1, 2 \rbrace^R$ is
enumerated (at most $3^7$ in the reductions that fit 48 bets) and the hits of the **best
column** of the code $C$ are counted:

```math
P(H_R = h) = \sum_{o \,:\, h(o) = h} \; \prod_{i=1}^{R} p_{i(o_i)}, \qquad
h(o) = \max_{c \in C} \#\{i : c_i = o_i\}
```

The two blocks are independent, so the hits of the ticket are the convolution of the reduced
block with the direct one:

```math
P(H = h) = \sum_{k} P(H_R = k)\, P(H_{\text{direct}} = h - k)
```

This is exact (no simulation) and the tests check it against a brute-force enumeration.

**Search.** For every reduction within the budget (with 48 bets: the 1st, 2nd and 3rd), every
way of putting it on the 10 least predictable matches (which are triples and which doubles) and
every placement of the direct doubles and triples that still fit on the other matches are
evaluated, about 50,000 tickets in 0.4 seconds; the ticket with the highest $P(H \ge 11)$ wins,
with the same tie-breaks as v3+v1F·D. The best direct ticket (v3+v1F·D) is a candidate too, so v3+v1F·R's
planned chance is never lower than v3+v1F·D's. The prizes in euros count every column the reduction
really plays.

**Backtest.** Same 376 matchdays and the same 48 bets as v3+v1F·D (table above), and the same 414 real
draws in euros:

| | v3+v1F·D (direct) | v3+v1F·R (direct or reduced) |
|---|---|---|
| Ticket chosen | 4 doubles + 1 triple (374 of 376) | 3rd reduction + 1 direct double (289), 2nd reduction + 1 direct triple (87) |
| Matches covered | 5 | 7 or 8 |
| Prize tickets (10+) | 156 / 376 | **191 / 376** (+22%) |
| 11 or more hits | 86 | **107** |
| Mean hits | 9.16 | 9.56 |
| Planned / real prize rate | 41.3% / 41.5% | 50.0% / 50.8% |
| Hit rate of all matches | 57.9% | 62.5% |
| Real draws in euros: return | 35% | **49%** |
| 14 hits in real draws | 0 | 1 |

**Observations.**

- **The reduction wins every matchday.** v3+v1F·R always played a reduction: covering two or three
  more matches is worth more than the 14 guaranteed when everything covered is right, which
  with 48 bets is rare anyway. It wins 35 more prize tickets (+22%), mostly 11 and 12 hits, the
  categories that are always paid.
- **The planned chance can be trusted.** 50.0% planned against 50.8% real, as for v3+v1F·D.
- **More money back, still a loss.** On the real draws v3+v1F·R gets back 49% of what it spends
  against 35% for v3+v1F·D, and it got the only 14 of the backtest (a draw of favourites, so small).
  It still loses half of the money: the quiniela keeps 45% of the takings.
- **Why not in v3+v1F·B.** The big-win ticket looks for prizes of 10,000 € or more, which come almost
  only from 14 hits (a 13 pays less than about 9,200 € in three draws out of four, see
  `prizes.html`), and with the same bets a direct ticket concentrates more chance on the 14
  (and pays the chained 13s and 12s that come with it). v3+v1F·B stays direct.

### v3+v1F·E and v3+v1F·B · Money tickets (`money`)

[`quiniela/prediction/money.py`](src/quiniela/prediction/money.py) ·
backtest in [`quiniela/evaluation/money.py`](src/quiniela/evaluation/money.py)

**Idea.** v3+v1F·D maximises the chance of a prize, but a prize is shared among its winners and the
quiniela gives back only 55% of the takings: on average every bet loses money. v3+v1F·E (*expected prize*) looks at
**euros**: it builds the ticket with the highest expected prize and recommends playing it **only
when that prize is above its cost**, which could happen with a jackpot or when results the
public does not expect are likely. In its prudent form (the default) it has never recommended
playing. It needs the [official draws](#official-draws-real-tickets-winners-and-prizes).

**Prize of a category.** With $B$ bets in the draw (takings $0.75\,B$), a share $s_k$ of the
takings for $k$ hits and $W_k$ other winners, each winning column gets

```math
\text{prize}_k = \frac{s_k \cdot 0.75\,B}{W_k + 1}, \qquad s_{14} = 16\%, \; s_{13} = s_{12} = s_{11} = 7.5\%, \; s_{10} = 9\%
```

(the 10-hit prize is not paid below 1.50 €); the Pleno al 15 shares 7.5% plus the announced
jackpot.

**How many winners: the surprise of the results.** The public mostly marks favourites: when
they win there are many winners and small prizes; after surprises, few winners share large
ones. With $m_{i, r_i}$ the market probability of the real result of match $i$, the surprise of
a draw is

```math
S = -\sum_{i=1}^{14} \log m_{i, r_i}
```

and the share of winning bets of every category follows it (prize model, fitted on past draws
with 10+ La Liga / Segunda matches):

```math
\log \frac{W_k + 0.5}{B} = a_k + b_k\, S
```

| Hits | $b_k$ | $R^2$ |
|---|---|---|
| 14 | -1.05 | 0.72 |
| 11 | -0.93 | 0.84 |

Each extra unit of surprise divides the winners by about $e^{1} \approx 2.7$, so prizes grow
fast with surprises. Given the real surprise, the model's average prize per category matches the
real one (11 hits: 82 € against 82 €; 13 hits: 13,750 € against 13,331 €). The Pleno al 15
winners are a fixed share of the 14-hit winners.

**Expected prize of a ticket.** 4,000 sets of results are simulated from the v3 probabilities
(the other ticket matches get the league shares of 1, X and 2). Each one gives the hits of every
column, its surprise $S$ (from the market probabilities, the stand-in for the public) and so the
prizes. A ticket with $d$ doubles and $t$ triples has $2^d 3^t$ columns, and lower categories
also pay: if $h$ singles and $d_h$ doubles are right, the number of columns with $h + j$ hits
is the coefficient of $x^j$ in

```math
2^{\,d - d_h} \,(x + 1)^{d_h} \,(x + 2)^{t}
```

(a right double has one right and one wrong sign; a missed one, two wrong; a triple, one right
and two wrong).

**Plan.** For every mix of doubles and triples within 48 bets (at least two, the minimum
ticket), the placements with the highest chance of 11+ hits are simulated and the one with the
best expected net prize (expected prize minus $0.75 \cdot 2^d 3^t$) is kept. Two details avoid
overestimating it:

- **Winner's curse.** The best of hundreds of tickets on the same simulations looks better than
  it is (rare 14-hit prizes make the estimates noisy): in a test, a ticket "worth" 21.4 € was
  worth 5.0 € on new simulations. So the ticket is chosen on one set of simulations and valued
  on another.
- **Typical winners.** Adding random noise to the number of winners inflated the average prize
  of 11 to 13 hits by 20-50% over the real one, so the valuation uses the fitted share.

**Prudent verdict.** Most of a ticket's expected prize comes from 14 hits and the Pleno al 15,
a few times per thousand draws: a lottery that a few hundred draws cannot check. So v3+v1F·E is
**prudent**: only the prizes of 10 to 13 hits count, both to choose the ticket and to decide to
play it ($w = 0$ below); the full version ($w = 1$) counts everything:

```math
\text{play} \iff E[\text{prize}_{10-13}] + w \cdot E[\text{prize}_{14 + \text{Pleno}}] > 0.75 \cdot 2^d 3^t
```

v3+v1F·E is not shown in the dashboards (it never plays); its backtest is in `evaluation.html`. The
dashboards show **v3+v1F·B**, the big-win ticket below.

**Backtest in euros.** The **real tickets** of 2015-16 to 2025-26 with 10+ La Liga / Segunda
matches (414 draws; the other matches get the home sign), paid with the **real prizes** of each
draw, sharing every pool with its real winners. Everything is known before the draw: the v3
walk-forward probabilities, the odds, a prize model fitted on earlier draws only, the expected
bets (average of the last five draws) and the announced jackpot.

| Strategy | Draws played | Spent | Won | Net | Return | Best prize | Draws with 13 / 12 / 11 / 10 hits |
|---|---|---|---|---|---|---|---|
| One column, v3 signs (0.75 €) | 414 | 311 € | 101 € | -209 € | 33% | 30 € | 0 / 3 / 9 / 23 |
| v3+v1F·D: 48 bets (direct), 11+, every draw | 414 | 14,895 € | 5,270 € | -9,625 € | 35% | 1,213 € | 5 / 20 / 44 / 91 |
| v3+v1F·R: 48 bets (direct or reduced), 11+, every draw | 414 | 14,904 € | 7,268 € | -7,636 € | 49% | 985 € | 3 / 30 / 56 / 91 |
| **v3+v1F·E (prudent): only when 10-13 hits pay** | **0** | 0 € | 0 € | **0 €** | – | – | – |
| v3+v1F·E counting 14 hits and the Pleno al 15 | 143 | 3,921 € | 1,579 € | -2,342 € | 40% | 227 € | 2 / 7 / 13 / 25 |
| Full v3+v1F·E ticket every draw | 414 | 9,908 € | 4,160 € | -5,748 € | 42% | 1,213 € | 3 / 16 / 37 / 66 |
| v3+v1F·B big win: best chance of 10,000+ € within 50 €, every draw | 414 | 19,865 € | 7,278 € | -12,587 € | 37% | 791 € | 6 / 21 / 46 / 74 |

*Return: euros won per euro spent; the quiniela gives back 55% on average to all players. Only
v3+v1F·R got 14 hits, once in 414 draws (2018-19, jornada 34, a draw of favourites: 862 € with all
its columns).*

**Observations.**

- **Nothing makes money**, and the prudent v3+v1F·E says so: in 11 seasons no ticket's expected
  prize from 10 to 13 hits reached its cost (at most 60% of it, 42% in a typical draw), so it
  would not have played a single draw. The least bad ticket is almost always one double (two
  bets, the minimum). The quiniela keeps 45% of the takings and our probabilities are close to
  the market's, roughly what the public plays.
- **Its prize estimates are right where they can be checked.** In the draws the full version
  played it expected 11.68 € per ticket from 10 to 13 hits and got 12.37 €; the chance of each
  number of hits, the surprise of the results (14.07 expected, 14.11 real) and the average prize
  per category given the surprise also match.
- **The rest is a lottery.** The full version plays mostly draws with a jackpot, and most of
  its expected prize (about 27 € of 39 € per ticket) comes from 14 hits and the Pleno al 15:
  about one ticket in several hundred. None came up, which is not unusual with so few tickets,
  so the backtest can neither confirm nor reject it; its return even changes with the
  simulations (40% in this run, 46% in another with different random draws).
- **Rules in between** (a minimum jackpot, a fraction of the 14-hit prizes) looked better on
  these same draws (e.g. 8 draws with a 122% return), but choosing a rule on the data that
  grades it would overfit; they are not used.

**Big-win mode** (`plan_big_win`, v3+v1F·B, the ticket in the dashboards). The prudent v3+v1F·E answers
"is it worth it on average?" (never). But a quiniela player usually looks for something else:
**a large prize at once**, accepting to lose on average. That needs many hits (13 or 14) **and**
few other winners (results the public did not expect), and both are in one number, the chance
of a prize of at least $X$ within a weekly budget:

```math
\max_{\text{ticket}} \; P(\text{payout} \ge X), \qquad 0.75 \cdot 2^d 3^t \le 50 \text{ €}, \quad X = 10{,}000 \text{ €}
```

- **14 hits and the Pleno al 15**: their probability is exact, $\prod_i q_i$ (the coverage of
  every match); only the surprise *given* 14 hits is simulated, drawing each result among the
  marked signs. Plain simulations would see a 14 a handful of times and pick tickets by noise.
- **12 and 13 hits**: simulated. In every simulated draw the chance that $c_k$ winning columns
  reach $X$ uses the spread of the winners, $\sigma_k$, because large prizes are precisely the
  draws with fewer winners than usual:

```math
P_k = \Phi\left(\frac{\log(c_k\, s_k\, 0.75\, B / X) - \log B - a_k - b_k S}{\sigma_k}\right)
```

The ticket is chosen on one set of simulations and valued on another. The dashboards do not
give a verdict: they show **what can happen** with the ticket, the chance of each payout range
(nothing back, less than the cost, up to 1,000 €, 1,000-10,000 €, 10,000-100,000 €, more), with
the chance of $X$ or more, of 13+ and 14 hits, and how much comes back on average.

Backtest (last row of the table above): with 48 € a draw it almost always plays 6 doubles
(64 bets). Its plans gave a prize of 10,000 € or more a **0.06% chance per draw (about once
every 1,700 draws)**, 0.25 expected in the 414 draws; none came, as expected. It gets back
37% of the money, with six 13-hit tickets in 11 seasons (the best one 791 €, the 13-hit prizes
of those draws being small). A large prize with 50 € a week is a matter of decades.

**Traffic lights: is this a good matchday to play?** Under the v3+v1F·D, v3+v1F·R and v3+v1F·B tickets,
`current.html` shows a traffic light (green *recommended to play*, amber *undetermined*, red
*not recommended*, each with its icon and label) for four planned chances of the tickets:

| Ticket | Chance ranked | Past tickets compared |
|---|---|---|
| v3+v1F·B (big win) | of a prize of 10,000 € or more | the big-win tickets of the 414 past official draws |
| v3+v1F·B (big win) | of 13 or more hits | the same |
| v3+v1F·D | of 11 or more hits (its target) | the v3+v1F·D tickets of the same draws |
| v3+v1F·R | of 11 or more hits (its target) | the v3+v1F·R tickets of the same draws |

The rule is a percentile: the ticket's chance $c$ is ranked among the chances $c_j$ the same
strategy had in the $n$ past real tickets of the money backtest,

```math
\text{percentile} = \frac{100}{n} \, \#\{j : c_j < c\}, \qquad
\text{green} \ge 75, \quad \text{red} \le 25, \quad \text{amber in between}
```

The two v3+v1F·B lights often disagree, and that is the point: a matchday of clear favourites gives
many hits (green on 13+ hits) but small prizes (red on 10,000 €), while open matches or a jackpot
give the opposite. Both together say which kind of matchday it is: a likely but small prize, or
a rare but large one. **No traffic light makes the quiniela profitable**: they only compare
matchdays with each other, and every ticket loses money on average; the dashboard says so in
red under the lights, with the rule in small print.

### Next steps

See [Roadmap](#roadmap) at the end of this file.

## Liga F (women's league)

Since 2026-27 weekend tickets have **4 Liga F matches** (10 of the 11 Sunday draws from the
3rd jornada; before, only two summer draws of 2023-24 had women's matches), 29% of a weekend
ticket. The women's league is modelled and evaluated **apart from the men's**: its own
models (v1F, v2F...), its own data and its own tables in every dashboard.

| Season | Liga F matches (ESPN) | Played | On the official tickets | With public picks | 1 | X | 2 | Goals per match |
|---|---|---|---|---|---|---|---|---|
| 2022-23 | 240 | 240 | 0 | 0 | 44.2% | 20.4% | 35.4% | 3.17 |
| 2023-24 | 240 | 240 | 16 | 0 | 44.2% | 18.3% | 37.5% | 3.15 |
| 2024-25 | 240 | 240 | 0 | 0 | 41.7% | 21.7% | 36.7% | 2.82 |
| 2025-26 | 240 | 240 | 0 | 0 | 39.6% | 20.8% | 39.6% | 2.88 |
| 2026-27 (to 15 November) | 240 | 82 | 44 | 4 | 45.1% | 22.0% | 32.9% | 2.98 |

*Fewer draws than the men's leagues (about 20% against 28%) and many more away wins (about 37%
against 26%): a very uneven league, where the strong teams win anywhere.*

### Data

**Results and fixtures: ESPN** ([`quiniela/io/espn.py`](src/quiniela/io/espn.py)). ESPN's
public scoreboard (the JSON its own website uses, `site.api.espn.com/.../soccer/esp.w.1/scoreboard`)
has every Liga F match since 2022-23, its first professional season (240 a season): date, teams,
score and status, the season's fixtures included. It is read month by month (about twelve
small requests a season; soccerdata's ESPN reader asks for every match day and has no scores),
cached in `data/cache/espn/` and saved in `data/women/espn/<season>.parquet`. **`invoke update`
downloads the missing completed seasons and refreshes the current one** every time (a failure
is a warning; the men's update goes on). It is not an official, documented API and has no
odds, so everything is kept locally. ESPN also has team statistics per match (shots, shots on
target, possession...), not used yet (see [Roadmap](#roadmap)).

Other sources checked: StatsBomb's open data has the Liga F of 2023-24 only; FBref, Flashscore and
Sofascore are behind bot protection or forbid scripts; football-data.co.uk and TheSportsDB have
no Liga F; FootyStats is paid.

**The tickets and the public's picks** ([`quiniela/data/women.py`](src/quiniela/data/women.py)):

- the Liga F matches (marked `(F)`) of the official draws saved in `data/lae/`, with results;
- the share of bettors marking 1, X and 2, only on the betting page of the next ticket: every
  `invoke update` / `invoke dashboards.build` **archives `data/lae/upcoming.json`** in
  `data/lae/public/<season>_j<jornada>.json`;
- each ticket match is **linked to its ESPN match** (same teams within four days), through an
  alias table for the names that differ (`At. Madrid (f)` → Atlético Madrid, `Madrid (f)` →
  Madrid CFF, `Tenerife (f)` → Granadilla Tenerife / Tenerife...): 58 of the 60 ticket matches
  are linked; the other two were women's matches of another competition (Las Planas).

Everything is joined in `data/women/liga_f.parquet`. Running `invoke update` **every week**
before the deadline (and clicking the bookmark when it opens the betting page) builds the v1F
history: a jornada without its ticket saved has no v1F prediction. `data/` is not versioned, so
back up `data/lae/public/`: it cannot be downloaded again.

### v1F · Public picks (`public_picks`)

[`quiniela/prediction/public_picks.py`](src/quiniela/prediction/public_picks.py), registered in
`WOMEN_MODELS`. The public's share $s_k$ of each sign, normalised,

```math
P(k) = \frac{s_k}{s_1 + s_X + s_2}, \qquad \hat r = \underset{k}{\arg\max}\; s_k
```

Without public picks it falls back to the shares of 1, X and 2 of the earlier Liga F matches.
It is the Liga F model of the tickets (`WOMEN_PLANNER_MODEL`): the v3 column shows it on the
Liga F matches and v3+v1F·D, ·R, ·B plan them with its probabilities.

### v2F · Pi-ratings (`pi_ratings_f`)

The [v4 pi-ratings](#v4--pi-ratings-pi_ratings) (same model, same parameters, `PiRatingsModel`)
trained on every Liga F match from ESPN: home and away ratings learned from goal differences,
mapped to 1X2 by an ordered logistic regression fitted on the Liga F itself. On the next ticket
it predicts each women's match trained on the matches played before the sales deadline.

### Evaluation (`quiniela.evaluation.women`)

- **v2F and the reference** (the **result shares** of the earlier matches, always the most
  frequent result) predict **every Liga F match walk-forward, week by week**, trained on the
  matches played before that week; 2022-23 only trains.
- **v1F** is scored on the ticket matches whose betting page was saved.
- Each model is scored on its own matches, and all of them together on the **common** matches
  (the ticket matches with public picks): the fair comparison of v1F and v2F.

| 2023-24 to 2026-27 (802 matches) | v2F (pi-ratings) | Result shares (reference) |
|---|---|---|
| Hit rate | **59.2%** | 42.1% |
| Log-loss | **0.8955** | 1.0585 |
| Brier | **0.5239** | 0.6419 |
| RPS | **0.1822** | 0.2395 |
| Hit rate of home wins / draws / away wins | 81.1% / 0.0% / 67.0% | 100% / 0% / 0% |

| Season | v2F hit rate | v2F log-loss |
|---|---|---|
| 2023-24 | 62.9% | 0.8298 |
| 2024-25 | 57.1% | 0.9514 |
| 2025-26 | 58.3% | 0.8886 |
| 2026-27 (82 matches) | 57.3% | 0.9448 |

**Observations.**

- **The Liga F is much more predictable than the men's leagues**: v2F gets 59% right with results
  alone, against 49% for the same model on La Liga and Segunda (and a log-loss of 0.90 against
  1.02). A few teams (Barcelona above all, then Real Madrid and Atlético) win almost every match,
  home or away.
- **It never marks a draw**, like the men's models: draws are 20% of the matches but never the
  most likely result.
- **v1F against v2F** has no common match yet: the first ticket with public picks archived is
  jornada 21, scored once its draw is saved. v1F stays the Liga F model of the tickets until that
  comparison has enough matches; then the better one plans them.

## Releases

Each model version is released with a git tag in commitizen's `v<version>` format, using the
version in `pyproject.toml`:

| Tag | Models | Notes |
|---|---|---|
| [`v0.11.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.11.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 pi-ratings (independent), v1F public picks and **v2F pi-ratings (Liga F)**, tickets v3+v1F·D, v3+v1F·R and v3+v1F·B (v3+v1F·E in the backtest), market reference | **Liga F results from ESPN** (every match since 2022-23, fixtures included, downloaded by `invoke update` into `data/women/espn/`), linked to the women's matches of the official tickets through a team alias table. New model **v2F**: the pi-ratings on the Liga F, evaluated walk-forward week by week on 802 matches (59.2% hit rate and log-loss 0.896 against 42.1% and 1.059 for the result-shares reference). Dashboards: v2F column on the next ticket and the official tickets, v1F-against-v2F comparison on the common ticket matches, Liga F data by season from ESPN. |
| [`v0.10.2`](https://github.com/jmquintana79/quiniela/releases/tag/v0.10.2) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 pi-ratings (independent), v1F public picks (Liga F), tickets v3+v1F·D, v3+v1F·R and v3+v1F·B (v3+v1F·E in the backtest), market reference | Fix: the table-of-contents anchors take a `toc-` prefix, so they no longer clash with the ids the pages fill in (the **Standings** of `current.html` lost its title and buttons when switching league). |
| [`v0.10.1`](https://github.com/jmquintana79/quiniela/releases/tag/v0.10.1) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 pi-ratings (independent), v1F public picks (Liga F), tickets v3+v1F·D, v3+v1F·R and v3+v1F·B (v3+v1F·E in the backtest), market reference | LAE data from `invoke update` with one click: it checks whether the next ticket or a played draw is missing, opens the LAE pages in your browser and picks the files up from Downloads after a click on the **Quiniela bookmark** (`tools/lae_bookmarklet.js`, installed from `dashboards/bookmarklet.html`; replaces the console script); `--no-browser` only warns. |
| [`v0.10.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.10.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 pi-ratings (independent), **v1F public picks (Liga F)**, tickets v3+v1F·D, v3+v1F·R and v3+v1F·B (v3+v1F·E in the backtest), market reference | **Liga F apart**: its history (`data/women/liga_f.parquet`: the women's matches of the official tickets with results, and the public picks of every next ticket archived in `data/lae/public/`), its own model counter (**v1F**, the public's picks, `WOMEN_MODELS`) and evaluation (`quiniela.evaluation.women`, with a result-shares reference); every dashboard shows men's and Liga F figures apart, and a **table of contents** after the header. The tickets are named **v3+v1F·D, ·R, ·B, ·E**: v3 on the men's matches and v1F on the Liga F ones. |
| [`v0.9.2`](https://github.com/jmquintana79/quiniela/releases/tag/v0.9.2) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 pi-ratings (independent), tickets v3·D, v3·R and v3·B (v3·E in the backtest), market reference | Evaluation: new **hit rate by result** table (1, X, 2) for every model and for v3·D and v3·R, where a double or triple is a hit for every result it covers (draws: v3 3.7%, v3·D 21%, v3·R 32%); `evaluation.html` reordered: metrics, hit rate by result, contingency tables, then the rest. README documents the consensus test of the models (no gain). |
| [`v0.9.1`](https://github.com/jmquintana79/quiniela/releases/tag/v0.9.1) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 pi-ratings (independent), tickets v3·D, v3·R and v3·B (v3·E in the backtest), market reference | Evaluation: **contingency tables** (observed result × predicted sign) of every model, with the league and season filters, and of the v3 tickets on the official tickets (observed result × marked signs: 1, X, 2, 1X, 12, X2, 1X2), before the metrics; **macro-F1** as an informative metric (v3 0.374, the highest; never bold). The euro backtest stores each ticket's picks and results. |
| [`v0.9.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.9.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, **v4 pi-ratings** (independent), tickets v3·D, v3·R and v3·B (v3·E in the backtest), market reference | New notation **model · ticket**: the tickets are decision layers over a model, so v4 → **v3·D** (direct), v4R → **v3·R** (reduced), v5 → **v3·B** (big win) and **v3·E** (prudent expected prize), freeing the version numbers for models. New model **v4 pi-ratings** (Constantinou & Fenton, 2013: home and away ratings learned from goal differences, promoted teams take the ratings of the ones they replace, ordered logistic mapping): best independent model (log-loss 1.0178 against 1.0215 for Dixon-Coles, 30 single-ticket prizes against 27), but it adds nothing to v3's pool on top of the market, so v3 and the tickets are unchanged. Dashboards: models first (v1 to v4), then the v3 tickets. |
| [`v0.8.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.8.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, **v4D direct ticket and v4R reduced ticket**, v5 money ticket (big-win mode in the dashboards), market reference | v4 split into **v4D** (direct doubles and triples, the former v4) and **v4R** (the same 48 bets with the official reductions as candidates: covering codes with the official guarantees, exact hit distribution of the best column, direct doubles and triples on top). Backtest: v4R 191 prize tickets in 376 matchdays against 156 for v4D (planned 50.0%, real 50.8%), and 49% of the money back in 414 real draws against 35%. Both side by side in every dashboard (columns, banners, prices table, official tickets with R marks, traffic lights, evaluation and euros). |
| [`v0.7.2`](https://github.com/jmquintana79/quiniela/releases/tag/v0.7.2) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 ticket planner, v5 money ticket (big-win mode in the dashboards), market reference | New `prizes.html` dashboard: every official draw since 2010-11 with its jackpot and, per category (14 + Pleno al 15 down to 10 hits), the prize per winner and the number of winners, plus a density chart per category on a common log euro axis (1 € to 15 M €). `season.html` matchdays table shows the prize won in euros by v3, v4 and v5. Roadmap and TODO list only pending work. |
| [`v0.7.1`](https://github.com/jmquintana79/quiniela/releases/tag/v0.7.1) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 ticket planner, v5 money ticket (big-win mode in the dashboards), market reference | Traffic lights in `current.html` (green / amber / red by the percentile among past real tickets): v5 by its chance of 10,000 € or more and of 13+ hits, v4 by its chance of 11+ hits, with the rule in small print and a red warning that none makes the quiniela profitable. The money backtest stores the planned hit chances. |
| [`v0.7.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.7.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 ticket planner, v5 money ticket (big-win mode in the dashboards), market reference | Pleno al 15 of the official ticket in `current.html` (Dixon-Coles' goals categories for a La Liga / Segunda match, the public's most marked ones otherwise); price of each ticket under the Pleno and hits of the 14 matches in the table footer; sales deadline in large type, without the countdown; `season.html` official tickets with what v1 to v5 marked, the Pleno pick and each ticket's hits and price. |
| [`v0.6.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.6.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market + public picks**, v4 ticket planner, v5 money ticket (big-win mode in the dashboards), market reference | The next official ticket in `current.html` (saved with `tools/lae_upcoming.js`): sales deadline with countdown, predictions without colours until results, notice of football-data's fixtures; v3 uses the public's picks for the matches no model covers (women's football, national teams), and v4/v5 plan on them; match category (M, F, O, U) and official-ticket flag in every table, official tickets in the historical and season dashboards, ticket matches by category in the evaluation; `current.html` shows only v3, v4 and v5. Backtests recalculated (unchanged: past public picks do not exist). |
| [`v0.5.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.5.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market**, v4 ticket planner, **v5 money ticket**, market reference | Official draws from LAE (real tickets, takings, winners and prizes since 2010-11); v4 aims at 11+ hits; v5 money ticket: prize model driven by the surprise of the results, ticket with the best expected prize within 48 bets, played only when it pays; prudent by default (only 10-13 hit prizes count), and in 414 real draws 2015-16 to 2025-26 it never found one worth playing (v4: 35% of the money back; v5 counting 14 hits and the Pleno al 15: 40%). Dashboards: v5 column with its verdict and euros won; evaluation: real tickets in euros. README: when to run v3 (the odds arrive on Friday afternoon). |
| [`v0.4.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.4.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market**, **v4 ticket planner** (doubles and triples on v3), market reference | Confidence and margin of every prediction; chance of each prize before the matchday; v4 ticket planner (48 bets, best mix of doubles and triples for 10+ hits): 160 prizes in 376 backtest matchdays, against 130 with random doubles and 35 without them. Dashboards: confidence column and sorting, prize chances, v4 column; evaluation: hit rate by confidence tier and ticket strategies. |
| [`v0.3.0`](https://github.com/jmquintana79/quiniela/releases/tag/v0.3.0) | v1 head-to-head, v2 Dixon-Coles (independent), **v3 Dixon-Coles + market**, market reference | v3: log-linear pool of Dixon-Coles and the market odds, trained on out-of-sample forecasts, Dixon-Coles alone without odds. Best log-loss, Brier and RPS of the backtest, also better than the market; same hit rate as the market (50.0%). |
| [`v0.2.3`](https://github.com/jmquintana79/quiniela/releases/tag/v0.2.3) | v1 head-to-head, **v2 Dixon-Coles**, market reference | Dashboards split into `current.html` (current matchday with every model, the market and hit rates) and `season.html`; prediction cells show only the sign (probability on hover); played matches never listed as upcoming. |
| [`v0.2.2`](https://github.com/jmquintana79/quiniela/releases/tag/v0.2.2) | v1 head-to-head, **v2 Dixon-Coles**, market reference | `invoke update`: one task to download what is missing, predict the current season and rebuild the dashboards. |
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
│   ├── backtest/tickets.parquet             # backtest tickets with doubles and triples
│   ├── backtest/money.parquet               # real tickets of past draws played in euros
│   ├── lae/lae_<season>.json                # official draws saved from LAE (by hand)
│   ├── lae/upcoming.json                    # next official ticket (Quiniela bookmark)
│   ├── lae/public/<season>_j<NN>.json       # archive of every next ticket: Liga F public picks (back it up)
│   ├── women/liga_f.parquet                 # Liga F matches of the tickets with results, public picks and ESPN link
│   ├── women/espn/<season>.parquet          # every Liga F match from ESPN (since 2022-23)
│   └── cache/                               # raw CSV files cached by soccerdata
├── src/quiniela/
│   ├── config/paths.py      # filesystem paths used by the project
│   ├── io/                  # downloads: football-data (download), Liga F from ESPN (espn), LAE one-click (lae)
│   ├── data/                # match loaders, matchday inference, official draws (prizes), Liga F history (women)
│   ├── prediction/          # model registry, models (v1-v4, v1F, v2F), pipeline, confidence, ticket rules, planner, reduced, money
│   ├── evaluation/          # metrics, the multi-season backtest, the backtest in euros and the Liga F evaluation
│   ├── visualization/       # dashboard builder and HTML templates
│   ├── update.py            # `invoke update`: every step in one go
│   └── scripts.py           # `quiniela` command line entry point
├── tools/lae_bookmarklet.js # Quiniela bookmark: saves the next ticket and the draws from LAE pages
├── docs/api/                # generated API docs (not versioned)
├── tasks/                   # invoke tasks (update, download, predict, backtest, dashboards, dev)
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

## Roadmap

Pending improvements, in order of priority. The detailed plan (in Spanish) is in
[`TODO.todo`](TODO.todo); what is already done is in [Releases](#releases).

1. **Club strength ratings (medium priority).** soccerdata includes ClubElo, daily Elo ratings
   of European clubs with decades of history and current values. They include cup and
   European matches that Dixon-Coles does not see. The pre-match Elo difference would be one
   more input of v3 (one more term of the pool). Pending: check the API (it was down when
   tried), its Segunda coverage, a local copy of the ratings, and matching team names between
   sources. v3 shows that Dixon-Coles adds nothing once the odds are known, so Elo has to be
   judged the same way: by what it adds on top of the market.
2. **Playing against the crowd with the public's picks (low priority).** v3+v1F·E and v3+v1F·B already maximise
   the expected prize, with the market probabilities standing in for what the public marks.
   The official ticket file (`data/lae/upcoming.json`) holds the real share of bettors per
   sign: in close matches a slightly less likely but less popular sign can be worth more (e.g.
   X at 35% picked by 10% of bettors, against 1 at 40% picked by 80%). No official history of
   those shares exists, so it has to be built by saving the file every week; with it, the
   prize model of v3+v1F·E and v3+v1F·B could use the public's own picks instead of the market, judged by the
   euro backtest.

Other ideas:

- **Shots on target** (`HST`, `AST`) as a less noisy measure of performance than goals; the
  [pi-ratings](#v4--pi-ratings-pi_ratings) could learn from the shots-on-target difference
  instead of the goal difference.
- **Liga F: a v2F variant with shots on target** (pending, not implemented). ESPN gives the shots
  on target of each team in every Liga F match (`summary?event=<id>`, one request per match, so
  they would be cached like the scoreboard): the pi-ratings could learn from the shots-on-target
  difference, or from a mix of goals and shots, and be compared with v2F walk-forward.
- **Liga F: the tickets on v2F.** Once the v1F-against-v2F comparison on the ticket matches has
  enough matches, plan the Liga F matches of the tickets with the better of the two.
