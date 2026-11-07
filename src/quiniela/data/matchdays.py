"""Group matches into league rounds and quiniela matchdays (*jornadas*).

football-data.co.uk does not publish round numbers, so they are inferred:

1. **League round** (per league): a match belongs to round
   ``max(home team's match count, away team's match count)``. When one team
   has a postponed game, its rival's count is still right, so most matches
   land in the correct round. Then every match is moved to the round whose
   median date is closest, so a rescheduled match joins the round played
   around the same dates.
2. **Quiniela matchday** (all leagues): a quiniela ticket mixes Primera and
   Segunda matches of the same week. Each La Liga round is paired with the
   Segunda round closest in time (at most ``MAX_PAIRING_DAYS`` apart). Rounds
   without a partner (e.g. Segunda during international breaks) form a
   matchday on their own. Matchdays are numbered from 1 in date order.
"""

import numpy as np
import pandas as pd

# Leagues in the order a quiniela ticket lists them; the first one anchors the pairing
LEAGUE_ORDER = ["SP1", "SP2"]

# Maximum distance between two league rounds paired into the same matchday
MAX_PAIRING_DAYS = 3


def league_rounds(matches: pd.DataFrame) -> pd.Series:
    """Infer the round number of every match of one league and season.

    Args:
        matches: Matches of a single league and season with ``date``,
            ``home_team`` and ``away_team`` columns.

    Returns:
        Round numbers (starting at 1), aligned with ``matches.index``.
    """
    if matches.empty:
        return pd.Series(dtype="int64", index=matches.index, name="round")

    ordered = matches.sort_values("date", kind="stable")
    # Running match count of each team, as home and as away side
    appearances = pd.concat(
        [
            pd.DataFrame({"row": ordered.index, "team": ordered["home_team"], "side": "home"}),
            pd.DataFrame({"row": ordered.index, "team": ordered["away_team"], "side": "away"}),
        ]
    )
    order = {row: i for i, row in enumerate(ordered.index)}
    appearances["position"] = appearances["row"].map(order)
    appearances = appearances.sort_values(["position", "side"])
    appearances["count"] = appearances.groupby("team").cumcount() + 1
    rounds = appearances.groupby("row")["count"].max().reindex(ordered.index)

    # Move every match to the round with the closest median date
    dates = ordered["date"].astype("int64")
    medians = dates.groupby(rounds).median()
    distances = np.abs(dates.to_numpy()[:, None] - medians.to_numpy()[None, :])
    closest = pd.Series(medians.index[distances.argmin(axis=1)], index=ordered.index)

    numbers = {r: i + 1 for i, r in enumerate(sorted(closest.unique()))}
    return closest.map(numbers).reindex(matches.index).rename("round")


def assign_matchdays(matches: pd.DataFrame) -> pd.DataFrame:
    """Assign the league round and the quiniela matchday of every match.

    Args:
        matches: Matches of one season with ``league``, ``date``,
            ``home_team`` and ``away_team`` columns.

    Returns:
        DataFrame aligned with ``matches.index`` with ``round`` (per league)
        and ``matchday`` (shared by all leagues) columns.
    """
    out = pd.DataFrame(index=matches.index, columns=["round", "matchday"], dtype="int64")
    if matches.empty:
        return out

    out["round"] = pd.concat(
        [league_rounds(group) for _, group in matches.groupby("league")]
    ).reindex(matches.index)

    # One unit per league round, placed at its median date
    units = (
        matches.assign(round=out["round"])
        .groupby(["league", "round"])["date"]
        .median()
        .rename("center")
        .reset_index()
    )
    anchor_league = next((lg for lg in LEAGUE_ORDER if lg in set(units["league"])), None)
    anchors = units[units["league"] == anchor_league]
    others = units[units["league"] != anchor_league]

    # Pair each other-league round with the closest anchor round, closest pairs first
    candidates = [
        (abs((o.center - a.center).days), o.Index, a.Index)
        for o in others.itertuples()
        for a in anchors.itertuples()
        if o.league != a.league and abs(o.center - a.center) <= pd.Timedelta(days=MAX_PAIRING_DAYS)
    ]
    group_of = {i: i for i in units.index}
    taken: set[tuple[int, str]] = set()
    for _, other, anchor in sorted(candidates):
        key = (anchor, units.at[other, "league"])
        if group_of[other] == other and key not in taken:
            group_of[other] = anchor
            taken.add(key)
    units["group"] = units.index.map(group_of)

    # Number matchdays by the date of their anchor (or standalone) round
    starts = units.groupby("group")["center"].min().sort_values()
    numbers = {group: i + 1 for i, group in enumerate(starts.index)}
    units["matchday"] = units["group"].map(numbers)

    lookup = units.set_index(["league", "round"])["matchday"]
    keys = pd.MultiIndex.from_arrays([matches["league"], out["round"]])
    out["matchday"] = lookup.reindex(keys).to_numpy()
    return out.astype("int64")
