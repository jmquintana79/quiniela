"""Club matches of the official tickets outside La Liga / Segunda, linked to ESPN.

LAE writes the clubs in Spanish and short (``Inter De Milán``, ``Manch.
United``, ``At. Paranaense``, ``Sheffield U.``); ESPN in English and long
(``Internazionale``, ``Manchester United``, ``Athletico-PR``, ``Sheffield
United``). There are thousands of clubs, so instead of a table of names a
ticket match is linked by **similarity** to the ESPN matches played around
its date (``MATCH_DAYS``): every name is normalised (accents, punctuation
and words such as ``FC`` or ``Club`` removed; Spanish exonyms such as
``Nápoles`` or ``Oporto`` translated by ``ALIASES``) and two names are alike
when the words of one start the words of the other (``Man`` / ``Manchester``)
or their letters mostly coincide. The ESPN match whose two teams are the most
alike, in either order, is taken when both are alike enough
(``MIN_SIMILARITY``), or when one is all but identical and the other close
(``STRONG_SIMILARITY``, ``WEAK_SIMILARITY``): a few days of matches leave
little room for mistakes.
"""

import difflib
import functools
import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from quiniela.data.national import is_national

# Spanish (or LAE) names of clubs -> the words ESPN uses, after normalising
ALIASES: dict[str, str] = {
    "psg": "paris saint germain",
    "inter de milan": "internazionale",
    "inter milan": "internazionale",
    "inter": "internazionale",
    "napoles": "napoli",
    "oporto": "porto",
    "brujas": "brugge",
    "salzburgo": "salzburg",
    "marsella": "marseille",
    "bolonia": "bologna",
    "estrella roja": "red star",
    "cska moscu": "cska moscow",
    "spartak moscu": "spartak moscow",
    "lokomotiv moscu": "lokomotiv moscow",
    "dinamo kiev": "dynamo kyiv",
    "basilea": "basel",
    "man city": "manchester city",
    "manch city": "manchester city",
    "manch united": "manchester united",
    "man united": "manchester united",
    "manchester utd": "manchester united",
    "at madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "r madrid": "real madrid",
    "rm castilla": "real madrid castilla",
    "sporting lisboa": "sporting cp",
    "sporting de lisboa": "sporting cp",
    "sporting portugal": "sporting cp",
    "sporting de portugal": "sporting cp",
    "bayern": "bayern munich",
    "leipzig": "rb leipzig",
    "milan": "ac milan",
    "roma": "as roma",
    "colonia": "cologne",
    "fc colonia": "cologne",
    "m gladbach": "monchengladbach",
    "gladbach": "monchengladbach",
    "copenhague": "kobenhavn",
    "fc copenhague": "kobenhavn",
    "burdeos": "bordeaux",
    "niza": "nice",
    "rennes": "stade rennais",
    "borussia m": "monchengladbach",
    "aik solna": "aik",
    "sport recife": "sport",
    "brann bergen": "brann",
    "hamkam": "hamarkameratene",
    "sporting c portugal": "sporting cp",
    "royale union sg": "union st gilloise",
    "kobenhavn": "kobenhavn",
    "aek atenas": "aek athens",
    "panathinaikos": "panathinaikos",
    "celta": "celta vigo",
    "athletic": "athletic club",
    "betis": "real betis",
    "at paranaense": "athletico pr",
    "atletico paranaense": "athletico pr",
    "at mineiro": "atletico mg",
    "atletico mineiro": "atletico mg",
    "at goianiense": "atletico goianiense",
    "atletico go": "atletico goianiense",
    "new york rb": "new york red bulls",
    "frankfurt": "eintracht frankfurt",
    "eintracht": "eintracht frankfurt",
    "wolfsburgo": "wolfsburg",
    "hamburgo": "hamburg",
    "sparta praga": "sparta prague",
    "slavia praga": "slavia prague",
    "plzen": "viktoria plzen",
    "legia varsovia": "legia warsaw",
    "gante": "gent",
    "amberes": "antwerp",
    "estambul": "istanbul basaksehir",
    "terek grozny": "akhmat grozny",
    "tottenham": "tottenham hotspur",
    "wolverhampton": "wolverhampton wanderers",
    "west ham": "west ham united",
    "newcastle": "newcastle united",
    "brighton": "brighton hove albion",
    "leicester": "leicester city",
    "leeds": "leeds united",
    "sheffield u": "sheffield united",
    "sheffield w": "sheffield wednesday",
    "blackburn r": "blackburn rovers",
    "qpr": "queens park rangers",
    "montreal imp": "montreal",
    "seattle s": "seattle sounders",
    "los angeles fc": "lafc",
    "malmoe": "malmo",
}

# Words left out when comparing names (club forms and articles)
STOP_WORDS = {
    "fc", "cf", "cd", "ud", "sd", "sc", "ac", "as", "ss", "if", "ifk", "fk", "sk", "bk", "afc",
    "club", "de", "del", "la", "rc", "rcd", "ca", "sv", "vfb", "vfl", "tsg", "cp", "sl", "ff",
    "bv", "ssc", "us", "calcio", "cfr", "ogc", "osc", "rsc", "krc", "kaa", "tsv", "1", "kv",
}  # fmt: skip

# Days between the ticket's kick-off (or draw date, without it) and the ESPN match
MATCH_DAYS = 2
DRAW_DAYS = 5

# Days after a draw from which its links are kept (its ESPN matches are all in by then)
SETTLED_DAYS = 10

# Similarity of both teams to accept an ESPN match, or of one team and the other
MIN_SIMILARITY = 0.6
STRONG_SIMILARITY = 0.95
WEAK_SIMILARITY = 0.5


@functools.lru_cache(maxsize=100_000)
def normalise(name: str) -> str:
    """Team name without accents, punctuation, men's / women's marks and aliases applied."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\((m|f)\)", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in ALIASES:
        return ALIASES[text]
    words = " ".join(_words(text))
    return ALIASES.get(words, text)


def _words(text: str) -> list[str]:
    """Words of a normalised name that are not ``STOP_WORDS`` (all of them if none is left)."""
    return [w for w in text.split() if w not in STOP_WORDS] or text.split()


@functools.lru_cache(maxsize=500_000)
def similarity(lae_name: str, espn_name: str) -> float:
    """How alike an LAE and an ESPN team name are, from 0 to 1.

    Args:
        lae_name: Team as written on the ticket.
        espn_name: Team as ESPN writes it.

    Returns:
        1 for the same normalised name; otherwise the larger of the share of
        the LAE words that start an ESPN word (or are started by one, three
        letters at least), lowered when the numbers of words differ, and the
        share of coinciding letters (``difflib``).
    """
    first, second = normalise(lae_name), normalise(espn_name)
    if first == second:
        return 1.0
    words, others = _words(first), _words(second)

    def alike(word: str, other: str) -> bool:
        return (
            word == other
            or (len(word) >= 3 and other.startswith(word))
            or (len(other) >= 3 and word.startswith(other))
        )

    share = sum(any(alike(w, o) for o in others) for w in words) / len(words)
    balance = min(len(words), len(others)) / max(len(words), len(others))
    letters = difflib.SequenceMatcher(None, " ".join(words), " ".join(others)).ratio()
    return max(share * (0.75 + 0.25 * balance), letters)


def is_club_match(lae_home: object, lae_away: object) -> bool:
    """Whether a ticket match may be between two men's clubs (not women's, not national teams)."""
    if not isinstance(lae_home, str) or not isinstance(lae_away, str):
        return False
    text = f"{lae_home} {lae_away}".lower()
    if "(f)" in text or "fem" in text or re.search(r"\bs\d\d\b", text):
        return False
    return not is_national(lae_home, lae_away)


def link_espn(rows: pd.DataFrame, espn: pd.DataFrame) -> pd.DataFrame:
    """The ESPN match of each ticket club match.

    Args:
        rows: Ticket matches (``lae_home``, ``lae_away``, ``kickoff`` and
            ``draw_date``, the date used without a kick-off).
        espn: ESPN club matches (``quiniela.io.espn_clubs.load_club_matches``).

    Returns:
        Indexed like ``rows``: ``espn_id`` (``None`` when no match is alike
        enough), ``swapped`` (ESPN lists the teams the other way round) and
        ``similarity`` (of the less alike team).
    """
    out = pd.DataFrame(
        {"espn_id": None, "swapped": False, "similarity": 0.0}, index=rows.index
    ).astype({"espn_id": object})
    if rows.empty or espn.empty:
        return out
    days = espn["date"].dt.normalize()
    by_day = {day: group for day, group in espn.groupby(days)}
    kickoffs = pd.to_datetime(rows["kickoff"]).dt.normalize()
    draws = pd.to_datetime(rows["draw_date"]).dt.normalize()
    for index, home, away, kickoff, draw in zip(
        rows.index, rows["lae_home"], rows["lae_away"], kickoffs, draws, strict=True
    ):
        known = not pd.isna(kickoff)
        when, width = (kickoff, MATCH_DAYS) if known else (draw, DRAW_DAYS)
        if pd.isna(when):
            continue
        near = [
            by_day[day]
            for day in pd.date_range(
                when - pd.Timedelta(days=width), when + pd.Timedelta(days=width)
            )
            if day in by_day
        ]
        if not near:
            continue
        candidates = pd.concat(near)
        best = (0.0, None, False)
        for espn_id, espn_home, espn_away in zip(
            candidates["espn_id"], candidates["home_team"], candidates["away_team"], strict=True
        ):
            for swapped, (first, second) in (
                (False, (espn_home, espn_away)),
                (True, (espn_away, espn_home)),
            ):
                pair = (similarity(str(home), str(first)), similarity(str(away), str(second)))
                low, high = min(pair), max(pair)
                accepted = low >= MIN_SIMILARITY or (
                    high >= STRONG_SIMILARITY and low >= WEAK_SIMILARITY
                )
                if accepted and low > best[0]:
                    best = (low, espn_id, swapped)
        if best[1] is not None:
            out.loc[index] = [best[1], best[2], best[0]]
    return out


def _links_version() -> str:
    """Short hash of what decides a link: aliases, words, windows, thresholds, competitions."""
    from quiniela.io.espn_clubs import COMPETITIONS

    rules: list[object] = [sorted(ALIASES.items()), sorted(STOP_WORDS), MATCH_DAYS, DRAW_DAYS]
    rules += [MIN_SIMILARITY, STRONG_SIMILARITY, WEAK_SIMILARITY, COMPETITIONS]
    return hashlib.sha1(repr(rules).encode()).hexdigest()[:10]


def cached_links(rows: pd.DataFrame, espn: pd.DataFrame, cache_dir: Path | None) -> pd.DataFrame:
    """``link_espn`` of the ticket matches, reusing the links of the draws already played.

    Linking thousands of matches by similarity takes about half a minute, and a
    played draw's links never change: they are kept in
    ``<cache_dir>/ticket_links_<version>.parquet`` (a new version, a new file,
    when the linking rules change) for the draws older than ``SETTLED_DAYS``.

    Args:
        rows: Ticket matches (``season``, ``jornada``, ``position``,
            ``lae_home``, ``lae_away``, ``kickoff``, ``draw_date``).
        espn: ESPN club matches.
        cache_dir: Folder of the cache (``None``: no cache).

    Returns:
        As ``link_espn``.
    """
    if cache_dir is None or rows.empty:
        return link_espn(rows, espn)
    keys = ["season", "jornada", "position", "lae_home", "lae_away"]
    path = cache_dir / f"ticket_links_{_links_version()}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=[*keys, "espn_id"])
    known = rows[keys].reset_index().merge(cached, on=keys, how="inner").set_index("index")
    missing = rows.loc[~rows.index.isin(known.index)]
    fresh = link_espn(missing, espn)
    settled = pd.to_datetime(missing["draw_date"]) < pd.Timestamp.now() - pd.Timedelta(
        days=SETTLED_DAYS
    )
    if settled.any():
        new = missing.loc[settled, keys].join(fresh.loc[settled[settled].index])
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([cached, new.astype({"espn_id": object})], ignore_index=True).to_parquet(
            path, index=False
        )
    out = pd.concat([known[["espn_id", "swapped", "similarity"]], fresh])
    return out.reindex(rows.index).astype({"swapped": bool, "similarity": float})


def ticket_clubs(
    matched: pd.DataFrame, espn: pd.DataFrame, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Club matches of the official tickets outside La Liga / Segunda, with their ESPN match.

    Args:
        matched: Ticket matches linked to football-data
            (``quiniela.data.prizes.match_tickets``: ``league`` empty when the
            match is not a La Liga / Segunda one).
        espn: ESPN club matches.
        cache_dir: Folder where the links of the played draws are kept
            (``cached_links``); ``None`` links every match again.

    Returns:
        The ticket club matches (positions 1-14) found in ESPN: the ticket
        columns plus ``espn_id``, ``swapped``, ``similarity``, ``competition``,
        ``date`` (of the ESPN match), ``home_id`` and ``away_id`` (in the
        ticket's order), ``FTR`` (the ticket's sign, or the ESPN result from the
        ticket's home team) and ``game``.
    """
    rows = matched[(matched["position"] <= 14) & matched["league"].isna()]
    rows = rows[
        [is_club_match(h, a) for h, a in zip(rows["lae_home"], rows["lae_away"], strict=True)]
    ]
    columns = [*matched.columns, "espn_id", "swapped", "similarity", "competition", "date"]
    columns += ["home_id", "away_id", "FTR", "game"]
    links = cached_links(rows, espn, cache_dir)
    rows = rows.join(links.drop(columns=[c for c in links if c in rows]), how="inner")
    rows = rows[rows["espn_id"].notna()]
    if rows.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))
    found = espn.set_index("espn_id").loc[rows["espn_id"]]
    swapped = rows["swapped"].to_numpy(dtype=bool)
    rows = rows.assign(
        competition=found["competition"].to_numpy(),
        date=found["date"].to_numpy(),
        home_id=np.where(swapped, found["away_id"], found["home_id"]),
        away_id=np.where(swapped, found["home_id"], found["away_id"]),
    )
    result = found["FTR"].to_numpy(dtype=object)
    flipped = np.array([{"H": "A", "A": "H"}.get(r, r) for r in result], dtype=object)
    result = np.where(swapped, flipped, result)
    if "sign" in rows.columns:
        # The ticket's official sign wins (an abandoned match awarded to a team, for example)
        official = rows["sign"].where(rows["sign"].isin(["H", "D", "A"]))
        result = official.fillna(pd.Series(result, index=rows.index)).to_numpy(dtype=object)
    rows = rows.assign(
        FTR=result,
        game=pd.Series(found["date"].to_numpy()).dt.strftime("%Y-%m-%d").to_numpy()
        + " "
        + rows["lae_home"].astype(str)
        + "-"
        + rows["lae_away"].astype(str),
    )
    return rows[list(dict.fromkeys(columns))].reset_index(drop=True)
