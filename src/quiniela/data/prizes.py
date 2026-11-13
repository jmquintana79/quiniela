"""Official quiniela draws: real tickets, takings, winners and prizes per category.

Source: the results service of Loterías y Apuestas del Estado (LAE), the same
JSON its website uses (one request per season)::

    https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAQU
        &celebrados=true&fechaInicioInclusiva=20150801&fechaFinInclusiva=20160731

The service only answers browsers (scripts get "Access Denied"), so its JSON is
saved by hand, one file per season, in ``data/lae/`` (see the README) and read
from there. Every file is a list of draws; repeated draws are dropped.

Each draw has the 15 matches of the real ticket (14 plus the Pleno al 15),
the takings, the number of winners and the prize of each category, and the
jackpot announced for the Pleno al 15. Prize rules, checked against every
draw since 2015-16: a bet costs 0.75 EUR and the takings are shared as 16% for
14 hits, 7.5% each for 13, 12 and 11 hits and 9% for 10 hits (not paid when it
would be below 1.50 EUR); the Pleno al 15 (14 hits plus the 15th match) gets
7.5% plus the announced jackpot.

Ticket matches are matched to football-data matches (``league`` and ``game``)
by season, teams (LAE names translated with ``TEAM_ALIASES``) and date, so
the models' predictions can be looked up for the real ticket. Matches from
other competitions (Champions League, national teams, Primera RFEF, summer
leagues...) stay unmatched.
"""

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quiniela.config import LAE_DIR

logger = logging.getLogger(__name__)

# Price of one bet (column) in EUR
BET_PRICE = 0.75

# Share of the takings of each category (hits -> share); the Pleno al 15 also gets the jackpot
PRIZE_SHARES: dict[int, float] = {14: 0.16, 13: 0.075, 12: 0.075, 11: 0.075, 10: 0.09}
SPECIAL_SHARE = 0.075

# The 10-hit prize is not paid below this amount (it goes to the jackpot)
MIN_10_HIT_PRIZE = 1.50

# LAE category number -> column suffix
CATEGORIES: dict[int, str] = {1: "special", 2: "14", 3: "13", 4: "12", 5: "11", 6: "10"}

# LAE team names (normalised: lower case, no accents or dots) -> football-data names
TEAM_ALIASES: dict[str, str] = {
    "athletic club": "Ath Bilbao",
    "athletic": "Ath Bilbao",
    "athletic club b": "Ath Bilbao B",
    "bilbao athletic": "Ath Bilbao B",
    "at madrid": "Ath Madrid",
    "atletico madrid": "Ath Madrid",
    "r madrid": "Real Madrid",
    "espanyol": "Espanol",
    "deportivo": "La Coruna",
    "rayo vallecano": "Vallecano",
    "real sociedad": "Sociedad",
    "r sociedad": "Sociedad",
    "real sociedad b": "Sociedad B",
    "r sociedad b": "Sociedad B",
    "sporting": "Sp Gijon",
    "racing": "Santander",
    "racing santander": "Santander",
    "racing de santander": "Santander",
    "r santander": "Santander",
    "r oviedo": "Oviedo",
    "r zaragoza": "Zaragoza",
    "r valladolid": "Valladolid",
    "cultural": "Cultural Leonesa",
    "reus": "Reus Deportiu",
    "extremadura": "Extremadura UD",
    "r majadahonda": "Rayo Majadahonda",
    "sevilla at": "Sevilla B",
    "racing ferrol": "Ferrol",
    "r ferrol": "Ferrol",
    "ucam": "UCAM Murcia",
}

# Some clubs changed name in football-data between seasons
ALTERNATIVE_NAMES: dict[str, list[str]] = {"Cultural Leonesa": ["Leonesa"]}

# A ticket match and its football-data match can be a few days apart (dates of
# the LAE ticket are the scheduled ones; rescheduled matches are left out)
MAX_DATE_GAP = pd.Timedelta(days=4)

SIGNS = {"1": "H", "X": "D", "2": "A"}

DRAW_COLUMNS = [
    "draw_id",
    "season",
    "jornada",
    "date",
    "bets",
    "takings",
    "jackpot",
    *[f"winners_{c}" for c in CATEGORIES.values()],
    *[f"prize_{c}" for c in CATEGORIES.values()],
]

TICKET_COLUMNS = [
    "draw_id",
    "season",
    "jornada",
    "position",
    "draw_date",
    "lae_home",
    "lae_away",
    "kickoff",
    "sign",
    "FTHG",
    "FTAG",
]


def normalise(name: str) -> str:
    """Team name in lower case without accents, dots or repeated spaces.

    Recent tickets mark men's teams with ``(m)``; it is dropped (a women's
    team, ``(f)``, keeps its mark and never matches a football-data team).
    """
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    plain = plain.lower().replace("(m)", " ")
    return " ".join(plain.replace(".", " ").split())


def _number(value: object) -> float:
    """LAE numbers: amounts with decimal comma (``"5227,17"``); empty means missing."""
    if value is None or value == "":
        return float("nan")
    return float(str(value).replace(".", "").replace(",", "."))


def _season(label: str) -> str:
    """LAE season (``"2015-2016"``) as the project's label (``"2015-16"``)."""
    start, end = label.split("-")
    return f"{start}-{end[-2:]}"


def _read_draws(lae_dir: Path) -> list[dict[str, Any]]:
    """Every draw of the saved LAE files, without repetitions."""
    draws: dict[str, dict[str, Any]] = {}
    for path in sorted(lae_dir.glob("*.json")):
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            logger.warning("Skipping %s: not a list of draws", path.name)
            continue
        for draw in content:
            draws[str(draw["id_sorteo"])] = draw
    return list(draws.values())


def load_draws(lae_dir: Path = LAE_DIR) -> pd.DataFrame:
    """One row per draw with its takings, jackpot, winners and prizes.

    Args:
        lae_dir: Folder with the saved LAE JSON files.

    Returns:
        ``DRAW_COLUMNS``: amounts in EUR, ``jackpot`` is the amount announced
        for the Pleno al 15 before the draw, ``winners_<c>`` and ``prize_<c>``
        (prize per winner) for ``c`` in ``special``, ``14`` ... ``10``.
        Sorted by date; empty without files.
    """
    rows = []
    for draw in _read_draws(lae_dir):
        if not draw.get("temporada") or not draw.get("escrutinio"):
            continue
        row: dict[str, Any] = {
            "draw_id": str(draw["id_sorteo"]),
            "season": _season(str(draw["temporada"])),
            "jornada": int(draw["jornada"]) if draw.get("jornada") is not None else None,
            "date": pd.Timestamp(str(draw["fecha_sorteo"])[:10]),
            "bets": _number(draw.get("apuestas")) if draw.get("apuestas") else float("nan"),
            # Takings and jackpot come in cents
            "takings": float(draw.get("recaudacion") or 0) / 100,
            "jackpot": float(draw.get("fondo_bote") or 0) / 100,
        }
        for entry in draw["escrutinio"]:
            suffix = CATEGORIES.get(int(entry["categoria"]))
            if suffix:
                row[f"winners_{suffix}"] = _number(entry.get("ganadores"))
                row[f"prize_{suffix}"] = _number(entry.get("premio"))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=DRAW_COLUMNS)
    draws = pd.DataFrame(rows).reindex(columns=DRAW_COLUMNS)
    return draws.sort_values("date").reset_index(drop=True)


def load_tickets(lae_dir: Path = LAE_DIR) -> pd.DataFrame:
    """The 15 matches of every real ticket.

    Args:
        lae_dir: Folder with the saved LAE JSON files.

    Returns:
        ``TICKET_COLUMNS``: ``position`` 1-14 (15 is the Pleno al 15),
        ``sign`` ``H``, ``D`` or ``A`` (``None`` for the Pleno al 15 or a void
        match), the score and the scheduled kick-off.
    """
    rows = []
    for draw in _read_draws(lae_dir):
        if not draw.get("temporada"):
            continue
        for match in draw.get("partidos") or []:
            score = str(match.get("marcador") or "").replace(" ", "").split("-")
            goals = [int(g) if g.isdigit() else np.nan for g in score] if len(score) == 2 else []
            rows.append(
                {
                    "draw_id": str(draw["id_sorteo"]),
                    "season": _season(str(draw["temporada"])),
                    "jornada": int(draw["jornada"]) if draw.get("jornada") is not None else None,
                    "position": int(match["posicion"]),
                    "draw_date": pd.Timestamp(str(draw["fecha_sorteo"])[:10]),
                    "lae_home": str(match["local"]).strip(),
                    "lae_away": str(match["visitante"]).strip(),
                    "kickoff": pd.to_datetime(match.get("fecha_completa"), errors="coerce"),
                    "sign": SIGNS.get(str(match.get("signo") or "").strip()),
                    "FTHG": goals[0] if goals else np.nan,
                    "FTAG": goals[1] if goals else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame(columns=TICKET_COLUMNS)
    return pd.DataFrame(rows, columns=TICKET_COLUMNS)


def match_tickets(tickets: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Find the football-data match of every ticket match.

    Args:
        tickets: Output of :func:`load_tickets`.
        matches: football-data matches with ``season``, ``league``, ``game``,
            ``date``, ``home_team`` and ``away_team``.

    Returns:
        ``tickets`` with ``league``, ``game``, ``home_team`` and
        ``away_team`` (``NaN`` for matches of other competitions or teams).
    """
    names: dict[str, dict[str, str]] = {}
    for season, group in matches.groupby("season"):
        teams = set(group["home_team"]) | set(group["away_team"])
        names[str(season)] = {normalise(str(team)): str(team) for team in teams}

    def translate(season: str, lae_name: str) -> str | None:
        known = names.get(season, {})
        key = normalise(lae_name)
        candidate = TEAM_ALIASES.get(key)
        options = [candidate] if candidate else []
        options += ALTERNATIVE_NAMES.get(candidate or "", [])
        for option in options:
            if normalise(option) in known:
                return known[normalise(option)]
        return known.get(key)

    out = tickets.copy()
    out["home_team"] = [
        translate(s, n) for s, n in zip(out["season"], out["lae_home"], strict=True)
    ]
    out["away_team"] = [
        translate(s, n) for s, n in zip(out["season"], out["lae_away"], strict=True)
    ]
    lookup = matches[["season", "league", "game", "date", "home_team", "away_team"]].rename(
        columns={"date": "match_date"}
    )
    merged = out.merge(lookup, on=["season", "home_team", "away_team"], how="left")
    # The same pairing can also be a cup match: keep it only when the dates agree
    # (some old tickets have no kick-off: the draw date, a day or two later, is used)
    reference = merged["kickoff"].dt.normalize().fillna(merged["draw_date"])
    gap = (merged["match_date"].dt.normalize() - reference).abs()
    far = gap.isna() | (gap > MAX_DATE_GAP)
    merged.loc[far, ["league", "game"]] = np.nan
    return merged.drop(columns="match_date")


def draw_coverage(matched: pd.DataFrame) -> pd.DataFrame:
    """How many of the 14 matches of each draw have a football-data match.

    Args:
        matched: Output of :func:`match_tickets`.

    Returns:
        One row per draw with ``matched`` (0-14).
    """
    first14 = matched[matched["position"] <= 14]
    counts = first14.groupby("draw_id")["game"].apply(lambda g: int(g.notna().sum()))
    return counts.rename("matched").reset_index()
