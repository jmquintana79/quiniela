"""National-team matches of the official tickets, linked to the international results.

The quiniela puts national-team matches on the tickets of the international
windows (qualifiers, Nations League, friendlies, the Euro and the World
Cup). LAE writes the countries in Spanish (``Inglaterra``, ``Rep. Checa``,
sometimes with a men's mark, ``Gales (m)``); ``LAE_NAMES`` translates them to
the names of the results file (``quiniela.io.international``). Women's and
under-21 teams (``Fem``, ``S21``, ``(f)``) are not translated: the results
file only has the men's senior teams.

``ticket_national`` links every ticket match between two translated
countries to its match in the results file: same pair of teams, in either
order (LAE sometimes lists the away team first at a neutral ground), within
``MATCH_DAYS`` of the kick-off (or of the draw date when the ticket has no
kick-off).
"""

import re

import pandas as pd

# LAE (Spanish) name of a national team -> name in the international results file
LAE_NAMES: dict[str, str] = {
    "Albania": "Albania",
    "Alemania": "Germany",
    "Andorra": "Andorra",
    "Arabia Saudí": "Saudi Arabia",
    "Argelia": "Algeria",
    "Argentina": "Argentina",
    "Armenia": "Armenia",
    "Ary Macedonia": "North Macedonia",
    "Arym": "North Macedonia",
    "Australia": "Australia",
    "Austria": "Austria",
    "Azerbaiyán": "Azerbaijan",
    "Bélgica": "Belgium",
    "Bielorrusia": "Belarus",
    "Bolivia": "Bolivia",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-herzegovina": "Bosnia and Herzegovina",
    "Brasil": "Brazil",
    "Bulgaria": "Bulgaria",
    "Cabo Verde": "Cape Verde",
    "Camerún": "Cameroon",
    "Canada": "Canada",
    "Canadá": "Canada",
    "Chile": "Chile",
    "China": "China",
    "Chipre": "Cyprus",
    "Colombia": "Colombia",
    "Corea Del Sur": "South Korea",
    "Costa De Marfil": "Ivory Coast",
    "Costa Rica": "Costa Rica",
    "Croacia": "Croatia",
    "Curazao": "Curaçao",
    "Dinamarca": "Denmark",
    "Ecuador": "Ecuador",
    "Ee.uu": "United States",
    "Egipto": "Egypt",
    "Emiratos Árabes": "United Arab Emirates",
    "Escocia": "Scotland",
    "Eslovaquia": "Slovakia",
    "Eslovenia": "Slovenia",
    "España": "Spain",
    "Estados Unidos": "United States",
    "Estonia": "Estonia",
    "Finlandia": "Finland",
    "Francia": "France",
    "Fyr Macedonia": "North Macedonia",
    "Gales": "Wales",
    "Georgia": "Georgia",
    "Ghana": "Ghana",
    "Gibraltar": "Gibraltar",
    "Grecia": "Greece",
    "Haití": "Haiti",
    "Holanda": "Netherlands",
    "Honduras": "Honduras",
    "Hungría": "Hungary",
    "Inglaterra": "England",
    "Irak": "Iraq",
    "Irán": "Iran",
    "Irlanda": "Republic of Ireland",
    "Irlanda Del Norte": "Northern Ireland",
    "Islandia": "Iceland",
    "Islas Faroe": "Faroe Islands",
    "Islas Feroe": "Faroe Islands",
    "Israel": "Israel",
    "Italia": "Italy",
    "Jamaica": "Jamaica",
    "Japón": "Japan",
    "Jordania": "Jordan",
    "Kazajistán": "Kazakhstan",
    "Kazajstán": "Kazakhstan",
    "Kosovo": "Kosovo",
    "Leichtenstein": "Liechtenstein",
    "Letonia": "Latvia",
    "Liechtenstein": "Liechtenstein",
    "Lituania": "Lithuania",
    "Luxemburgo": "Luxembourg",
    "Macedonia": "North Macedonia",
    "Macedonia Del Norte": "North Macedonia",
    "Malí": "Mali",
    "Malta": "Malta",
    "Marruecos": "Morocco",
    "México": "Mexico",
    "Moldavia": "Moldova",
    "Montenegro": "Montenegro",
    "Nigeria": "Nigeria",
    "Noruega": "Norway",
    "Nueva Zelanda": "New Zealand",
    "Países Bajos": "Netherlands",
    "Panamá": "Panama",
    "Paraguay": "Paraguay",
    "Perú": "Peru",
    "Polonia": "Poland",
    "Portugal": "Portugal",
    "Qatar": "Qatar",
    "Rep  Irlanda": "Republic of Ireland",
    "Rep. Checa": "Czech Republic",
    "Rep. De Corea": "South Korea",
    "Rep. Irlanda": "Republic of Ireland",
    "República Checa": "Czech Republic",
    "República De Irlanda": "Republic of Ireland",
    "Republica China": "China",
    "Rumanía": "Romania",
    "Rusia": "Russia",
    "San Marino": "San Marino",
    "Senegal": "Senegal",
    "Serbia": "Serbia",
    "Sudáfrica": "South Africa",
    "Suecia": "Sweden",
    "Suiza": "Switzerland",
    "Túnez": "Tunisia",
    "Turquía": "Turkey",
    "Ucrania": "Ukraine",
    "Uruguay": "Uruguay",
    "Uzbekistán": "Uzbekistan",
    "Venezuela": "Venezuela",
}

# Days between the ticket's kick-off (or draw date) and the match in the results file
MATCH_DAYS = 4

# Men's mark LAE adds to some names since 2026-27 ("Gales (m)")
_MEN_MARK = re.compile(r"\s*\((m|M)\)\s*$")


def national_team(lae_name: object) -> str | None:
    """Name in the results file of a national team as LAE writes it.

    Args:
        lae_name: Team as written on the ticket.

    Returns:
        The results-file name, or ``None`` for a club, a women's or an
        under-21 team.
    """
    if not isinstance(lae_name, str):
        return None
    return LAE_NAMES.get(_MEN_MARK.sub("", lae_name).strip())


def is_national(lae_home: object, lae_away: object) -> bool:
    """Whether a ticket match is between two men's senior national teams."""
    return national_team(lae_home) is not None and national_team(lae_away) is not None


def _pair_key(home: str, away: str) -> str:
    """Key of a pair of teams, the same in both orders."""
    return "|".join(sorted((home, away)))


def _pair_keys(homes: pd.Series, aways: pd.Series) -> pd.Series:
    """``_pair_key`` of every match."""
    return pd.Series(
        [_pair_key(h, a) for h, a in zip(homes, aways, strict=True)], index=homes.index
    )


def ticket_national(tickets: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """National-team matches of the official tickets with their international result.

    Args:
        tickets: Ticket matches (``quiniela.data.prizes.load_tickets``:
            ``season``, ``jornada``, ``position``, ``draw_date``, ``kickoff``,
            ``lae_home``, ``lae_away``, ``sign``).
        results: International results (``quiniela.io.international.load_results``).

    Returns:
        One row per ticket match between two national teams (positions 1-14)
        found in the results: the ticket columns plus ``home_team`` and
        ``away_team`` (results-file names, in the ticket's order), ``date``
        (of the match), ``neutral``, ``tournament``, ``FTR`` (the ticket's
        official sign, or the result from the ticket's home team without it), ``swapped`` (the results file
        lists the teams the other way round) and ``game``.
    """
    columns = [*tickets.columns, "home_team", "away_team", "date", "neutral", "tournament"]
    columns += ["FTR", "swapped", "game"]
    rows = tickets[tickets["position"] <= 14].copy()
    rows["home_team"] = rows["lae_home"].map(national_team)
    rows["away_team"] = rows["lae_away"].map(national_team)
    rows = rows.dropna(subset=["home_team", "away_team"])
    if rows.empty or results.empty:
        return pd.DataFrame(columns=columns)
    when = pd.to_datetime(rows["kickoff"]).dt.normalize()
    rows["_when"] = when.fillna(pd.to_datetime(rows["draw_date"]).dt.normalize())
    # Both orders of a pair share one key: LAE may list the teams the other way round
    keys = _pair_keys(results["home_team"], results["away_team"])
    by_pair = dict(tuple(results.groupby(keys.to_numpy())))
    out = []
    for record in rows.to_dict("records"):
        key = _pair_key(record["home_team"], record["away_team"])
        candidates = by_pair.get(key, results.iloc[:0])
        near = candidates[(candidates["date"] - record["_when"]).abs().dt.days <= MATCH_DAYS]
        if near.empty:
            continue
        match = near.loc[(near["date"] - record["_when"]).abs().idxmin()]
        swapped = match["home_team"] != record["home_team"]
        result = match["FTR"]
        if swapped:
            result = {"H": "A", "A": "H"}.get(str(result), result)
        # The ticket's official sign wins (an abandoned match awarded to a team, for example)
        if record.get("sign") in ("H", "D", "A"):
            result = record["sign"]
        record.update(
            date=match["date"],
            neutral=bool(match["neutral"]),
            tournament=match["tournament"],
            FTR=result,
            swapped=bool(swapped),
            game=f"{match['date']:%Y-%m-%d} {record['home_team']}-{record['away_team']}",
        )
        out.append(record)
    if not out:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out)[columns].reset_index(drop=True)
