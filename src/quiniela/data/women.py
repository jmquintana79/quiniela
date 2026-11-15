"""Liga F (women's league) history: the official ticket matches with results and public picks.

No source of Liga F results is used yet: the history is what the quiniela
itself publishes.

- **Results** come from the official draws saved from LAE (``data/lae/``):
  every ticket match marked ``(F)`` with its score.
- **Public picks** ("Pronósticos más apostados", the share of bettors marking
  1, X and 2) are only on the betting page of the next ticket, so they are
  kept by archiving every ``data/lae/upcoming.json`` in ``data/lae/public/``
  (one file per season and jornada). They are the v1F model
  (``quiniela.prediction.public_picks``); without them a match has no v1F
  prediction.

``liga_f_history`` joins both by season, jornada and ticket position, and
``save_liga_f`` writes the table to ``data/women/liga_f.parquet``.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

from quiniela.config import LAE_DIR, PUBLIC_DIR, WOMEN_DIR
from quiniela.data.prizes import (
    UPCOMING_FILE,
    UpcomingTicket,
    load_tickets,
    load_upcoming,
    load_upcoming_file,
    match_category,
)

logger = logging.getLogger(__name__)

# File of the saved Liga F history
LIGA_F_FILE = "liga_f.parquet"

# Columns of the Liga F history
LIGA_F_COLUMNS = [
    "season",
    "jornada",
    "position",
    "date",
    "home",
    "away",
    "FTHG",
    "FTAG",
    "FTR",
    "public_H",
    "public_D",
    "public_A",
]


def ticket_season(closes: pd.Timestamp) -> str:
    """Season of a ticket from its sales deadline (seasons start in August).

    Args:
        closes: When the ticket's sales close.

    Returns:
        The season label, e.g. ``"2026-27"``.
    """
    start = closes.year if closes.month >= 8 else closes.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def archive_upcoming(lae_dir: Path = LAE_DIR, public_dir: Path = PUBLIC_DIR) -> Path | None:
    """Keep a copy of the saved next ticket, so its public picks are not lost.

    The copy is named after the ticket's season and jornada; a newer save of
    the same jornada replaces it (the percentages move until sales close).

    Args:
        lae_dir: Folder with ``upcoming.json``.
        public_dir: Archive folder.

    Returns:
        The archived file, or ``None`` without a valid ``upcoming.json``.
    """
    ticket = load_upcoming(lae_dir)
    if ticket is None:
        return None
    public_dir.mkdir(parents=True, exist_ok=True)
    target = public_dir / f"{ticket_season(ticket.closes)}_j{ticket.jornada:02d}.json"
    shutil.copyfile(lae_dir / UPCOMING_FILE, target)
    logger.info("Public picks of jornada %d archived in %s", ticket.jornada, target)
    return target


def _public_rows(ticket: UpcomingTicket) -> pd.DataFrame:
    """Women's matches of a saved ticket with the public's share of each sign."""
    rows = ticket.matches[ticket.matches["position"] <= 14].copy()
    rows["category"] = [
        match_category(h, a, None) for h, a in zip(rows["lae_home"], rows["lae_away"], strict=True)
    ]
    rows = rows[rows["category"] == "F"]
    return pd.DataFrame(
        {
            "season": ticket_season(ticket.closes),
            "jornada": ticket.jornada,
            "position": rows["position"].astype(int),
            "home": rows["lae_home"],
            "away": rows["lae_away"],
            "public_H": rows["public_1"].astype(float),
            "public_D": rows["public_X"].astype(float),
            "public_A": rows["public_2"].astype(float),
            "closes": ticket.closes.tz_localize(None),
        }
    )


def load_public_picks(public_dir: Path = PUBLIC_DIR, lae_dir: Path = LAE_DIR) -> pd.DataFrame:
    """Public picks of every Liga F match of the archived tickets (and the current one).

    Args:
        public_dir: Archive of next-ticket files.
        lae_dir: Folder with the current ``upcoming.json``.

    Returns:
        One row per women's match: ``season``, ``jornada``, ``position``,
        ``home``, ``away``, ``public_H``, ``public_D``, ``public_A`` (percent)
        and ``closes``.
    """
    paths = sorted(public_dir.glob("*.json")) if public_dir.exists() else []
    tickets = [load_upcoming_file(path) for path in paths]
    current = load_upcoming(lae_dir)
    frames = [_public_rows(t) for t in [*tickets, current] if t is not None]
    if not frames:
        return pd.DataFrame(columns=[*LIGA_F_COLUMNS[:3], "home", "away", "closes"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["season", "jornada", "position"], keep="last")


def liga_f_history(lae_dir: Path = LAE_DIR, public_dir: Path = PUBLIC_DIR) -> pd.DataFrame:
    """Every Liga F match of the official tickets, with its result and public picks.

    Args:
        lae_dir: Folder with the saved LAE draws and ``upcoming.json``.
        public_dir: Archive of next-ticket files.

    Returns:
        ``LIGA_F_COLUMNS`` sorted by date: the result is empty for matches not
        played yet (the next ticket) and the public picks for the tickets
        whose betting page was not saved.
    """
    tickets = load_tickets(lae_dir)
    played = pd.DataFrame(columns=LIGA_F_COLUMNS)
    if not tickets.empty:
        tickets = tickets[tickets["position"] <= 14].copy()
        women = [
            match_category(h, a, None) == "F"
            for h, a in zip(tickets["lae_home"], tickets["lae_away"], strict=True)
        ]
        tickets = tickets[women]
        played = pd.DataFrame(
            {
                "season": tickets["season"],
                "jornada": tickets["jornada"].astype("Int64"),
                "position": tickets["position"].astype(int),
                "date": pd.to_datetime(tickets["kickoff"]),
                "home": tickets["lae_home"],
                "away": tickets["lae_away"],
                "FTHG": tickets["FTHG"],
                "FTAG": tickets["FTAG"],
                "FTR": tickets["sign"],
            }
        )
    picks = load_public_picks(public_dir, lae_dir)
    picks = picks.assign(jornada=picks["jornada"].astype("Int64"))
    keys = ["season", "jornada", "position"]
    out = played.merge(
        picks.drop(columns=["home", "away", "closes"]), on=keys, how="outer", suffixes=("", "_p")
    )
    # Matches only in the next ticket (not played yet): names and date from the picks
    pending = picks.set_index(keys)
    index = pd.MultiIndex.from_frame(out[keys])
    out["home"] = out["home"].fillna(pd.Series(pending["home"].reindex(index).to_numpy()))
    out["away"] = out["away"].fillna(pd.Series(pending["away"].reindex(index).to_numpy()))
    out["date"] = pd.to_datetime(out["date"]).fillna(
        pd.Series(pd.to_datetime(pending["closes"].reindex(index).to_numpy()))
    )
    out = out.reindex(columns=LIGA_F_COLUMNS)
    return out.sort_values(["date", "position"]).reset_index(drop=True)


def save_liga_f(women_dir: Path = WOMEN_DIR, lae_dir: Path = LAE_DIR) -> Path | None:
    """Archive the next ticket and save the Liga F history.

    Args:
        women_dir: Folder of the Liga F history.
        lae_dir: Folder with the saved LAE draws and ``upcoming.json``.

    Returns:
        The saved file, or ``None`` when there are no Liga F matches yet.
    """
    archive_upcoming(lae_dir, lae_dir / "public")
    history = liga_f_history(lae_dir, lae_dir / "public")
    if history.empty:
        return None
    women_dir.mkdir(parents=True, exist_ok=True)
    path = women_dir / LIGA_F_FILE
    history.to_parquet(path, index=False)
    logger.info("Liga F history: %d matches saved in %s", len(history), path)
    return path


def load_liga_f(women_dir: Path = WOMEN_DIR) -> pd.DataFrame:
    """The saved Liga F history (empty when it was never saved)."""
    path = women_dir / LIGA_F_FILE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=LIGA_F_COLUMNS)
