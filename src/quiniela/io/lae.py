"""Loterías y Apuestas del Estado (LAE) data with one click: the next ticket and the draws.

LAE answers web browsers only (scripts get "Access Denied"), so its data is
never downloaded by a script. ``invoke update`` does everything else:

1. :func:`missing_lae` decides what is missing: the next ticket (no saved
   ``upcoming.json``, or the saved one already closed) and the draws of the
   season (no file yet, or a closed ticket whose jornada is not among them).
2. :func:`fetch_lae` opens those LAE pages in your own browser and waits:
   one click on the **Quiniela bookmark** (``tools/lae_bookmarklet.js``,
   installed once from ``dashboards/bookmarklet.html``) on each page saves its
   data to the Downloads folder, as any visitor would.
3. :func:`collect_downloads` moves the files from Downloads to ``data/lae/``
   (``upcoming.json`` and ``lae_<season>.json``) as soon as they appear. The
   next ticket is then archived by season and jornada with the Liga F history
   (``quiniela.data.women``).

Without a terminal (or with ``--no-browser``) nothing is opened and the
missing data is reported as a warning.
"""

import json
import logging
import select
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from quiniela.config import DASHBOARDS_DIR, LAE_DIR, ROOT_DIR
from quiniela.data.prizes import MADRID, UPCOMING_FILE, load_draws, load_upcoming

logger = logging.getLogger(__name__)

# Pages opened in the browser
BETTING_PAGE = "https://juegos.loteriasyapuestas.es/jugar/la-quiniela/apuesta"
RESULTS_URL = (
    "https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAQU&celebrados=true"
    "&fechaInicioInclusiva={start}0801&fechaFinInclusiva={end}0731"
)

# Where the browser saves the files, and the bookmarklet
DOWNLOADS_DIR = Path.home() / "Downloads"
BOOKMARKLET_JS = ROOT_DIR / "tools" / "lae_bookmarklet.js"
BOOKMARKLET_PAGE = "bookmarklet.html"

# A closed ticket's draw is asked for once its matches are over (results are published
# after the last match, usually a day and a half after the sales close)
RESULTS_AFTER = pd.Timedelta(hours=36)

# How long to wait for the click (seconds) and how often to look at Downloads
WAIT_SECONDS = 300
POLL_SECONDS = 2


@dataclass
class LaeNeeds:
    """LAE data missing locally.

    Attributes:
        upcoming: The next ticket is missing (or the saved one already closed).
        results: Seasons (start years) whose draws must be saved again.
        reasons: Why, for the messages.
    """

    upcoming: bool = False
    results: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        """Whether anything is missing."""
        return self.upcoming or bool(self.results)


def season_start(day: datetime | pd.Timestamp) -> int:
    """Start year of the season a date belongs to (seasons start in August)."""
    return day.year if day.month >= 8 else day.year - 1


def missing_lae(lae_dir: Path = LAE_DIR, now: datetime | None = None) -> LaeNeeds:
    """What LAE data is missing: the next ticket and the season's draws.

    Args:
        lae_dir: Folder with the saved LAE files.
        now: Reference time (Madrid). Defaults to now.

    Returns:
        The missing data.
    """
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=MADRID)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize(MADRID)
    needs = LaeNeeds()
    ticket = load_upcoming(lae_dir)
    if ticket is None:
        needs.upcoming = True
        needs.reasons.append("no next ticket saved")
    elif ticket.closes < now_ts:
        needs.upcoming = True
        needs.reasons.append(f"the saved ticket (jornada {ticket.jornada}) already closed")

    start = season_start(now_ts)
    if not (lae_dir / f"lae_{start}-{(start + 1) % 100:02d}.json").exists():
        needs.results.append(start)
        needs.reasons.append(f"no draws saved for {start}-{(start + 1) % 100:02d}")
    elif ticket is not None and ticket.closes + RESULTS_AFTER < now_ts:
        draws = load_draws(lae_dir)
        season = f"{season_start(ticket.closes)}-{(season_start(ticket.closes) + 1) % 100:02d}"
        played = draws[(draws["season"] == season) & (draws["jornada"] == ticket.jornada)]
        if played.empty:
            needs.results.append(season_start(ticket.closes))
            needs.reasons.append(f"the draw of jornada {ticket.jornada} is not saved")
    return needs


def _valid_upcoming(path: Path) -> int | None:
    """Jornada of a next-ticket file saved by the bookmarklet, or None if it is not one."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        if content.get("closes") and content.get("matches"):
            return int(content["jornada"])
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        pass
    return None


def _valid_draws(path: Path) -> str | None:
    """Season (``2026-27``) of a draws file saved by the bookmarklet, or None."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        temporada = str(content[0]["temporada"])  # e.g. "2026-2027"
        start = int(temporada[:4])
        if "partidos" in content[0]:
            return f"{start}-{(start + 1) % 100:02d}"
    except (ValueError, TypeError, KeyError, IndexError, OSError):
        pass
    return None


def collect_downloads(downloads_dir: Path = DOWNLOADS_DIR, lae_dir: Path = LAE_DIR) -> list[Path]:
    """Move the files saved by the bookmarklet from Downloads to ``data/lae/``.

    ``quiniela_upcoming*.json`` becomes ``upcoming.json`` (the newest jornada
    wins) and ``quiniela_lae_<season>*.json`` becomes ``lae_<season>.json``.
    Files that are not valid LAE data are left where they are.

    Args:
        downloads_dir: Folder where the browser saves downloads.
        lae_dir: Destination folder.

    Returns:
        The files written in ``lae_dir``.
    """
    if not downloads_dir.exists():
        return []
    lae_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    upcoming = [
        (jornada, path)
        for path in sorted(downloads_dir.glob("quiniela_upcoming*.json"))
        if (jornada := _valid_upcoming(path)) is not None
    ]
    if upcoming:
        _, newest = max(upcoming, key=lambda item: (item[0], item[1].stat().st_mtime))
        target = lae_dir / UPCOMING_FILE
        target.write_bytes(newest.read_bytes())
        moved.append(target)
        for _, path in upcoming:
            path.unlink()
    for path in sorted(downloads_dir.glob("quiniela_lae_*.json")):
        season = _valid_draws(path)
        if season is None:
            continue
        target = lae_dir / f"lae_{season}.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        moved.append(target)
    for path in moved:
        logger.info("Saved %s", path)
    return moved


def write_bookmarklet_page(output_dir: Path = DASHBOARDS_DIR) -> Path:
    """Write the page that installs the Quiniela bookmark (drag its link to the bookmarks bar).

    Args:
        output_dir: Folder of the page.

    Returns:
        The page written.
    """
    code = BOOKMARKLET_JS.read_text(encoding="utf-8")
    # Comments are dropped: a bookmark is one line of JavaScript
    body = "\n".join(line for line in code.splitlines() if not line.strip().startswith("//"))
    # The browser percent-decodes a javascript: link before running it
    href = "javascript:" + quote(body, safe="")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Quiniela bookmark</title>
<style>body{{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;line-height:1.5;padding:0 16px}}
a.bm{{display:inline-block;padding:8px 14px;border-radius:8px;background:#2a78d6;color:#fff;text-decoration:none;font-weight:600}}</style>
</head><body>
<h1>Quiniela bookmark</h1>
<p>Drag this button to your browser's bookmarks bar (once):</p>
<p><a class="bm" href="{href}">Quiniela</a></p>
<p>Then, when <code>invoke update</code> opens a Loterías y Apuestas del Estado page, click
the <strong>Quiniela</strong> bookmark on it: the betting page saves the next ticket and the
results page the season's draws. <code>invoke update</code> picks the files up from your
Downloads folder.</p>
</body></html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / BOOKMARKLET_PAGE
    path.write_text(page, encoding="utf-8")
    return path


def _enter_pressed() -> bool:
    """Whether Enter was pressed (non-blocking; only with a terminal)."""
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def fetch_lae(
    interactive: bool = True,
    lae_dir: Path = LAE_DIR,
    downloads_dir: Path = DOWNLOADS_DIR,
    wait_seconds: int = WAIT_SECONDS,
) -> tuple[list[Path], list[str]]:
    """Get the missing LAE data with one click per page in your browser.

    Args:
        interactive: Open the pages and wait for the click (needs a terminal).
        lae_dir: Folder of the LAE files.
        downloads_dir: Folder where the browser saves downloads.
        wait_seconds: How long to wait for the files.

    Returns:
        ``(saved, warnings)``: files written in ``lae_dir`` and what is still missing.
    """
    # Files saved earlier (e.g. a click after the previous update stopped waiting)
    saved = collect_downloads(downloads_dir, lae_dir)
    needs = missing_lae(lae_dir)
    if not needs.any:
        return saved, []
    pages = [BETTING_PAGE] if needs.upcoming else []
    pages += [RESULTS_URL.format(start=s, end=s + 1) for s in needs.results]
    if not interactive or not sys.stdin.isatty():
        return saved, [
            "LAE data missing (" + "; ".join(needs.reasons) + "): run `invoke update` in a "
            "terminal and click the Quiniela bookmark on the pages it opens."
        ]
    page = write_bookmarklet_page()
    print(
        "\nLoterías y Apuestas del Estado data missing: " + "; ".join(needs.reasons) + ".\n"
        f"Opening {len(pages)} page(s) in your browser: click the Quiniela bookmark on each one.\n"
        f"(First time? Open {page} and drag the link to your bookmarks bar.)\n"
        "Waiting for the downloads... press Enter to continue without them.",
        flush=True,
    )
    for url in pages:
        webbrowser.open(url)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        saved += collect_downloads(downloads_dir, lae_dir)
        needs = missing_lae(lae_dir)
        if not needs.any or _enter_pressed():
            break
        time.sleep(POLL_SECONDS)
    warnings = [] if not needs.any else ["LAE data still missing: " + "; ".join(needs.reasons)]
    return saved, warnings
