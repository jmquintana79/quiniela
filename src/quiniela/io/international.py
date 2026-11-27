"""Men's national-team results since 1872 (martj42/international_results on GitHub).

The dataset by Mart Jürisoo lists every men's full international match since
the first one (Scotland-England, 1872): date, teams, score, tournament, city,
country and whether the ground was neutral. It is updated within days of
every international window and is one CSV file of a few megabytes, so it is
downloaded whole on every update and saved in ``data/national/results.csv``.

``load_results`` reads it with the column names of the men's matches
(``home_team``, ``away_team``, ``FTHG``, ``FTAG``, ``FTR``) plus
``tournament`` and ``neutral``: the input of the national-team Elo (v1N,
``quiniela.prediction.elo_national``).
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from quiniela.config import NATIONAL_DIR

logger = logging.getLogger(__name__)

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RESULTS_FILE = "results.csv"

# Columns of the national-team matches in the project's tables
COLUMNS = ["date", "home_team", "away_team", "FTHG", "FTAG", "FTR", "tournament", "neutral"]

# Seconds to wait for the download
TIMEOUT = 60


def download_results(output_dir: Path = NATIONAL_DIR, url: str = RESULTS_URL) -> Path:
    """Download the whole results file (replacing the saved one only when it is valid).

    Args:
        output_dir: Folder of the results file.
        url: Address of the CSV file.

    Returns:
        The file written.

    Raises:
        ValueError: The downloaded file does not have the expected columns.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "quiniela"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - fixed URL
        content: bytes = response.read()
    header = content.split(b"\n", 1)[0].decode("utf-8")
    if not {"date", "home_team", "away_team", "home_score", "away_score"} <= set(header.split(",")):
        raise ValueError(f"Unexpected national-team results file: {header[:80]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RESULTS_FILE
    # Written next to the old file and renamed: a failed write never leaves half a file
    partial = path.with_suffix(".part")
    partial.write_bytes(content)
    partial.replace(path)
    logger.info("National-team results saved in %s", path)
    return path


def parse_results(raw: pd.DataFrame) -> pd.DataFrame:
    """The results file with the project's column names, played matches only.

    Args:
        raw: The CSV as published (``home_score``, ``away_score``...).

    Returns:
        ``COLUMNS``, sorted by date (the file's order within a day kept).
    """
    played = raw.dropna(subset=["home_score", "away_score"]).copy()
    played["date"] = pd.to_datetime(played["date"])
    played["FTHG"] = played["home_score"].astype(int)
    played["FTAG"] = played["away_score"].astype(int)
    played["FTR"] = np.select(
        [played["FTHG"] > played["FTAG"], played["FTHG"] == played["FTAG"]], ["H", "D"], "A"
    )
    played["neutral"] = played["neutral"].astype(str).str.upper().eq("TRUE")
    played = played.sort_values("date", kind="stable").reset_index(drop=True)
    return played[COLUMNS]


def load_results(output_dir: Path = NATIONAL_DIR) -> pd.DataFrame:
    """Every played national-team match downloaded.

    Args:
        output_dir: Folder of the results file.

    Returns:
        ``COLUMNS``; empty when nothing was downloaded.
    """
    path = output_dir / RESULTS_FILE
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return parse_results(pd.read_csv(path))
