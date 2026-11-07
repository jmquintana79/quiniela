"""Command line entry points.

Examples::

    quiniela download historical --start 2015 --end 2025
    quiniela download historical --start 2015  # up to the last completed season
    quiniela download current
    quiniela dashboards
"""

import argparse
import logging

from quiniela import __version__


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(prog="quiniela", description="Quiniela (1X2) predictor.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download match data.")
    targets = download.add_subparsers(dest="target", required=True)

    historical = targets.add_parser("historical", help="Download completed seasons.")
    historical.add_argument(
        "--start", type=int, required=True, help="Start year of the first season (2015 = 2015-16)."
    )
    historical.add_argument(
        "--end",
        type=int,
        default=None,
        help="Start year of the last season (inclusive). Defaults to the last completed one.",
    )

    targets.add_parser("current", help="Download the season in progress and upcoming fixtures.")

    commands.add_parser("dashboards", help="Build the HTML dashboards from the downloaded data.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the quiniela command line interface.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    # Imported here so `quiniela --help` does not pay soccerdata's import time
    from quiniela.io import download

    if args.command == "dashboards":
        from quiniela.visualization.dashboards import build_dashboards

        paths = build_dashboards()
    elif args.target == "historical":
        end = args.end if args.end is not None else download.current_season() - 1
        paths = download.download_historical(args.start, end)
    else:
        paths = download.download_current()
    print(f"{len(paths)} files written")
