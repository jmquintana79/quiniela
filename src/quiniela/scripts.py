"""Command line entry points.

Examples:

    quiniela update                     # data, predictions and dashboards
    quiniela update --backtest          # also re-run the backtest
    quiniela download historical --start 2015 --end 2025
    quiniela download historical --start 2015  # up to the last completed season
    quiniela download current
    quiniela predict                    # last matchday, every model
    quiniela predict --matchday 12 --model dixon_coles
    quiniela predict --all
    quiniela backtest --start 2015      # every model over past seasons
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

    predict = commands.add_parser("predict", help="Predict matchdays of the current season.")
    predict.add_argument("--model", default="all", help="Model key, or 'all' (default).")
    which = predict.add_mutually_exclusive_group()
    which.add_argument(
        "--matchday", type=int, default=None, help="Matchday to predict (default: the last one)."
    )
    which.add_argument("--all", action="store_true", help="Predict every matchday.")

    backtest = commands.add_parser(
        "backtest", help="Walk-forward predictions of past seasons, to evaluate the models."
    )
    backtest.add_argument(
        "--start", type=int, default=2015, help="Start year of the first season (default: 2015)."
    )
    backtest.add_argument(
        "--end", type=int, default=None, help="Start year of the last season (default: last one)."
    )
    backtest.add_argument("--model", default="all", help="Model key, or 'all' (default).")

    commands.add_parser("dashboards", help="Build the HTML dashboards from the downloaded data.")

    update = commands.add_parser(
        "update", help="Download what is missing, predict the current season, rebuild dashboards."
    )
    update.add_argument(
        "--backtest", action="store_true", help="Also re-run the backtest of every model."
    )
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

    if args.command == "update":
        from quiniela.update import run_update

        report = run_update(run_backtest=args.backtest)
        print(report.summary())
        return
    if args.command == "dashboards":
        from quiniela.visualization.dashboards import build_dashboards

        paths = build_dashboards()
    elif args.command == "predict":
        from quiniela.prediction.pipeline import run

        paths = run(args.model, matchday=args.matchday, all_matchdays=args.all)
    elif args.command == "backtest":
        from quiniela.evaluation.backtest import (
            load_tickets,
            run_backtest,
            save_backtest,
            summarize,
            summarize_tickets,
        )

        results = run_backtest(args.start, args.end, args.model)
        paths = [save_backtest(results)]
        columns = ["model", "matches", "hit_rate", "log_loss", "brier", "rps", "prizes"]
        print(summarize(results)[columns].round(4).to_string(index=False))
        # Real tickets in euros (needs the LAE draws in data/lae)
        import pandas as pd

        from quiniela.evaluation.money import save_money, summarize_money

        money_path = save_money()
        if money_path is not None:
            paths.append(money_path)
            euros = summarize_money(pd.read_parquet(money_path))
            print(
                euros[["strategy", "played", "cost", "payout", "net", "roi"]]
                .round(2)
                .to_string(index=False)
            )
        # Doubles and triples with the planner's model (saved with the backtest)
        tickets = summarize_tickets(load_tickets())
        if not tickets.empty:
            ticket_columns = ["strategy", "bets", "tickets", "prizes", "14", "13", "12", "11"]
            ticket_columns += ["10", "mean_hits", "planned_prize", "prize_rate"]
            print(tickets[ticket_columns].round(3).to_string(index=False))
    elif args.target == "historical":
        end = args.end if args.end is not None else download.current_season() - 1
        paths = download.download_historical(args.start, end)
    else:
        paths = download.download_current()
    print(f"{len(paths)} files written")
