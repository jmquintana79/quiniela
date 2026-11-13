from pathlib import Path

import pandas as pd
import pytest

from quiniela.evaluation import backtest
from tests.helpers import simulate_league

TEAMS = {f"T{i}": s for i, s in enumerate([0.5, 0.3, 0.1, 0.0, -0.1, -0.2, -0.3, -0.5])}


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    seasons = []
    for year in (2021, 2022):
        matches = simulate_league(TEAMS, f"{year}-08-14", rounds=14, seed=year)
        seasons.append(
            matches.assign(season=backtest.season_label(year), AvgH=2.2, AvgD=3.3, AvgA=3.5)
        )
    return pd.concat(seasons, ignore_index=True)


@pytest.fixture(scope="module")
def results(history: pd.DataFrame) -> pd.DataFrame:
    frames = [
        backtest.run_backtest(2022, 2022, key, history=history, n_jobs=1)
        for key in ("head_to_head", "market")
    ]
    return pd.concat(frames, ignore_index=True)


def test_predicts_every_match_of_the_season(results: pd.DataFrame) -> None:
    assert set(results["model"]) == {"head_to_head", "market"}
    assert (results.groupby("model").size() == 14 * 4).all()
    assert results["season"].eq("2022-23").all()
    assert results["FTR"].notna().all()  # real results attached


def test_no_look_ahead(history: pd.DataFrame) -> None:
    # Fake results inside the backtested season must not change its first matchday
    first_day = history.loc[history["season"] == "2022-23", "date"].min()
    tampered = history.copy()
    later = (tampered["season"] == "2022-23") & (tampered["date"] > first_day)
    tampered.loc[later, "FTR"] = "A"

    def first_matchday(data: pd.DataFrame) -> pd.DataFrame:
        out = backtest.run_backtest(2022, 2022, "head_to_head", history=data, n_jobs=1)
        return out[out["date"] == first_day][["game", "p_H"]].reset_index(drop=True)

    pd.testing.assert_frame_equal(first_matchday(history), first_matchday(tampered))


def test_summarize(results: pd.DataFrame) -> None:
    overall = backtest.summarize(results).set_index("model")
    assert overall.loc["market", "matches"] == 56
    assert 0 <= overall.loc["head_to_head", "hit_rate"] <= 1
    assert {"log_loss", "brier", "rps", "missing"} <= set(overall.columns)
    by_season = backtest.summarize(results, ["season"])
    assert by_season["season"].unique().tolist() == ["2022-23"]
    assert "prizes" not in backtest.summarize(results, ["league"]).columns


def test_unknown_range(history: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No downloaded seasons"):
        backtest.run_backtest(1990, 1991, history=history, n_jobs=1)


def test_save_replaces_covered_models_and_seasons(results: pd.DataFrame, tmp_path: Path) -> None:
    backtest.save_backtest(results, tmp_path)
    again = results[results["model"] == "market"].assign(prediction="D")
    backtest.save_backtest(again, tmp_path)

    saved = backtest.load_backtest(tmp_path)
    assert len(saved) == len(results)
    assert saved.loc[saved["model"] == "market", "prediction"].eq("D").all()
    assert backtest.load_backtest(tmp_path / "missing").empty


def _planner_backtest(matchdays: int = 2) -> pd.DataFrame:
    """Backtest rows of the planner model: full 14-match La Liga tickets."""
    import numpy as np

    rng = np.random.default_rng(5)
    rows = []
    for matchday in range(1, matchdays + 1):
        probs = rng.dirichlet([5, 3, 3], size=14)
        for i in range(14):
            rows.append(
                {
                    "model": "market_blend",
                    "league": "SP1",
                    "season": "2022-23",
                    "game": f"{matchday}-{i}",
                    "matchday": matchday,
                    "date": pd.Timestamp("2022-08-13") + pd.Timedelta(weeks=matchday),
                    "home_team": f"H{matchday}{i}",
                    "away_team": f"A{matchday}{i}",
                    "prediction": "HDA"[int(probs[i].argmax())],
                    "p_H": probs[i, 0],
                    "p_D": probs[i, 1],
                    "p_A": probs[i, 2],
                    "FTR": rng.choice(["H", "D", "A"], p=probs[i]),
                }
            )
    return pd.DataFrame(rows)


def test_ticket_strategies() -> None:
    tickets = backtest.ticket_strategies(_planner_backtest(), n_jobs=1)
    counts = tickets.groupby("strategy").size()
    assert counts["random"] == 2 * backtest.RANDOM_DRAWS
    assert (counts.drop("random") == 2).all()
    assert set(counts.index) == set(backtest.TICKET_STRATEGIES)
    single = tickets[tickets["strategy"] == "single"]
    assert single["bets"].eq(1).all()
    assert tickets.loc[tickets["strategy"] != "single", "bets"].le(48).all()
    # The optimal plan has the best planned chance of a prize with the same budget
    planned = tickets.groupby(["strategy", "matchday"])["p_prize"].mean().unstack()
    assert (planned.loc["optimal"] >= planned.loc["heuristic"] - 1e-12).all()
    assert (planned.loc["v4"] >= planned.loc["optimal"] - 1e-12).all()

    summary = backtest.summarize_tickets(tickets).set_index("strategy")
    assert list(summary.index) == list(backtest.TICKET_STRATEGIES)
    assert (summary["tickets"] == 2).all()
    assert summary.loc["single", "bets"] == 1
    assert {"14", "13", "12", "11", "10", "prizes", "planned_prize"} <= set(summary.columns)


def test_ticket_strategies_without_the_planner_model(results: pd.DataFrame) -> None:
    assert backtest.ticket_strategies(results, n_jobs=1).empty


def test_save_backtest_also_saves_tickets(tmp_path: Path) -> None:
    backtest.save_backtest(_planner_backtest(1), tmp_path)
    tickets = backtest.load_tickets(tmp_path)
    assert len(tickets) == len(backtest.TICKET_STRATEGIES) - 1 + backtest.RANDOM_DRAWS
    assert backtest.load_tickets(tmp_path / "missing").empty


def test_ticket_picks() -> None:
    predictions = _planner_backtest(1)
    tickets = backtest.ticket_strategies(predictions, n_jobs=1)
    picks = backtest.ticket_picks(tickets)
    assert len(picks) == 14
    assert set(picks["game"]) == set(predictions["game"])
    v4 = tickets[tickets["strategy"] == "v4"].iloc[0]
    marked = sum(len(p) for p in picks["picks"])
    assert marked == 14 + v4["doubles"] + 2 * v4["triples"]
    assert backtest.ticket_picks(tickets.drop(columns=["picks"])).empty
