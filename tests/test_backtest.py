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
