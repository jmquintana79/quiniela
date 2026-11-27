from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quiniela.evaluation import ticket_metrics as tm


def _match(**values: object) -> dict[str, object]:
    row: dict[str, object] = {"p_H": 0.5, "p_D": 0.3, "p_A": 0.2, "FTR": "H"}
    row.update(values)
    return row


def test_match_metrics() -> None:
    rows = pd.DataFrame(
        [
            _match(
                m_H=0.4,
                m_D=0.3,
                m_A=0.3,
                s_H=0.8,
                s_D=0.1,
                s_A=0.1,
                v4_H=0.6,
                v4_A=0.1,
                AvgH=2.0,
                AvgD=3.0,
                AvgA=4.0,
                MaxH=2.2,
                MaxD=3.3,
                MaxA=4.4,
            ),
            _match(p_H=0.2, p_D=0.3, p_A=0.5, FTR="D"),
            _match(p_H=np.nan, p_D=np.nan, p_A=np.nan, FTR=None),
        ]
    )
    out = tm.match_metrics(rows)
    first = out.iloc[0]
    assert first["pick"] == "H" and first["fav_prob"] == pytest.approx(0.5)
    assert first["entropy"] == pytest.approx(
        -(0.5 * np.log(0.5) + 0.3 * np.log(0.3) + 0.2 * np.log(0.2))
    )
    assert first["strength"] == pytest.approx(0.5)
    assert first["edge_market"] == pytest.approx(np.log(0.5 / 0.4))
    # The public piles on the home win: its pick is overrated, the away win underrated
    assert first["public_pick"] == "H"
    assert first["edge_public"] == pytest.approx(np.log(0.5 / 0.8))
    assert first["value_sign"] == "D" and first["value"] == pytest.approx(3.0)
    assert first["spread"] == pytest.approx(np.log(1.1))
    assert first["hit"] == 1 and first["value_hit"] == 0
    assert first["log_loss"] == pytest.approx(-np.log(0.5))
    second = out.iloc[1]
    assert second["pick"] == "A" and second["hit"] == 0
    assert np.isnan(second["edge_public"]) and np.isnan(second["spread"])
    # No probabilities: no pick and no metrics
    third = out.iloc[2]
    assert pd.isna(third["pick"]) and np.isnan(third["fav_prob"]) and np.isnan(third["hit"])


def _ticket(shares: bool = True, men: int = 14) -> pd.DataFrame:
    rows = []
    for position in range(1, 15):
        row = _match(
            position=position,
            season="2026-27",
            jornada=21,
            draw_id="1",
            date=pd.Timestamp("2026-11-14"),
        )
        row["category"] = "M" if position <= men else "F"
        if shares:
            row.update(s_H=0.5, s_D=0.25, s_A=0.25)
        rows.append(row)
    return tm.match_metrics(pd.DataFrame(rows))


def test_draw_metrics() -> None:
    draws = tm.draw_metrics(_ticket(men=10), {"1": 1_000_000})
    row = draws.iloc[0]
    assert row["matched"] == 10 and row["public"] == 14
    assert row["expected_hits"] == pytest.approx(14 * 0.5)
    # Every pick (H at 0.5) is marked by half the public: no differentiation
    assert row["differentiation"] == pytest.approx(0.0)
    assert row["co_winners_14"] == pytest.approx(1_000_000 * 0.5**14)
    # Without the public's shares of every match
    partial = tm.draw_metrics(_ticket(shares=False)).iloc[0]
    assert np.isnan(partial["differentiation"]) and np.isnan(partial["co_winners_14"])


def test_walk_forward_lights_and_verdicts() -> None:
    n = tm.MIN_EARLIER + 3
    draws = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=n, freq="7D"), "p": np.arange(n, dtype=float)}
    )
    # Shuffled rows: the order is the date's, not the row's
    lights = tm.walk_forward_lights(draws.iloc[::-1].reset_index(drop=True), "p")[::-1]
    assert lights[: tm.MIN_EARLIER] == [None] * tm.MIN_EARLIER
    # Each new value is the highest so far: green
    assert lights[tm.MIN_EARLIER :] == ["green"] * 3
    assert tm.verdicts(["green", "amber", "red", None], ["red", "green", "amber", None]) == [
        "big",
        "small",
        "skip",
        None,
    ]
    # Too few league matches: no verdict
    assert tm.verdicts(["green"], ["green"], [tm.MIN_MATCHED - 1]) == [None]


def _money() -> pd.DataFrame:
    rows = []
    for draw_id in ("1", "2"):
        for strategy, cost, payout, hits in (
            ("single", 0.75, 0.0, 9),
            ("v3d", 36.0, 40.0, 12),
            ("v3r", 36.0, 0.0, 10),
            ("v3b", 48.0, 1_200.0 if draw_id == "2" else 0.0, 13),
            ("v3e", 0.0, 0.0, 0),
        ):
            rows.append(
                {
                    "draw_id": draw_id,
                    "strategy": strategy,
                    "cost": cost,
                    "payout": payout,
                    "net": payout - cost,
                    "best_hits": hits,
                    "p_hits": 0.05,
                    "p_big": 0.004,
                    "expected_net": -12.0 if strategy == "v3e" else np.nan,
                    "bets": 32 if strategy == "v3e" else 1,
                    "jackpot": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_money_kpis_and_add_kpis() -> None:
    kpis = tm.add_kpis(tm.money_kpis(_money()).set_index("draw_id"))
    assert kpis.loc["1", "net_v3d"] == pytest.approx(4.0)
    assert kpis.loc["1", "p_big_v3b"] == pytest.approx(0.004)
    # Expected return of the ticket v3+v2F·E planned (24 EUR), not of what it played (0)
    assert kpis.loc["1", "expected_return"] == pytest.approx((24 - 12) / 24)
    assert kpis.loc["1", "return_v3d"] == pytest.approx(40 / 36)
    assert kpis.loc["1", "hit12_v3d"] == 1 and kpis.loc["1", "hit13_v3b"] == 1
    assert kpis.loc["1", "big_v3b"] == 0 and kpis.loc["2", "big_v3b"] == 1


def test_quintiles() -> None:
    frame = pd.DataFrame(
        {"metric": np.arange(50, dtype=float), "kpi": np.repeat([0, 0, 1, 1, 1], 10)}
    )
    rows = tm.quintiles(frame, "metric", {"kpi": "kpi"})
    assert [r["n"] for r in rows] == [10] * 5
    assert [r["kpi"] for r in rows] == [0, 0, 1, 1, 1]
    assert rows[0]["low"] == 0 and rows[-1]["high"] == 49
    # Too few records for the groups
    assert tm.quintiles(frame.head(10), "metric", {"kpi": "kpi"}) == []


def test_season_draws() -> None:
    draws = tm.draw_metrics(_ticket())
    outcome = {"cost": 48.0, "payout": 60.0, "hits": 13, "value": 0.005, "signal": "green"}
    tickets = [
        {
            "draw_id": "1",
            "winners": "public",
            "outcomes": {
                key: {**outcome, "cost": 36.0 if key == "v3d" else 48.0} for key in tm.TICKETS
            },
        }
    ]
    out = tm.season_draws(tickets, draws)
    row = out.iloc[0]
    assert row["net_v3b"] == pytest.approx(12.0) and row["net_v3d"] == pytest.approx(24.0)
    assert row["p_big_v3b"] == pytest.approx(0.005) and row["light_v3b"] == "green"
    assert row["verdict"] == "big" and row["winners"] == "public"
    assert row["return_v3b"] == pytest.approx(60 / 48)


def test_ticket_matches_fills_the_public() -> None:
    matched = pd.DataFrame(
        {
            "draw_id": "7",
            "season": "2026-27",
            "jornada": 21,
            "position": [1, 2, 15],
            "draw_date": pd.Timestamp("2026-11-15"),
            "kickoff": pd.NaT,
            "lae_home": ["Betis (M)", "Alavés (F)", "España (M)"],
            "lae_away": ["Celta (M)", "Eibar (F)", "Italia (M)"],
            "sign": ["H", "A", None],
            "league": ["SP1", None, None],
            "game": ["g1", None, None],
        }
    )
    model = pd.DataFrame(
        {"league": ["SP1"], "game": ["g1"], "p_H": [0.6], "p_D": [0.25], "p_A": [0.15]}
    )
    odds = pd.DataFrame(
        {"league": ["SP1"], "game": ["g1"], "AvgH": [1.8], "AvgD": [3.6], "AvgA": [5.0]}
    )
    shares = {("2026-27", 21): np.tile([0.7, 0.2, 0.1], (14, 1))}
    out = tm.ticket_matches(matched, {"p": model, "m": None, "v4": None}, odds, shares)
    assert out["position"].tolist() == [1, 2]
    assert out["category"].tolist() == ["M", "F"]
    assert out.loc[0, "fav_prob"] == pytest.approx(0.6)
    assert out.loc[0, "edge_public"] == pytest.approx(np.log(0.6 / 0.7))
    # The women's match has no model: the public's picks stand in (as on the tickets)
    assert out.loc[1, "p_H"] == pytest.approx(0.7) and out.loc[1, "edge_public"] == pytest.approx(
        0.0
    )
    assert out.loc[1, "hit"] == 0


def test_upcoming_matches() -> None:
    from quiniela.prediction.pipeline import WOMEN_LEAGUE

    matches = pd.DataFrame(
        {
            "official_position": [2, 1, 15],
            "league": ["SP1", WOMEN_LEAGUE, "OTHER"],
            "game": ["a", "b", "c"],
            "home_team": ["X", "Y", "Z"],
            "away_team": ["U", "V", "W"],
            "FTR": None,
            "_public_H": [60.0, 30.0, None],
            "_public_D": [25.0, 30.0, None],
            "_public_A": [15.0, 40.0, None],
        }
    )
    planner = pd.DataFrame(
        {"game": ["a", "b"], "p_H": [0.5, 0.3], "p_D": [0.3, 0.3], "p_A": [0.2, 0.4]}
    )
    out = tm.upcoming_matches(matches, {"market_blend": planner}, "market_blend", "2026-27", 25)
    assert out["position"].tolist() == [1, 2]
    assert out["category"].tolist() == ["F", "M"]
    assert out.loc[1, "s_H"] == pytest.approx(0.6)
    assert out.loc[0, "pick"] == "A" and out["jornada"].eq(25).all()


def test_save_metrics(tmp_path: Path) -> None:
    rows = _ticket()
    paths = tm.save_metrics(rows, tm.draw_metrics(rows), tmp_path)
    assert [p.name for p in paths] == ["matches.parquet", "draws.parquet"]
    assert len(pd.read_parquet(paths[0])) == 14
