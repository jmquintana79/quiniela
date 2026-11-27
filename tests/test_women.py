import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quiniela.data import women
from quiniela.prediction.public_picks import PublicPicksModel
from tests.test_prizes import _draw, _match, _upcoming_file


def _lae(folder: Path) -> Path:
    matches = [
        _match(1, "Valladolid (m)", "Castellón (m)", "1", "2 - 0", "2026-09-06 12:00:00"),
        _match(11, "Alavés (f)", "Eibar (f)", "X", "1 - 1", "2026-09-06 12:00:00"),
    ]
    draw = _draw("900", "2026-09-06", matches)
    draw["temporada"], draw["jornada"] = "2026-2027", 4
    (folder / "lae_2026-27.json").write_text(json.dumps([draw]))
    _upcoming_file(folder)
    return folder


def test_archive_and_history(tmp_path: Path) -> None:
    lae = _lae(tmp_path)
    archived = women.archive_upcoming(lae, lae / "public")
    assert archived is not None and archived.name == "2026-27_j21.json"
    # Without ESPN data every women's match of the tickets counts
    history = women.liga_f_history(lae, lae / "public", espn_dir=None)
    # The played Liga F match of jornada 4 and the pending one of jornada 21 (with picks)
    assert history["jornada"].tolist() == [4, 21]
    played, pending = history.iloc[0], history.iloc[1]
    assert played["FTR"] == "D" and np.isnan(played["public_H"])
    assert pd.isna(pending["FTR"]) and pending["public_H"] == 35 and pending["home"] == "Alavés (F)"
    path = women.save_liga_f(tmp_path / "women", lae)
    assert path is not None and len(women.load_liga_f(tmp_path / "women")) == 2


def test_ticket_season() -> None:
    assert women.ticket_season(pd.Timestamp("2026-11-14")) == "2026-27"
    assert women.ticket_season(pd.Timestamp("2027-05-20")) == "2026-27"


def test_public_picks_model() -> None:
    X = pd.DataFrame({"public_H": [14.0, None], "public_D": [17.0, None], "public_A": [69.0, None]})
    model = PublicPicksModel().fit(X, pd.Series(["H", "H", "D", "A"]))
    probs = model.predict_proba(X)
    assert np.allclose(probs[0], [0.14, 0.17, 0.69])
    # Without picks: the shares of the training results
    assert np.allclose(probs[1], [0.5, 0.25, 0.25])
    assert model.predict(X).tolist() == ["A", "H"]
    assert model.has_picks(X).tolist() == [True, False]


def test_espn_candidates_and_link() -> None:
    names = {"Atlético Madrid", "Madrid CFF", "Real Madrid", "Granadilla Tenerife", "Eibar"}
    assert women.espn_candidates("At. Madrid (f)", names) == {"Atlético Madrid"}
    assert women.espn_candidates("Madrid (f)", names) == {"Madrid CFF"}
    assert women.espn_candidates("Real Madrid (F)", names) == {"Real Madrid"}
    assert women.espn_candidates("Las Planas (f)", names) == set()
    history = pd.DataFrame(
        {
            "home": ["Eibar (f)", "Las Planas (f)"],
            "away": ["At. Madrid (f)", "Eibar (f)"],
            "date": pd.to_datetime(["2026-11-14", "2026-11-14"]),
        }
    )
    espn_matches = pd.DataFrame(
        {
            "game": ["2026-11-15 Eibar-Atlético Madrid"],
            "home_team": ["Eibar"],
            "away_team": ["Atlético Madrid"],
            "date": pd.to_datetime(["2026-11-15 12:00"]),
        }
    )
    linked = women.link_espn(history, espn_matches)
    assert linked["espn_game"].tolist() == ["2026-11-15 Eibar-Atlético Madrid", None]


def test_results_backtest_walks_forward() -> None:
    from quiniela.evaluation.women import results_backtest, summarize_women
    from tests.helpers import simulate_league

    strengths = {"Strong": 0.9, "Good": 0.3, "Mid": 0.0, "Weak": -0.4, "Poor": -0.6, "Bad": -0.9}
    first = simulate_league(strengths, "2022-09-03", 20, league="LIGAF", seed=1).assign(
        season="2022-23"
    )
    second = simulate_league(strengths, "2023-09-02", 20, league="LIGAF", seed=2).assign(
        season="2023-24"
    )
    matches = pd.concat([first, second], ignore_index=True)
    predictions = results_backtest(matches)
    # The first season only trains; v2F, v3F and the reference predict the second one
    # (the simulated matches have no shots: v3F learns from the goals, like v2F)
    assert set(predictions["season"]) == {"2023-24"}
    assert set(predictions["model"]) == {"pi_ratings_f", "pi_ratings_shots_f", "liga_f_shares"}
    summary = summarize_women(predictions)
    scores = summary["results"]["all"]
    assert scores["pi_ratings_f"]["log_loss"] < scores["liga_f_shares"]["log_loss"]
    # Without v1F inputs there is no common comparison
    assert summary["common"] == {}


def test_history_keeps_only_the_liga_f(tmp_path: Path) -> None:
    from quiniela.io.espn import COLUMNS

    lae = _lae(tmp_path)
    espn = pd.DataFrame(
        [
            [
                "LIGAF",
                "2026-27",
                "2026-09-06 Alavés-Eibar",
                pd.Timestamp("2026-09-06 12:00"),
                "Alavés",
                "Eibar",
                1.0,
                1.0,
                "D",
                12.0,
                9.0,
                5.0,
                3.0,
                "STATUS_FULL_TIME",
                "1",
            ]
        ],
        columns=COLUMNS,
    )
    (tmp_path / "espn").mkdir()
    espn.to_parquet(tmp_path / "espn" / "2026-27.parquet")
    history = women.liga_f_history(lae, lae / "public", espn_dir=tmp_path / "espn")
    # Jornada 21's women's match has no ESPN Liga F fixture: another competition, left out
    assert history["jornada"].tolist() == [4]
    assert history["espn_game"].tolist() == ["2026-09-06 Alavés-Eibar"]


def test_load_public_shares(tmp_path: Path) -> None:
    archive = tmp_path / "public"
    archive.mkdir()
    matches = [
        {
            "position": i,
            "home": f"H{i} (M)",
            "away": f"A{i} (M)",
            "public": {"1": 60, "X": 25, "2": 15},
        }
        for i in range(1, 15)
    ]
    content = {"jornada": 22, "closes": "21/11/2026 - 14:00", "matches": matches}
    (archive / "2026-27_j22.json").write_text(json.dumps(content))
    # A ticket without the shares of every match is left out
    (archive / "2026-27_j23.json").write_text(
        json.dumps({**content, "jornada": 23, "matches": matches[:5]})
    )

    shares = women.load_public_shares(archive, tmp_path)

    assert list(shares) == [("2026-27", 22)]
    assert shares[("2026-27", 22)].shape == (14, 3)
    assert shares[("2026-27", 22)][0].tolist() == pytest.approx([0.6, 0.25, 0.15])
