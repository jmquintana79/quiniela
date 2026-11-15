import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    history = women.liga_f_history(lae, lae / "public")
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
