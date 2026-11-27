from collections.abc import Iterator

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_local_national_or_club_data(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never read the downloaded internationals or ESPN club matches of ``data/``.

    A test that needs them sets its own (``monkeypatch.setattr`` again; the
    tests of the loaders themselves call ``monkeypatch.undo()`` first); the
    dashboards' caches of their evaluations are emptied around every test.
    """
    from quiniela.io import espn_clubs, international
    from quiniela.visualization import dashboards

    monkeypatch.setattr(
        international, "load_results", lambda *_: pd.DataFrame(columns=international.COLUMNS)
    )
    monkeypatch.setattr(
        espn_clubs, "load_club_matches", lambda *_: pd.DataFrame(columns=espn_clubs.COLUMNS)
    )
    for cached in (
        dashboards._national_ticket_probs,
        dashboards._club_evaluation,
        dashboards._matched_tickets,
    ):
        cached.cache_clear()
    yield
    # A test may have replaced them (no cache then)
    for cached in (
        dashboards._national_ticket_probs,
        dashboards._club_evaluation,
        dashboards._matched_tickets,
    ):
        getattr(cached, "cache_clear", lambda: None)()
