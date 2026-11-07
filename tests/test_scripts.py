import pytest

from quiniela import __version__
from quiniela.scripts import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_download_requires_target() -> None:
    with pytest.raises(SystemExit):
        main(["download"])
