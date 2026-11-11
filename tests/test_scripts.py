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


def test_predict_matchday_and_all_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["predict", "--matchday", "3", "--all"])


def test_backtest_command_parses() -> None:
    from quiniela.scripts import _build_parser

    args = _build_parser().parse_args(["backtest", "--start", "2020", "--model", "market"])
    assert (args.start, args.end, args.model) == (2020, None, "market")
