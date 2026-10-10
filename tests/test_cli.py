import importlib.metadata

import pytest

from spatial_change import __version__
from spatial_change.cli import main


def test_version_flag_prints_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"spatial-change {__version__}"


def test_no_command_prints_usage_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


def test_console_script_targets_main() -> None:
    (entry_point,) = importlib.metadata.entry_points(
        group="console_scripts", name="spatial-change"
    )

    assert entry_point.load() is main
