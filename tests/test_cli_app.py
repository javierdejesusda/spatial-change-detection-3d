import importlib.util
import json
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from spatial_change import app as app_package
from spatial_change.app import handlers
from spatial_change.cli import main
from spatial_change.pipeline import RunConfig


class FakeUi(ModuleType):
    """Stands in for ``spatial_change.app.ui`` and records the launch."""

    def __init__(self) -> None:
        super().__init__("spatial_change.app.ui")
        self.launches: list[dict[str, Any]] = []
        self.error: OSError | None = None

    def launch_app(self, **kwargs: Any) -> None:
        self.launches.append(kwargs)
        if self.error is not None:
            raise self.error


@pytest.fixture(name="ui")
def ui_fixture(monkeypatch: pytest.MonkeyPatch) -> FakeUi:
    fake = FakeUi()
    monkeypatch.setitem(sys.modules, "spatial_change.app.ui", fake)
    monkeypatch.setattr(app_package, "ui", fake, raising=False)
    return fake


PORT_ERROR = (
    "Cannot find empty port in range: {first}-{last}. You can specify a "
    "different port by setting the GRADIO_SERVER_PORT environment variable "
    "or passing the `server_port` parameter to `launch()`."
)
"""What gradio raises when no port in the range can be bound."""


@pytest.fixture(name="runners")
def runners_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[RunConfig, Path | None]]:
    made: list[tuple[RunConfig, Path | None]] = []

    def pipeline_runner(config: RunConfig, cache_dir: Path | None) -> str:
        made.append((config, cache_dir))
        return "runner"

    monkeypatch.setattr(handlers, "pipeline_runner", pipeline_runner)
    return made


def test_app_serves_on_this_computer_with_the_default_settings(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    code = main(["app"])

    runs = tmp_path / ".cache" / "spatial-change" / "app"
    assert code == 0
    assert ui.launches == [
        {"runner": "runner", "runs_root": runs, "port": None}
    ]
    assert runners == [(RunConfig(), runs / "cache")]


def test_port_and_runs_dir_reach_the_launch(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    tmp_path: Path,
) -> None:
    code = main(["app", "--port", "8123", "--runs-dir", str(tmp_path)])

    assert code == 0
    (launch,) = ui.launches
    assert launch["port"] == 8123
    assert launch["runs_root"] == tmp_path
    assert runners[0][1] == tmp_path / "cache"


def test_the_app_cannot_be_told_to_listen_elsewhere(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["app", "--host", "0.0.0.0"])

    assert excinfo.value.code == 2
    assert "--host" in capsys.readouterr().err
    assert ui.launches == []


def test_a_busy_port_is_named_and_the_port_flag_is_the_way_out(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ui.error = OSError(PORT_ERROR.format(first=8123, last=8123))

    code = main(["app", "--port", "8123"])

    assert code == 1
    assert capsys.readouterr().err == (
        "error: port 8123 is in use; pick another with --port\n"
    )


def test_no_free_port_at_all_still_points_to_the_port_flag(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ui.error = OSError(PORT_ERROR.format(first=7860, last=7959))

    code = main(["app"])

    assert code == 1
    assert capsys.readouterr().err == (
        "error: no free port was found; pick one with --port\n"
    )


def test_any_other_launch_failure_keeps_its_first_line(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ui.error = OSError("[Errno 13] Permission denied: '/runs'\nmore")

    code = main(["app"])

    assert code == 1
    assert capsys.readouterr().err == (
        "error: [Errno 13] Permission denied: '/runs'\n"
    )


def test_run_settings_reach_the_runner_and_flags_win_over_the_file(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"max_keyframes": 20, "seed": 5}), encoding="utf-8"
    )

    code = main(
        [
            "app",
            "--config",
            str(path),
            "--describer",
            "template",
            "--device",
            "cpu",
            "--max-keyframes",
            "10",
        ]
    )

    assert code == 0
    assert runners[0][0] == RunConfig(
        describer="template", device="cpu", max_keyframes=10, seed=5
    )


def test_a_bad_setting_is_a_usage_error_before_anything_launches(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["app", "--max-keyframes", "1"])

    assert excinfo.value.code == 2
    assert "max_keyframes" in capsys.readouterr().err
    assert ui.launches == []


@pytest.mark.parametrize("missing", ["gradio", "gradio_rerun"])
def test_without_the_app_extra_the_message_names_it(
    ui: FakeUi,
    runners: list[tuple[RunConfig, Path | None]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    find_spec = importlib.util.find_spec

    def without(name: str, package: str | None = None) -> ModuleSpec | None:
        return None if name == missing else find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", without)

    code = main(["app"])

    assert code == 1
    assert capsys.readouterr().err == (
        "error: the web app needs the app extra; "
        "add --extra app to your uv sync command\n"
    )
    assert ui.launches == []
    assert runners == []


def test_app_help_exits_with_zero_and_names_the_local_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["app", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "127.0.0.1" in out
    assert "--runs-dir" in out
