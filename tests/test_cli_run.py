import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from spatial_change import pipeline
from spatial_change.cli import main
from spatial_change.domain import ChangeReport, Sim3
from spatial_change.pipeline import (
    Models,
    OutputFiles,
    PipelineResult,
    RunConfig,
    StageEvent,
)
from spatial_change.pipeline.outputs import OUTPUT_FILES


class Spy:
    """Stands in for ``run_videos`` and records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def __call__(
        self,
        before_path: Path,
        after_path: Path,
        out_dir: Path,
        config: RunConfig,
        models: Models,
        progress: Callable[[StageEvent], None] | None = None,
        *,
        overwrite: bool = False,
    ) -> PipelineResult:
        self.calls.append(
            {
                "before": before_path,
                "after": after_path,
                "out": out_dir,
                "config": config,
                "models": models,
                "overwrite": overwrite,
            }
        )
        if progress is not None:
            progress(StageEvent("voting", 4, 12, None))
            progress(StageEvent("voting", 4, 12, 4.23))
        if self.error is not None:
            raise self.error
        files = OutputFiles(*(out_dir / n for n in OUTPUT_FILES))
        report = ChangeReport(changes=[], unobserved_regions=[], metadata={})
        return PipelineResult(
            report, (), files, {"voting": 4.23}, Sim3.identity()
        )


@pytest.fixture(name="spy")
def spy_fixture(monkeypatch: pytest.MonkeyPatch) -> Spy:
    spy = Spy()
    monkeypatch.setattr(pipeline, "run_videos", spy)
    monkeypatch.setattr(pipeline, "default_models", lambda config: "models")
    return spy


def test_run_calls_the_pipeline_with_the_paths_and_the_defaults(
    spy: Spy, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "out"

    code = main(["run", "a.mp4", "b.mp4", "--out", str(out)])

    assert code == 0
    (call,) = spy.calls
    assert (call["before"], call["after"], call["out"]) == (
        Path("a.mp4"),
        Path("b.mp4"),
        out,
    )
    assert call["config"] == RunConfig()
    assert call["models"] == "models"
    assert call["overwrite"] is False
    captured = capsys.readouterr()
    assert "[4/12] voting: 4.2 s" in captured.err
    assert "0 changes" in captured.out
    assert str(out / "changes.json") in captured.out


def test_flags_override_the_defaults(spy: Spy, tmp_path: Path) -> None:
    code = main(
        [
            "run",
            "a.mp4",
            "b.mp4",
            "--out",
            str(tmp_path),
            "--describer",
            "template",
            "--device",
            "cpu",
            "--max-keyframes",
            "12",
            "--overwrite",
        ]
    )

    assert code == 0
    (call,) = spy.calls
    assert call["config"] == RunConfig(
        describer="template", device="cpu", max_keyframes=12
    )
    assert call["overwrite"] is True


def test_a_config_file_is_read_and_flags_win_over_it(
    spy: Spy, tmp_path: Path
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"sample_fps": 2.0, "max_keyframes": 20, "seed": 5}),
        encoding="utf-8",
    )

    code = main(
        [
            "run",
            "a.mp4",
            "b.mp4",
            "--out",
            str(tmp_path / "out"),
            "--config",
            str(path),
            "--max-keyframes",
            "10",
        ]
    )

    assert code == 0
    assert spy.calls[0]["config"] == RunConfig(
        sample_fps=2.0, max_keyframes=10, seed=5
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "a.mp4"],
        ["run", "a.mp4", "b.mp4"],
        ["run", "a.mp4", "b.mp4", "--out", "o", "--describer", "gpt"],
        ["run", "a.mp4", "b.mp4", "--out", "o", "--max-keyframes", "1"],
        ["run", "a.mp4", "b.mp4", "--out", "o", "--max-keyframes", "x"],
    ],
)
def test_usage_errors_exit_with_2(
    spy: Spy,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(arguments)

    assert excinfo.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert spy.calls == []


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ('{"max_keyframes": 1}', "max_keyframes"),
        ('{"unknown": 1}', "unknown"),
        ("{not json", "JSON"),
        (None, "missing.json"),
    ],
)
def test_a_bad_config_file_is_a_clean_usage_error(
    spy: Spy,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    content: str | None,
    fragment: str,
) -> None:
    path = tmp_path / "missing.json"
    if content is not None:
        path = tmp_path / "config.json"
        path.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["run", "a", "b", "--out", "o", "--config", str(path)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert fragment in err
    assert "Traceback" not in err
    assert spy.calls == []


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("video not found: a.mp4"),
        FileExistsError("out is not empty; pass --overwrite to write into it"),
        ValueError("no sharp frame in video: b.mp4"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_a_runtime_failure_is_one_line_and_exit_1(
    spy: Spy,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
) -> None:
    spy.error = error

    code = main(["run", "a.mp4", "b.mp4", "--out", str(tmp_path)])

    assert code == 1
    err = capsys.readouterr().err
    failure = [line for line in err.splitlines() if line.startswith("error")]
    assert failure == [f"error: {error}"]
    assert "Traceback" not in err


def test_run_help_exits_with_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--help"])

    assert excinfo.value.code == 0
    assert "--overwrite" in capsys.readouterr().out


def test_debug_lets_a_runtime_failure_through_with_its_traceback(
    spy: Spy, tmp_path: Path
) -> None:
    spy.error = RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        main(["run", "a.mp4", "b.mp4", "--out", str(tmp_path), "--debug"])


def test_debug_does_not_turn_a_bad_config_into_a_traceback(
    spy: Spy, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "a", "b", "--out", "o", "--max-keyframes", "1", "--debug"])

    assert excinfo.value.code == 2
