from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from spatial_change import pipeline
from spatial_change.domain import ChangeReport, Sim3
from spatial_change.pipeline import (
    Models,
    OutputFiles,
    PipelineResult,
    RoomScene,
    RunConfig,
    StageEvent,
)
from spatial_change.pipeline.outputs import OUTPUT_FILES


class RunSpy:
    """Stands in for ``run_videos`` and records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.errors: dict[str, Exception] = {}

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
        cache_dir: Path | None = None,
    ) -> PipelineResult:
        self.calls.append(
            {
                "before": before_path,
                "after": after_path,
                "out": out_dir,
                "config": config,
                "models": models,
                "overwrite": overwrite,
                "cache_dir": cache_dir,
            }
        )
        if progress is not None:
            progress(StageEvent("voting", 4, 12, None))
            progress(StageEvent("voting", 4, 12, 4.23))
        error = self.errors.get(out_dir.name, self.error)
        if error is not None:
            raise error
        files = OutputFiles(*(out_dir / n for n in OUTPUT_FILES))
        report = ChangeReport(changes=[], unobserved_regions=[], metadata={})
        return PipelineResult(
            report,
            (),
            files,
            {"voting": 4.23},
            Sim3.identity(),
            RoomScene((), (), {}),
        )


def install_run_spy(monkeypatch: pytest.MonkeyPatch) -> RunSpy:
    """Replaces the entry points of the pipeline with a spy.

    Args:
        monkeypatch: The fixture that undoes the replacement.

    Returns:
        The spy that records every ``run_videos`` call; ``default_models``
        returns the text ``models``.
    """
    spy = RunSpy()
    monkeypatch.setattr(pipeline, "run_videos", spy)
    monkeypatch.setattr(pipeline, "default_models", lambda config: "models")
    return spy
