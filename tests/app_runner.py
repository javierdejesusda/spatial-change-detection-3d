"""A fake pipeline runner for the tests of the web app."""

from collections.abc import Callable
from pathlib import Path

from report_scenes import one_of_each_report
from run_spy import fake_result

from spatial_change.domain import ChangeReport
from spatial_change.pipeline import PipelineResult, StageEvent

REPORT_PAGE = "<!doctype html><title>report</title>"


class FakeRunner:
    """Stands in for the pipeline: writes the four files and reports stages."""

    def __init__(
        self,
        report: ChangeReport | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.report = one_of_each_report() if report is None else report
        self.error = error
        self.calls: list[tuple[Path, Path, Path]] = []

    def __call__(
        self,
        before: Path,
        after: Path,
        out_dir: Path,
        progress: Callable[[StageEvent], None],
    ) -> PipelineResult:
        self.calls.append((before, after, out_dir))
        progress(StageEvent("keyframes", 1, 12, None))
        progress(StageEvent("keyframes", 1, 12, 2.5))
        if self.error is not None:
            raise self.error
        result = fake_result(out_dir, self.report, {"keyframes": 2.5})
        files = result.files
        files.changes.write_text(self.report.to_json(), encoding="utf-8")
        files.report.write_text(REPORT_PAGE, encoding="utf-8")
        files.recording.write_bytes(b"rrd")
        files.scene.write_bytes(b"glb")
        return result
