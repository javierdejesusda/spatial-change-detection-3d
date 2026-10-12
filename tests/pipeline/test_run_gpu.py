import time
from pathlib import Path

import pytest

from spatial_change.domain import ChangeReport
from spatial_change.pipeline import (
    RunConfig,
    StageEvent,
    default_models,
    run_videos,
)
from spatial_change.pipeline.outputs import OUTPUT_FILES

pytestmark = pytest.mark.gpu

PAIR = (
    Path(__file__).parents[2]
    / "data"
    / "scenediff"
    / "living_room_3_living_room_4"
)
TARGET_SECONDS = 600.0


def test_the_whole_system_runs_on_a_real_video_pair(tmp_path: Path) -> None:
    import torch

    before, after = PAIR / "video1.mp4", PAIR / "video2.mp4"
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not (before.is_file() and after.is_file()):
        pytest.skip(f"demo videos not found in {PAIR}")
    config = RunConfig()
    events: list[StageEvent] = []

    started = time.perf_counter()
    result = run_videos(
        before,
        after,
        tmp_path / "run",
        config,
        default_models(config),
        events.append,
    )
    wall = time.perf_counter() - started

    assert [p.name for p in result.files.all] == list(OUTPUT_FILES)
    assert all(p.stat().st_size > 0 for p in result.files.all)
    report = ChangeReport.from_json(result.files.changes.read_text("utf-8"))
    assert report == result.report
    for change in report.changes:
        assert change.label
        assert change.description
        assert change.before_box or change.after_box
    assert len(result.facts) == len(report.changes)
    stages = {e.name for e in events if e.seconds is not None}
    assert stages == set(result.seconds)

    timings = ", ".join(f"{n} {s:.1f}" for n, s in result.seconds.items())
    kinds = sorted(c.type.value for c in report.changes)
    print(
        f"wall {wall:.0f} s of {TARGET_SECONDS:.0f} s, "
        f"peak VRAM {report.metadata['peak_vram_mb']:.0f} MB, "
        f"{len(report.changes)} changes {kinds}, "
        f"{len(report.unobserved_regions)} unobserved regions; "
        f"seconds per stage: {timings}"
    )
