import dataclasses
import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pipeline_scenes import (
    DescribingVlm,
    Timeline,
    Tracked,
    synthetic_models,
    video_frames,
)
from visibility_scenes import SEEDS, scene_clouds

from spatial_change.domain import (
    ChangeReport,
    ChangeType,
    Epoch,
)
from spatial_change.pipeline import (
    STAGES,
    Models,
    PipelineResult,
    RunConfig,
    StageEvent,
    run_frames,
)
from spatial_change.pipeline.outputs import OUTPUT_FILES
from spatial_change.semantics import INDOOR_OBJECTS, validate_description
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticScenePair,
    generate_scene_pair,
)

CENTER_TOLERANCE_M = 0.35
EXPECTED_KIND = {
    ObjectStatus.MOVED: ChangeType.MOVED,
    ObjectStatus.ADDED: ChangeType.ADDED,
    ObjectStatus.REMOVED: ChangeType.REMOVED,
}
TEMPLATE = RunConfig(device="cpu", describer="template", voxel_size=None)
MODEL = RunConfig(device="cpu", describer="vlm", voxel_size=None)
SMALL_SCENE = SceneConfig(width=64, height=48, frames_per_epoch=6)
"""Runs in about a second and still reaches every stage on seed 0."""


@dataclasses.dataclass(eq=False)
class Run:
    result: PipelineResult
    timeline: Timeline
    events: list[StageEvent]
    pair: SyntheticScenePair
    vlm: DescribingVlm


def execute(seed: int, config: RunConfig, out_dir: Path) -> Run:
    pair, _, _ = scene_clouds(seed)
    timeline = Timeline()
    vlm = DescribingVlm()
    events: list[StageEvent] = []
    result = run_frames(
        video_frames(pair, Epoch.BEFORE),
        video_frames(pair, Epoch.AFTER),
        out_dir,
        config,
        synthetic_models(pair, timeline, lambda: vlm),
        events.append,
    )
    return Run(result, timeline, events, pair, vlm)


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[int], Run]:
    """Template runs by seed, each executed once for the whole module."""
    cache: dict[int, Run] = {}

    def run_for(seed: int) -> Run:
        if seed not in cache:
            cache[seed] = execute(seed, TEMPLATE, tmp_path_factory.mktemp("t"))
        return cache[seed]

    return run_for


@pytest.fixture(scope="module")
def template_run(runs: Callable[[int], Run]) -> Run:
    return runs(SEEDS[0])


@pytest.fixture(scope="module")
def vlm_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return execute(SEEDS[0], MODEL, tmp_path_factory.mktemp("v"))


def room_center(run: Run, center: np.ndarray) -> np.ndarray:
    return run.result.room_from_world.apply(center[None])[0]


def test_the_four_files_exist_and_the_report_reads_back(
    template_run: Run,
) -> None:
    files = template_run.result.files

    assert [p.name for p in files.all] == list(OUTPUT_FILES)
    assert all(p.stat().st_size > 0 for p in files.all)
    parsed = ChangeReport.from_json(files.changes.read_text("utf-8"))
    assert parsed == template_run.result.report


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_every_true_change_is_found_with_the_right_type_and_place(
    runs: Callable[[int], Run], seed: int
) -> None:
    template_run = runs(seed)
    changes = template_run.result.report.changes
    changed = [
        o
        for o in template_run.pair.objects
        if o.status is not ObjectStatus.UNCHANGED
    ]

    assert len(changes) == len(changed)
    for obj in changed:
        kind = EXPECTED_KIND[obj.status]
        found = [c for c in changes if c.type is kind]
        assert len(found) == 1, obj.object_id
        for truth, box in (
            (obj.box_before, found[0].before_box),
            (obj.box_after, found[0].after_box),
        ):
            if truth is None:
                assert box is None
                continue
            assert box is not None
            gap = np.linalg.norm(
                box.to_box().center - room_center(template_run, truth.center)
            )
            assert gap < CENTER_TOLERANCE_M, (obj.object_id, gap)


def test_every_change_is_complete_and_described(template_run: Run) -> None:
    result = template_run.result
    assert len(result.facts) == len(result.report.changes)

    for change, fact in zip(result.report.changes, result.facts, strict=True):
        assert change.label in INDOOR_OBJECTS
        assert change.description
        assert validate_description(change.description, fact).ok
        assert 0.0 < change.confidence <= 1.0
        assert change.before_box or change.after_box
        assert sum(change.evidence.model_dump().values()) > 0
        if change.type is ChangeType.MOVED:
            assert change.displacement_m is not None
            assert change.rotation_deg is not None
            assert change.before_box and change.after_box


def test_the_moved_object_keeps_its_true_displacement(
    template_run: Run,
) -> None:
    (moved,) = [
        c for c in template_run.result.report.changes if c.type.value == "moved"
    ]
    (truth,) = [
        o for o in template_run.pair.objects if o.status is ObjectStatus.MOVED
    ]

    assert moved.displacement_m == pytest.approx(truth.displacement_m, abs=0.1)


def test_unobserved_regions_are_reported_apart_from_the_changes(
    template_run: Run,
) -> None:
    report = template_run.result.report
    pair = template_run.pair
    zone = room_center(template_run, pair.unobserved_zone.center)

    assert report.unobserved_regions
    assert len({r.region_id for r in report.unobserved_regions}) == len(
        report.unobserved_regions
    )
    covering = [
        r
        for r in report.unobserved_regions
        if r.box.to_box().contains(zone[None])[0]
    ]
    assert covering
    assert all(r.epoch_seen is pair.zone_seen_in for r in covering)
    for obj_id in pair.zone_object_ids:
        truth = pair.object_by_id(obj_id)
        center = room_center(template_run, truth.box_before.center)  # type: ignore[union-attr]
        for change in report.changes:
            boxes = [b for b in (change.before_box, change.after_box) if b]
            assert all(
                np.linalg.norm(b.to_box().center - center) > 0.3 for b in boxes
            )


def test_the_progress_events_come_in_stage_order(template_run: Run) -> None:
    stages = [s for s in STAGES if s != "keyframes"]

    names = [(e.name, e.seconds is None) for e in template_run.events]

    assert names == [(s, start) for s in stages for start in (True, False)]
    assert all(e.total == len(STAGES) for e in template_run.events)
    seconds = template_run.result.seconds
    assert set(seconds) == set(stages)
    assert all(v >= 0.0 for v in seconds.values())


def test_models_load_one_at_a_time_in_order_and_all_unload(
    template_run: Run,
) -> None:
    timeline = template_run.timeline

    assert timeline.most_loaded == 1
    assert not timeline.loaded
    assert timeline.roles() == [
        "reconstructor",
        "encoder",
        "segmenter",
        "encoder",
        "labeler",
    ]


def test_the_metadata_records_the_run(template_run: Run) -> None:
    metadata = template_run.result.report.metadata
    pair = template_run.pair

    assert metadata["frames_before"] == len(pair.before.frames)
    assert metadata["frames_after"] == len(pair.after.frames)
    assert metadata["describer"] == "template"
    assert metadata["device"] == "cpu"
    assert metadata["model_reconstruction"] == "oracle"
    assert metadata["seed"] == 0
    assert "seconds_voting" in metadata
    assert "peak_vram_mb" not in metadata


def test_the_model_describer_sees_crops_and_unloads_last(vlm_run: Run) -> None:
    timeline = vlm_run.timeline
    changes = vlm_run.result.report.changes

    assert timeline.roles()[-1] == "vlm"
    assert timeline.most_loaded == 1
    assert not timeline.loaded
    assert len(vlm_run.vlm.calls) >= len(
        [c for c in changes if c.type is not ChangeType.UNCERTAIN]
    )
    assert any(count > 0 for count, _ in vlm_run.vlm.calls)
    assert vlm_run.vlm.unload_count == 1
    assert vlm_run.result.report.metadata["describer"] == "vlm"
    for change, fact in zip(changes, vlm_run.result.facts, strict=True):
        assert validate_description(change.description, fact).ok


class Raiser:
    """A model whose every use fails."""

    def __getattr__(self, name: str) -> object:
        if name == "unload":
            return lambda: None
        raise RuntimeError("model failed")


def failing_models(
    models: Models, role: str, nth: int, timeline: Timeline
) -> Models:
    """Replaces the ``nth`` model a role builds with one that fails."""
    original = getattr(models, role)
    calls = itertools.count()

    def make() -> object:
        if next(calls) == nth:
            return Tracked(f"{role}#failing", Raiser(), timeline)
        return original()

    return dataclasses.replace(models, **{role: make})


@pytest.mark.parametrize(
    ("role", "nth"),
    [
        ("reconstructor", 0),
        ("encoder", 0),
        ("encoder", 1),
        ("segmenter", 0),
        ("labeler", 0),
        ("vlm", 0),
    ],
)
def test_a_failing_stage_still_unloads_its_model(
    tmp_path: Path, role: str, nth: int
) -> None:
    pair = generate_scene_pair(SEEDS[0], SMALL_SCENE)
    timeline = Timeline()
    models = failing_models(
        synthetic_models(pair, timeline), role, nth, timeline
    )
    (tmp_path / "changes.json").write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="model failed"):
        run_frames(
            video_frames(pair, Epoch.BEFORE),
            video_frames(pair, Epoch.AFTER),
            tmp_path,
            MODEL,
            models,
            overwrite=True,
        )

    assert not timeline.loaded
    assert ("unload", f"{role}#failing") in timeline.events
    assert not (tmp_path / "changes.json").exists()


def test_a_non_empty_out_dir_fails_before_any_model_loads(
    tmp_path: Path,
) -> None:
    (tmp_path / "old.txt").write_text("x", encoding="utf-8")
    pair, _, _ = scene_clouds(SEEDS[0])
    timeline = Timeline()

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_frames(
            video_frames(pair, Epoch.BEFORE),
            video_frames(pair, Epoch.AFTER),
            tmp_path,
            TEMPLATE,
            synthetic_models(pair, timeline),
        )

    assert timeline.events == []


def test_overwrite_writes_into_a_non_empty_out_dir(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("x", encoding="utf-8")
    pair, _, _ = scene_clouds(SEEDS[0])

    result = run_frames(
        video_frames(pair, Epoch.BEFORE),
        video_frames(pair, Epoch.AFTER),
        tmp_path,
        TEMPLATE,
        synthetic_models(pair, Timeline()),
        overwrite=True,
    )

    assert all(p.exists() for p in result.files.all)
    assert (tmp_path / "old.txt").exists()


def test_a_run_needs_frames_in_both_epochs(tmp_path: Path) -> None:
    pair, _, _ = scene_clouds(SEEDS[0])

    with pytest.raises(ValueError, match="keyframe"):
        run_frames(
            [],
            video_frames(pair, Epoch.AFTER),
            tmp_path,
            TEMPLATE,
            synthetic_models(pair, Timeline()),
        )


def test_too_many_frames_are_rejected_before_anything_happens(
    tmp_path: Path,
) -> None:
    (tmp_path / "old.txt").write_text("x", encoding="utf-8")
    pair, _, _ = scene_clouds(SEEDS[0])
    timeline = Timeline()
    models = dataclasses.replace(synthetic_models(pair, timeline), max_frames=4)

    with pytest.raises(ValueError, match="max_frames"):
        run_frames(
            video_frames(pair, Epoch.BEFORE),
            video_frames(pair, Epoch.AFTER),
            tmp_path,
            TEMPLATE,
            models,
        )

    assert timeline.events == []


def test_empty_frames_are_rejected_before_the_out_dir_is_checked(
    tmp_path: Path,
) -> None:
    (tmp_path / "old.txt").write_text("x", encoding="utf-8")
    pair, _, _ = scene_clouds(SEEDS[0])

    with pytest.raises(ValueError, match="keyframe"):
        run_frames(
            video_frames(pair, Epoch.BEFORE),
            [],
            tmp_path,
            TEMPLATE,
            synthetic_models(pair, Timeline()),
        )


def test_without_changes_no_describer_is_built(tmp_path: Path) -> None:
    pair, _, _ = scene_clouds(SEEDS[0])
    same = video_frames(pair, Epoch.BEFORE)
    built: list[int] = []

    def vlm() -> DescribingVlm:
        built.append(1)
        return DescribingVlm()

    result = run_frames(
        same,
        same,
        tmp_path,
        MODEL,
        synthetic_models(pair, Timeline(), vlm),
    )

    assert result.report.changes == []
    assert built == []
