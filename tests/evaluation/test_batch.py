import dataclasses
import json
import pickle
import re
import shutil
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from eval_pairs import (
    CROP_WINDOW,
    CROPPED,
    SIZES,
    own_segments,
    with_after_window,
    with_metadata,
    with_uncertain_move,
    write_pair,
)
from pipeline_scenes import inert_models
from rle_helpers import encode_mask

from spatial_change.alignment import RefineConfig
from spatial_change.domain import ChangeType
from spatial_change.evaluation import (
    SceneEval,
    evaluate,
    gt_items,
    load_pair_annotations,
    parse_pair_annotations,
)
from spatial_change.evaluation import batch as batch_module
from spatial_change.evaluation.batch import (
    BatchSummary,
    PairEvent,
    PairRecord,
    run_batch,
    settings_digest,
    summarize,
)
from spatial_change.evaluation.predictions import (
    load_predictions,
)
from spatial_change.evaluation.projection import ProjectionConfig
from spatial_change.evaluation.scene import (
    PairEvalConfig,
    build_scene,
    project_run,
)
from spatial_change.pipeline import (
    Models,
    PipelineResult,
    RoomScene,
    RunConfig,
    StageEvent,
)
from spatial_change.pipeline.config import StageConfigs
from spatial_change.visualization import RerunConfig

CONFIG = RunConfig(device="cpu", describer="template")
EVAL = PairEvalConfig(stride=1, max_length=32)


MODELS = dataclasses.replace(
    inert_models(reconstruction="oracle", features="fake"),
    settings={"reconstruction": {"max_frames": 128}},
)


@dataclass
class Call:
    pair: str
    before: Path
    after: Path
    out_dir: Path
    config: RunConfig
    models: Models
    overwrite: bool
    cache_dir: Path | None


class Runner:
    """Stands in for ``run_videos``: answers each pair with a prepared outcome.

    A pair whose outcome is an exception emits the start of two stages, the
    end of the first, and then raises.
    """

    def __init__(
        self, outcomes: Mapping[str, PipelineResult | BaseException]
    ) -> None:
        self.outcomes = dict(outcomes)
        self.calls: list[Call] = []

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
        pair = out_dir.parent.name
        self.calls.append(
            Call(
                pair,
                before_path,
                after_path,
                out_dir,
                config,
                models,
                overwrite,
                cache_dir,
            )
        )
        self.before_run(pair, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "changes.json").write_text("{}", encoding="utf-8")
        outcome = self.outcomes[pair]
        if progress is not None:
            progress(StageEvent("keyframes", 1, 12, None))
            progress(StageEvent("keyframes", 1, 12, 0.5))
            progress(StageEvent("reconstruction", 2, 12, None))
        if isinstance(outcome, BaseException):
            raise outcome
        if progress is not None:
            progress(StageEvent("reconstruction", 2, 12, 1.5))
        return outcome

    def before_run(self, pair: str, out_dir: Path) -> None:
        """Hook for tests that look at the disk when a run starts."""


@pytest.fixture(scope="module")
def run(template_run: PipelineResult) -> PipelineResult:
    return with_metadata(template_run, **SIZES, cache_hits="reconstruction")


def make_data(root: Path, names: list[str], result: PipelineResult) -> Path:
    data = root / "data"
    segments = own_segments(result, EVAL)
    for name in names:
        write_pair(data, name, segments)
    return data


def read_record(out: Path, pair: str) -> PairRecord:
    text = (out / "pairs" / pair / "result.json").read_text(encoding="utf-8")
    return PairRecord.model_validate_json(text)


def batch(
    root: Path,
    names: list[str],
    runner: Runner,
    eval_config: PairEvalConfig = EVAL,
    **options: object,
) -> BatchSummary:
    return run_batch(
        names,
        root / "data",
        root / "out",
        CONFIG,
        MODELS,
        eval_config=eval_config,
        runner=runner,
        **options,  # type: ignore[arg-type]
    )


def test_each_pair_gets_a_folder_and_the_batch_a_summary(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)

    batch(tmp_path, ["a", "b"], Runner({"a": run, "b": run}))

    out = tmp_path / "out"
    for pair in ("a", "b"):
        folder = out / "pairs" / pair
        assert sorted(p.name for p in folder.iterdir()) == [
            "pred.npz",
            "result.json",
            "run",
        ]
        assert (folder / "run" / "changes.json").is_file()
    assert (out / "summary.json").is_file()
    assert [p.name for p in out.iterdir() if p.name != "pairs"] == [
        "summary.json"
    ]


def test_the_record_is_the_commit_marker_and_is_written_last(
    tmp_path: Path, run: PipelineResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_data(tmp_path, ["a"], run)
    folder = tmp_path / "out" / "pairs" / "a"
    seen: list[bool] = []
    real = batch_module.save_predictions

    def spy(*args: object, **kwargs: object) -> None:
        seen.append((folder / "result.json").exists())
        real(*args, **kwargs)  # type: ignore[arg-type]
        seen.append((folder / "result.json").exists())

    monkeypatch.setattr(batch_module, "save_predictions", spy)

    batch(tmp_path, ["a"], Runner({"a": run}))

    assert seen == [False, False]
    assert (folder / "result.json").exists()


def test_a_recomputed_pair_loses_its_old_record_before_it_runs(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    batch(tmp_path, ["a"], Runner({"a": run}))
    folder = tmp_path / "out" / "pairs" / "a"
    (folder / "run" / "stale.txt").write_text("old", encoding="utf-8")
    seen: list[tuple[bool, bool, bool]] = []

    class Watching(Runner):
        def before_run(self, pair: str, out_dir: Path) -> None:
            seen.append(
                (
                    (folder / "result.json").exists(),
                    (folder / "pred.npz").exists(),
                    (out_dir / "stale.txt").exists(),
                )
            )

    options = {"eval_config": EVAL, "runner": Watching({"a": run})}
    run_batch(
        ["a"],
        tmp_path / "data",
        tmp_path / "out",
        RunConfig(device="cpu", describer="template", seed=3),
        MODELS,
        **options,  # type: ignore[arg-type]
    )

    assert seen == [(False, False, False)]


def test_an_ok_record_holds_what_the_run_found(
    tmp_path: Path, run: PipelineResult
) -> None:
    result = with_metadata(
        run, floor_observed=False, cache_hits="reconstruction,appearance"
    )
    make_data(tmp_path, ["a"], result)

    batch(tmp_path, ["a"], Runner({"a": result}))

    record = read_record(tmp_path / "out", "a")
    metadata = result.report.metadata
    assert record.pair == "a"
    assert record.status == "ok"
    assert record.settings == settings_digest(CONFIG, MODELS, EVAL)
    assert record.seconds >= 0.0
    assert record.stage_seconds == dict(result.seconds)
    assert record.cache_hits == ("reconstruction", "appearance")
    assert (record.n_gt, record.n_changes, record.n_scored) == (3, 3, 3)
    assert (record.stride, record.max_length) == (EVAL.stride, EVAL.max_length)
    assert record.changes_by_type == {"added": 1, "moved": 1, "removed": 1}
    assert record.floor_observed is False
    assert record.camera_height_m == metadata["camera_height_m"]
    assert record.after_cropped is False
    assert record.frames_before == metadata["frames_before"]
    assert record.frames_after == metadata["frames_after"]
    assert record.gt_matches_video is True
    assert record.metrics is not None
    assert record.metrics.ap_per_scene == 1.0
    assert (record.error_type, record.error, record.failed_stage) == (
        None,
        None,
        None,
    )


def test_changes_seen_in_no_scored_frame_are_counted_but_not_scored(
    tmp_path: Path, run: PipelineResult
) -> None:
    bare = dataclasses.replace(run, scene=RoomScene((), (), run.scene.sources))
    make_data(tmp_path, ["a"], run)

    batch(tmp_path, ["a"], Runner({"a": bare}))

    record = read_record(tmp_path / "out", "a")
    assert (record.n_changes, record.n_scored) == (3, 0)
    assert record.metrics is not None
    assert (record.metrics.tp, record.metrics.fn) == (0, 3)


def test_a_pair_without_annotations_runs_and_every_prediction_is_wrong(
    tmp_path: Path, run: PipelineResult
) -> None:
    result = with_metadata(run, video_before_width=64, video_before_height=48)
    write_pair(
        tmp_path / "data",
        "a",
        {"objects": [], "video1_objects": {}, "video2_objects": {}},
    )

    batch(tmp_path, ["a"], Runner({"a": result}))

    record = read_record(tmp_path / "out", "a")
    assert record.status == "ok"
    assert (record.n_gt, record.n_scored) == (0, 3)
    assert record.gt_matches_video is None
    assert record.metrics is not None
    assert (record.metrics.tp, record.metrics.fp) == (0, 3)
    assert record.metrics.ap_per_scene == 0.0
    stored = load_predictions(tmp_path / "out" / "pairs" / "a" / "pred.npz")
    assert (stored.height, stored.width) == (32, 24)
    assert len(stored.items) == 3


def test_a_ground_truth_of_another_size_is_recorded_and_still_scored(
    tmp_path: Path, run: PipelineResult
) -> None:
    result = with_metadata(run, video_before_width=64, video_before_height=48)
    make_data(tmp_path, ["a"], run)

    batch(tmp_path, ["a"], Runner({"a": result}))

    record = read_record(tmp_path / "out", "a")
    assert record.status == "ok"
    assert record.gt_matches_video is False
    assert record.metrics is not None
    assert record.metrics.ap_per_scene == 1.0


def test_a_failure_keeps_its_stage_and_the_batch_goes_on(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    runner = Runner(
        {"a": RuntimeError("CUDA out of memory\nmore text"), "b": run}
    )

    summary = batch(tmp_path, ["a", "b"], runner)

    failed = read_record(tmp_path / "out", "a")
    assert failed.status == "failed"
    assert failed.failed_stage == "reconstruction"
    assert failed.error_type == "RuntimeError"
    assert failed.error == "CUDA out of memory"
    assert failed.stage_seconds == {"keyframes": 0.5}
    assert failed.metrics is None
    assert (failed.n_changes, failed.n_scored) == (0, 0)
    assert failed.n_gt == 3
    assert not (tmp_path / "out" / "pairs" / "a" / "pred.npz").exists()
    assert read_record(tmp_path / "out", "b").status == "ok"
    assert [c.pair for c in runner.calls] == ["a", "b"]
    assert (summary.ok, summary.failed) == (1, 1)


def test_an_error_without_a_message_is_named_by_its_type(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)

    batch(tmp_path, ["a"], Runner({"a": ValueError()}))

    assert read_record(tmp_path / "out", "a").error == "ValueError"


def test_a_failure_while_scoring_is_recorded_as_such(
    tmp_path: Path, run: PipelineResult
) -> None:
    broken = dataclasses.replace(
        run,
        scene=RoomScene((), (), dict(list(run.scene.sources.items())[:1])),
    )
    make_data(tmp_path, ["a"], run)

    batch(tmp_path, ["a"], Runner({"a": broken}))

    record = read_record(tmp_path / "out", "a")
    assert (record.status, record.failed_stage) == ("failed", "scoring")
    assert record.error_type == "ValueError"


def test_memory_is_collected_once_the_failed_stage_is_out_of_the_traceback(
    tmp_path: Path, run: PipelineResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Heavy:
        pass

    class Hoarding(Runner):
        def __call__(self, *args: object, **kwargs: object) -> PipelineResult:
            heavy = Heavy()
            held.append(weakref.ref(heavy))
            raise RuntimeError("boom")

    make_data(tmp_path, ["a"], run)
    held: list[weakref.ref[Heavy]] = []
    alive: list[bool] = []
    real = batch_module.gc.collect

    def collect() -> int:
        count = real()
        alive.append(held[0]() is not None)
        return count

    monkeypatch.setattr(batch_module.gc, "collect", collect)

    batch(tmp_path, ["a"], Hoarding({}))

    assert alive == [False]


def test_an_interrupt_stops_the_batch_and_keeps_the_finished_pairs(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b", "c"], run)
    runner = Runner({"a": run, "b": KeyboardInterrupt(), "c": run})

    with pytest.raises(KeyboardInterrupt):
        batch(tmp_path, ["a", "b", "c"], runner)

    pairs = tmp_path / "out" / "pairs"
    assert (pairs / "a" / "result.json").exists()
    assert not (pairs / "b" / "result.json").exists()
    assert [c.pair for c in runner.calls] == ["a", "b"]


HOSTILE = b"cbuiltins\nprint\n(S'pwned'\ntR."


@pytest.mark.parametrize(
    ("segments", "error_type"),
    [
        (HOSTILE, "UnpicklingError"),
        ({"objects": []}, "ValueError"),
        (None, "FileNotFoundError"),
    ],
)
def test_annotations_are_read_safely_before_anything_runs(
    tmp_path: Path,
    run: PipelineResult,
    capsys: pytest.CaptureFixture[str],
    segments: object,
    error_type: str,
) -> None:
    path = write_pair(tmp_path / "data", "a", {}) / "segments.pkl"
    if segments is None:
        path.unlink()
    elif isinstance(segments, bytes):
        path.write_bytes(segments)
    else:
        path.write_bytes(pickle.dumps(segments))
    runner = Runner({"a": run})

    batch(tmp_path, ["a"], runner)

    record = read_record(tmp_path / "out", "a")
    assert (record.status, record.failed_stage) == ("failed", "annotations")
    assert record.error_type == error_type
    assert runner.calls == []
    assert "pwned" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("videos", "fragment"),
    [
        (("video1.mp4", "video2.avi"), "video2"),
        (("video1.mp4", "notes.txt"), "video2"),
        (("clip.mp4", "video2.mp4"), "video1"),
    ],
)
def test_a_missing_video_fails_the_pair_before_anything_runs(
    tmp_path: Path,
    run: PipelineResult,
    videos: tuple[str, str],
    fragment: str,
) -> None:
    write_pair(tmp_path / "data", "a", own_segments(run, EVAL), videos)
    runner = Runner({"a": run})

    batch(tmp_path, ["a"], runner)

    record = read_record(tmp_path / "out", "a")
    assert (record.status, record.failed_stage) == ("failed", "input")
    assert fragment in (record.error or "")
    assert runner.calls == []


def test_a_pair_folder_that_is_absent_fails_at_input(
    tmp_path: Path, run: PipelineResult
) -> None:
    (tmp_path / "data").mkdir()
    runner = Runner({"a": run})

    batch(tmp_path, ["a"], runner)

    record = read_record(tmp_path / "out", "a")
    assert (record.status, record.failed_stage) == ("failed", "input")
    assert runner.calls == []


def test_videos_are_found_whatever_the_case_of_the_extension(
    tmp_path: Path, run: PipelineResult
) -> None:
    write_pair(
        tmp_path / "data",
        "a",
        own_segments(run, EVAL),
        ("VIDEO1.MP4", "Video2.Mov"),
    )
    runner = Runner({"a": run})

    batch(tmp_path, ["a"], runner)

    (call,) = runner.calls
    assert call.before == tmp_path / "data" / "a" / "VIDEO1.MP4"
    assert call.after == tmp_path / "data" / "a" / "Video2.Mov"


@pytest.mark.parametrize(
    "name", ["../x", "a/b", "a\\b", "a b", "", "a.b", ".."]
)
def test_a_pair_name_that_could_leave_the_folder_is_rejected(
    tmp_path: Path, run: PipelineResult, name: str
) -> None:
    runner = Runner({"a": run})

    with pytest.raises(ValueError, match="pair"):
        batch(tmp_path, ["a", name], runner)
    with pytest.raises(ValueError, match="pair"):
        summarize(["a", name], tmp_path / "data", tmp_path / "out", EVAL)

    assert runner.calls == []
    assert not (tmp_path / "out").exists()


def test_a_pair_listed_twice_is_rejected(
    tmp_path: Path, run: PipelineResult
) -> None:
    with pytest.raises(ValueError, match="twice"):
        batch(tmp_path, ["a", "a"], Runner({"a": run}))


def test_the_runner_gets_the_cache_the_run_folder_and_overwrite(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    runner = Runner({"a": run})
    cache = tmp_path / "cache"

    batch(tmp_path, ["a"], runner, cache_dir=cache)

    (call,) = runner.calls
    assert call.cache_dir == cache
    assert call.out_dir == tmp_path / "out" / "pairs" / "a" / "run"
    assert call.overwrite is True
    assert call.config == CONFIG.model_copy(update={"frame_stride": 1})
    assert call.models is MODELS
    assert call.before == tmp_path / "data" / "a" / "video1.mp4"
    assert call.after == tmp_path / "data" / "a" / "video2.mov"


@pytest.mark.parametrize("asked", [None, 7])
def test_the_frames_of_the_run_are_the_frames_that_are_scored(
    tmp_path: Path, run: PipelineResult, asked: int | None
) -> None:
    make_data(tmp_path, ["a"], run)
    runner = Runner({"a": run})
    config = RunConfig(device="cpu", describer="template", frame_stride=asked)

    run_batch(
        ["a"],
        tmp_path / "data",
        tmp_path / "out",
        config,
        MODELS,
        eval_config=PairEvalConfig(stride=5, max_length=32),
        runner=runner,
    )

    assert runner.calls[0].config.frame_stride == 5
    assert runner.calls[0].config == config.model_copy(
        update={"frame_stride": 5}
    )


def test_no_cache_dir_means_no_cache(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    runner = Runner({"a": run})

    batch(tmp_path, ["a"], runner)

    assert runner.calls[0].cache_dir is None


def test_progress_reports_every_pair_once_in_order(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    events: list[PairEvent] = []
    first = Runner({"a": RuntimeError("boom"), "b": run})
    batch(tmp_path, ["a", "b"], first, on_pair=events.append)
    again: list[PairEvent] = []

    batch(tmp_path, ["a", "b"], Runner({"a": run}), on_pair=again.append)

    assert [
        (e.position, e.total, e.record.pair, e.resumed) for e in events
    ] == [
        (1, 2, "a", False),
        (2, 2, "b", False),
    ]
    assert [e.record.status for e in events] == ["failed", "ok"]
    assert [(e.record.pair, e.record.status, e.resumed) for e in again] == [
        ("a", "ok", False),
        ("b", "ok", True),
    ]


def test_a_second_run_reuses_every_finished_pair(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    first = batch(tmp_path, ["a", "b"], Runner({"a": run, "b": run}))
    summary_file = tmp_path / "out" / "summary.json"
    text = summary_file.read_text(encoding="utf-8")
    runner = Runner({})

    second = batch(tmp_path, ["a", "b"], runner)

    assert runner.calls == []
    assert second.model_dump() == first.model_dump()
    assert summary_file.read_text(encoding="utf-8") == text


def test_a_failed_pair_is_tried_again(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    batch(tmp_path, ["a", "b"], Runner({"a": RuntimeError("boom"), "b": run}))
    retry = Runner({"a": run})
    events: list[PairEvent] = []

    summary = batch(tmp_path, ["a", "b"], retry, on_pair=events.append)

    assert [c.pair for c in retry.calls] == ["a"]
    assert (summary.ok, summary.failed) == (2, 0)
    assert [e.resumed for e in events] == [False, True]
    assert read_record(tmp_path / "out", "a").status == "ok"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "truncated"])
def test_an_ok_record_with_unusable_predictions_is_recomputed(
    tmp_path: Path, run: PipelineResult, damage: str
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    batch(tmp_path, ["a", "b"], Runner({"a": run, "b": run}))
    path = tmp_path / "out" / "pairs" / "a" / "pred.npz"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"garbage")
    else:
        path.write_bytes(path.read_bytes()[:40])
    runner = Runner({"a": run})

    batch(tmp_path, ["a", "b"], runner)

    assert [c.pair for c in runner.calls] == ["a"]
    load_predictions(path)


@pytest.mark.parametrize("damage", ["deleted", "garbage", "wrong shape"])
def test_a_pair_without_a_readable_record_is_recomputed(
    tmp_path: Path, run: PipelineResult, damage: str
) -> None:
    make_data(tmp_path, ["a"], run)
    batch(tmp_path, ["a"], Runner({"a": run}))
    path = tmp_path / "out" / "pairs" / "a" / "result.json"
    if damage == "deleted":
        path.unlink()
    elif damage == "garbage":
        path.write_text("{not json", encoding="utf-8")
    else:
        path.write_text('{"pair": "a"}', encoding="utf-8")
    runner = Runner({"a": run})

    batch(tmp_path, ["a"], runner)

    assert [c.pair for c in runner.calls] == ["a"]
    assert read_record(tmp_path / "out", "a").status == "ok"


def test_records_of_other_settings_are_never_reused(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    batch(tmp_path, ["a"], Runner({"a": run}))
    runner = Runner({"a": run})

    run_batch(
        ["a"],
        tmp_path / "data",
        tmp_path / "out",
        RunConfig(device="cpu", describer="template", seed=4),
        MODELS,
        eval_config=EVAL,
        runner=runner,
    )

    assert [c.pair for c in runner.calls] == ["a"]
    assert read_record(tmp_path / "out", "a").settings == settings_digest(
        RunConfig(device="cpu", describer="template", seed=4), MODELS, EVAL
    )


def test_where_things_live_does_not_change_the_settings(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    batch(tmp_path, ["a"], Runner({"a": run}), cache_dir=tmp_path / "one")
    shutil.copytree(tmp_path / "data", tmp_path / "moved")
    runner = Runner({})

    run_batch(
        ["a"],
        tmp_path / "moved",
        tmp_path / "out",
        CONFIG,
        MODELS,
        eval_config=EVAL,
        runner=runner,
        cache_dir=tmp_path / "two",
    )

    assert runner.calls == []


def test_the_settings_digest_is_stable_and_follows_what_changes_results() -> (
    None
):
    base = settings_digest(CONFIG, MODELS, EVAL)
    other_ids = dataclasses.replace(
        MODELS, ids={"reconstruction": "other", "features": "fake"}
    )
    other_settings = dataclasses.replace(
        MODELS, settings={"reconstruction": {"max_frames": 64}}
    )
    variants = {
        "seed": settings_digest(
            RunConfig(device="cpu", describer="template", seed=1), MODELS, EVAL
        ),
        "keyframes": settings_digest(
            RunConfig(device="cpu", describer="template", max_keyframes=20),
            MODELS,
            EVAL,
        ),
        "device": settings_digest(
            RunConfig(device="cuda", describer="template"), MODELS, EVAL
        ),
        "splat radius": settings_digest(
            CONFIG,
            MODELS,
            dataclasses.replace(
                EVAL, projection=ProjectionConfig(splat_radius=5)
            ),
        ),
        "eval stride": settings_digest(
            CONFIG, MODELS, dataclasses.replace(EVAL, stride=30)
        ),
        "model id": settings_digest(CONFIG, other_ids, EVAL),
        "model settings": settings_digest(CONFIG, other_settings, EVAL),
    }

    assert settings_digest(CONFIG, MODELS, EVAL) == base
    assert len(base) == 64
    assert len({base, *variants.values()}) == len(variants) + 1


def test_the_digest_ignores_what_scoring_alone_decides() -> None:
    base = settings_digest(CONFIG, MODELS, EVAL)

    assert base == settings_digest(
        CONFIG, MODELS, dataclasses.replace(EVAL, uncertain_as="moved")
    )
    assert base == settings_digest(
        RunConfig(device="cpu", describer="template", frame_stride=30),
        MODELS,
        EVAL,
    )


def more_refine_rounds(stages: StageConfigs, config: RunConfig) -> object:
    return dataclasses.replace(
        stages, refine=dataclasses.replace(RefineConfig(), max_rounds=9)
    )


def another_palette(stages: StageConfigs, config: RunConfig) -> object:
    colors = dict(RerunConfig().change_colors)
    colors[ChangeType.ADDED] = (1, 2, 3)
    return dataclasses.replace(
        stages, rerun=RerunConfig(seed=config.seed, change_colors=colors)
    )


@pytest.mark.parametrize("tweak", [more_refine_rounds, another_palette])
def test_a_stage_default_is_part_of_the_settings(
    monkeypatch: pytest.MonkeyPatch,
    tweak: Callable[[StageConfigs, RunConfig], object],
) -> None:
    base = settings_digest(CONFIG, MODELS, EVAL)
    real = batch_module.build_stage_configs

    def tweaked(config: RunConfig, device: str) -> object:
        return tweak(real(config, device), config)

    monkeypatch.setattr(batch_module, "build_stage_configs", tweaked)

    assert settings_digest(CONFIG, MODELS, EVAL) != base


@pytest.fixture
def mixed(tmp_path: Path, run: PipelineResult) -> tuple[Path, list[str]]:
    """Five pairs: plain, floor not seen, cropped, odd size, and failed."""
    names = ["plain", "nofloor", "cropped", "oddsize", "broken"]
    cropped = with_after_window(with_metadata(run, **CROPPED), CROP_WINDOW)
    results: dict[str, PipelineResult | BaseException] = {
        "plain": run,
        "nofloor": with_metadata(run, floor_observed=False),
        "cropped": cropped,
        "oddsize": with_metadata(
            run, video_before_width=64, video_before_height=48
        ),
        "broken": RuntimeError("no floor found"),
    }
    data = tmp_path / "data"
    for name in names:
        source = cropped if name == "cropped" else run
        write_pair(data, name, own_segments(source, EVAL))
    runner = Runner(results)
    batch(tmp_path, names, runner)
    return tmp_path, names


def test_the_summary_counts_and_lists_what_needs_a_look(
    mixed: tuple[Path, list[str]],
) -> None:
    root, names = mixed

    summary = summarize(names, root / "data", root / "out", EVAL)

    assert (summary.total, summary.ok, summary.failed, summary.pending) == (
        5,
        4,
        1,
        0,
    )
    assert summary.failure_rate == pytest.approx(0.2)
    assert [
        (r.pair, r.failed_stage, r.error_type, r.error)
        for r in summary.pairs
        if r.status == "failed"
    ] == [("broken", "reconstruction", "RuntimeError", "no floor found")]
    assert summary.floor_unobserved == ("nofloor",)
    assert summary.after_cropped == ("cropped",)
    assert summary.gt_mismatch == ("oddsize",)
    assert summary.changes_by_type == {"added": 4, "moved": 4, "removed": 4}
    assert [r.pair for r in summary.pairs] == names


def test_pooled_metrics_score_a_failed_pair_as_missed_and_also_without_it(
    tmp_path: Path, run: PipelineResult
) -> None:
    good = own_segments(run, EVAL)
    lost = np.zeros((32, 32), dtype=bool)
    lost[:4, :4] = True
    bad = {
        "objects": [
            {"original_obj_idx": 7, "in_video1": True, "in_video2": False}
        ],
        "video1_objects": {7: {"0": encode_mask(lost)}},
        "video2_objects": {},
    }
    write_pair(tmp_path / "data", "good", good)
    write_pair(tmp_path / "data", "bad", bad)

    summary = batch(
        tmp_path,
        ["good", "bad"],
        Runner({"good": run, "bad": RuntimeError("boom")}),
    )

    annotations = parse_pair_annotations("good", good)
    scene = build_scene(
        "good", annotations, project_run(run, annotations, EVAL), EVAL
    )
    missed = SceneEval(
        "bad",
        32,
        32,
        gt_items(parse_pair_annotations("bad", bad), 32, 32, 1),
        [],
    )
    assert summary.metrics_ok_only == evaluate([scene])
    assert summary.metrics == evaluate([scene, missed])
    assert summary.metrics_ok_only is not None and summary.metrics is not None
    assert (summary.metrics_ok_only.tp, summary.metrics_ok_only.fn) == (3, 0)
    assert (summary.metrics.tp, summary.metrics.fn) == (3, 1)
    assert summary.metrics.ap_per_scene == pytest.approx(0.75)
    assert summary.failure_rate == pytest.approx(0.5)
    assert summary.metrics.pixel_fn == summary.metrics_ok_only.pixel_fn + 16


MOVED_EVAL = dataclasses.replace(EVAL, uncertain_as="moved")


@pytest.mark.parametrize(
    ("setting", "config", "n_scored", "tp", "fn"),
    [("drop", EVAL, 2, 2, 1), ("moved", MOVED_EVAL, 3, 3, 0)],
)
def test_a_pair_records_what_became_of_its_uncertain_change(
    tmp_path: Path,
    run: PipelineResult,
    setting: str,
    config: PairEvalConfig,
    n_scored: int,
    tp: int,
    fn: int,
) -> None:
    make_data(tmp_path, ["a"], run)

    summary = batch(
        tmp_path, ["a"], Runner({"a": with_uncertain_move(run)}), config
    )

    record = read_record(tmp_path / "out", "a")
    assert record.uncertain_as == summary.uncertain_as == setting
    assert (record.n_changes, record.n_uncertain) == (3, 1)
    assert record.changes_by_type == {"added": 1, "removed": 1, "uncertain": 1}
    assert record.n_scored == n_scored
    assert record.metrics is not None
    assert (record.metrics.tp, record.metrics.fn) == (tp, fn)
    assert summary.metrics == record.metrics
    assert '"uncertain_as": "' + setting + '"' in (
        tmp_path / "out/summary.json"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("config", [EVAL, MOVED_EVAL], ids=["drop", "moved"])
def test_every_uncertain_object_is_stored_whatever_the_setting(
    tmp_path: Path, run: PipelineResult, config: PairEvalConfig
) -> None:
    make_data(tmp_path, ["a"], run)
    uncertain = with_uncertain_move(run)
    index = next(
        i
        for i, c in enumerate(uncertain.report.changes)
        if c.type is ChangeType.UNCERTAIN
    )

    batch(tmp_path, ["a"], Runner({"a": uncertain}), config)

    stored = load_predictions(tmp_path / "out/pairs/a/pred.npz")
    assert stored.uncertain == {index}
    assert len(stored.items) == 3


def test_a_failed_pair_records_the_uncertain_setting_it_ran_with(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)

    batch(tmp_path, ["a"], Runner({"a": RuntimeError("boom")}), MOVED_EVAL)

    record = read_record(tmp_path / "out", "a")
    assert (record.status, record.uncertain_as, record.n_uncertain) == (
        "failed",
        "moved",
        0,
    )


def test_the_uncertain_setting_never_reruns_a_pair(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    runner = Runner({"a": with_uncertain_move(run)})
    events: list[PairEvent] = []

    dropped = batch(tmp_path, ["a"], runner)
    moved = batch(tmp_path, ["a"], runner, MOVED_EVAL, on_pair=events.append)
    back = batch(tmp_path, ["a"], runner)

    assert len(runner.calls) == 1
    assert [e.resumed for e in events] == [True]
    assert [r.uncertain_as for r in (*dropped.pairs, *moved.pairs)] == [
        "drop",
        "moved",
    ]
    assert (moved.uncertain_as, back.uncertain_as) == ("moved", "drop")


def test_a_summary_pools_the_stored_pairs_with_the_setting_it_is_given(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    dropped = batch(tmp_path, ["a"], Runner({"a": with_uncertain_move(run)}))
    files = tmp_path / "data", tmp_path / "out"

    moved = summarize(["a"], *files, MOVED_EVAL)
    again = summarize(["a"], *files, EVAL)

    assert dropped.metrics is not None and moved.metrics is not None
    assert (dropped.metrics.tp, dropped.metrics.fn) == (2, 1)
    assert (moved.metrics.tp, moved.metrics.fn) == (3, 0)
    assert moved.uncertain_as == "moved"
    assert again.model_dump() == dropped.model_dump()
    assert '"uncertain_as": "drop"' in (
        tmp_path / "out/summary.json"
    ).read_text(encoding="utf-8")


def test_every_pair_of_a_summary_is_scored_with_the_setting_of_the_summary(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    runner = Runner({"a": with_uncertain_move(run), "b": run})
    batch(tmp_path, ["a", "b"], runner)
    out = tmp_path / "out"
    stored = {p: (out / "pairs" / p / "result.json").read_bytes() for p in "ab"}
    scenes = [
        build_scene(
            name,
            load_pair_annotations(name, tmp_path / "data" / name),
            load_predictions(out / "pairs" / name / "pred.npz"),
            MOVED_EVAL,
        )
        for name in ("a", "b")
    ]

    moved = summarize(["a", "b"], tmp_path / "data", out, MOVED_EVAL)

    assert [
        (r.pair, r.uncertain_as, r.n_uncertain, r.n_scored) for r in moved.pairs
    ] == [
        ("a", "moved", 1, 3),
        ("b", "moved", 0, 3),
    ]
    assert [r.metrics for r in moved.pairs] == [evaluate([s]) for s in scenes]
    assert moved.metrics == evaluate(scenes)
    entries = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {e["uncertain_as"] for e in entries["pairs"]} == {"moved"}
    assert entries["uncertain_as"] == "moved"
    assert stored == {
        p: (out / "pairs" / p / "result.json").read_bytes() for p in "ab"
    }
    assert read_record(out, "a").uncertain_as == "drop"


def test_a_pair_run_as_moved_is_summarized_as_dropped(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    moved = batch(
        tmp_path, ["a"], Runner({"a": with_uncertain_move(run)}), MOVED_EVAL
    )
    out = tmp_path / "out"
    stored = (out / "pairs/a/result.json").read_bytes()

    dropped = summarize(["a"], tmp_path / "data", out, EVAL)

    (record,) = dropped.pairs
    assert record.uncertain_as == dropped.uncertain_as == "drop"
    assert record.n_scored == 2
    assert record.metrics is not None
    assert (record.metrics.tp, record.metrics.fn) == (2, 1)
    assert moved.metrics is not None
    assert (moved.metrics.tp, moved.metrics.fn) == (3, 0)
    assert dropped.metrics == record.metrics
    assert (out / "pairs/a/result.json").read_bytes() == stored
    assert read_record(out, "a").uncertain_as == "moved"


def test_a_failed_pair_is_listed_with_the_setting_of_the_summary(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    batch(tmp_path, ["a"], Runner({"a": RuntimeError("boom")}))

    summary = summarize(["a"], tmp_path / "data", tmp_path / "out", MOVED_EVAL)

    (record,) = summary.pairs
    assert (record.status, record.uncertain_as) == ("failed", "moved")
    assert (record.metrics, record.n_scored) == (None, 0)
    assert (record.failed_stage, record.error) == ("reconstruction", "boom")
    assert summary.uncertain_as == "moved"
    assert summary.metrics is not None
    assert summary.metrics.fn == 3
    assert read_record(tmp_path / "out", "a").uncertain_as == "drop"


def test_the_ok_only_metrics_follow_the_setting_of_the_summary(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    batch(
        tmp_path,
        ["a", "b"],
        Runner({"a": with_uncertain_move(run), "b": RuntimeError("boom")}),
    )
    files = tmp_path / "data", tmp_path / "out"

    dropped = summarize(["a", "b"], *files, EVAL)
    moved = summarize(["a", "b"], *files, MOVED_EVAL)

    assert dropped.metrics_ok_only is not None
    assert moved.metrics_ok_only is not None
    assert (dropped.metrics_ok_only.tp, dropped.metrics_ok_only.fn) == (2, 1)
    assert (moved.metrics_ok_only.tp, moved.metrics_ok_only.fn) == (3, 0)
    assert dropped.metrics is not None and moved.metrics is not None
    assert (dropped.metrics.tp, dropped.metrics.fn) == (2, 4)
    assert (moved.metrics.tp, moved.metrics.fn) == (3, 3)


def test_pairs_scored_on_other_frames_are_not_pooled(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    batch(tmp_path, ["a"], Runner({"a": run}))
    kept = (tmp_path / "out/summary.json").read_bytes()
    files = tmp_path / "data", tmp_path / "out"

    for other, asked in (
        (dataclasses.replace(EVAL, stride=2), "stride 2, max length 32"),
        (dataclasses.replace(EVAL, max_length=64), "stride 1, max length 64"),
    ):
        phrase = f"made with stride 1, max length 32, not {asked}"
        with pytest.raises(ValueError, match=re.escape(phrase)):
            summarize(["a"], *files, other)

    assert (tmp_path / "out/summary.json").read_bytes() == kept


def test_pairs_scored_on_other_frames_are_pending_beside_current_ones(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    batch(tmp_path, ["a", "b"], Runner({"a": run, "b": run}))
    batch(
        tmp_path,
        ["b"],
        Runner({"b": run}),
        PairEvalConfig(stride=2, max_length=32),
    )

    summary = summarize(
        ["a", "b"],
        tmp_path / "data",
        tmp_path / "out",
        PairEvalConfig(stride=2, max_length=32),
    )

    assert (summary.ok, summary.pending) == (1, 1)
    assert [r.pair for r in summary.pairs] == ["b"]


def test_a_batch_with_no_ok_pair_has_no_ok_only_metrics(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)

    summary = batch(tmp_path, ["a"], Runner({"a": RuntimeError("boom")}))

    assert summary.metrics_ok_only is None
    assert summary.metrics is not None
    assert summary.metrics.fn == 3


def test_a_failed_pair_with_broken_annotations_does_not_stop_the_summary(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    write_pair(tmp_path / "data", "b", {"objects": []})

    summary = batch(tmp_path, ["a", "b"], Runner({"a": run, "b": run}))

    assert (summary.ok, summary.failed) == (1, 1)
    assert summary.metrics_ok_only == summary.metrics


def test_pairs_not_run_yet_are_pending(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a", "b"], run)
    batch(tmp_path, ["a"], Runner({"a": run}))

    summary = summarize(["a", "b"], tmp_path / "data", tmp_path / "out", EVAL)

    assert (summary.total, summary.ok, summary.failed, summary.pending) == (
        2,
        1,
        0,
        1,
    )
    assert summary.metrics_ok_only == summary.metrics


def test_summarize_rebuilds_the_same_file_from_the_disk(
    mixed: tuple[Path, list[str]],
) -> None:
    root, names = mixed
    path = root / "out" / "summary.json"
    text = path.read_text(encoding="utf-8")
    path.unlink()

    rebuilt = summarize(names, root / "data", root / "out", EVAL)

    assert path.read_text(encoding="utf-8") == text
    assert BatchSummary.model_validate_json(text) == rebuilt
    assert json.loads(text)["total"] == 5


def test_summarize_needs_no_models_and_changes_no_record(
    mixed: tuple[Path, list[str]],
) -> None:
    root, names = mixed
    records = {
        p.name: (p / "result.json").read_bytes()
        for p in (root / "out" / "pairs").iterdir()
    }

    summarize(names, root / "data", root / "out", EVAL)

    assert records == {
        p.name: (p / "result.json").read_bytes()
        for p in (root / "out" / "pairs").iterdir()
    }


def test_predictions_on_disk_are_what_was_projected(
    tmp_path: Path, run: PipelineResult
) -> None:
    make_data(tmp_path, ["a"], run)
    batch(tmp_path, ["a"], Runner({"a": run}))
    stored = load_predictions(tmp_path / "out" / "pairs" / "a" / "pred.npz")
    annotations = parse_pair_annotations("a", own_segments(run, EVAL))
    projected = project_run(run, annotations, EVAL)

    assert (stored.height, stored.width) == (projected.height, projected.width)
    assert [i.obj_id for i in stored.items] == [
        i.obj_id for i in projected.items
    ]
    assert stored.uncertain == projected.uncertain
