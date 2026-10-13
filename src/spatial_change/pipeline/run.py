"""The whole system: two videos, or their keyframes, to the four outputs.

``run_videos`` decodes the keyframes and hands them to ``run_frames``,
which does everything after decoding. Each model is built when its stage
starts and unloaded when the stage ends, so at most one heavy model is on
the GPU at a time.

Importing this module does not load torch, transformers, open3d, or cv2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from spatial_change import __version__
from spatial_change.cache import StageCache
from spatial_change.change import (
    ChangeSources,
    CrossAppearance,
    LiftedObjects,
    associate_with_sources,
    compare_appearance,
    describe_instances,
    find_candidates,
    find_unobserved_regions,
    fuse_appearance,
    lift_change_candidates,
)
from spatial_change.diagnostics.profiling import CudaMemoryProbe
from spatial_change.domain import (
    ChangeReport,
    EpochReconstruction,
    Frame,
    Sim3,
)
from spatial_change.pipeline.alignment import AlignedEpochs, align_epochs
from spatial_change.pipeline.config import (
    RunConfig,
    StageConfigs,
    build_stage_configs,
    resolve_device,
)
from spatial_change.pipeline.crops import crops_for_changes
from spatial_change.pipeline.models import Models, loaded
from spatial_change.pipeline.outputs import (
    OutputFiles,
    prepare_out_dir,
    write_outputs,
)
from spatial_change.pipeline.progress import ProgressCallback, StageClock
from spatial_change.pipeline.stage_cache import (
    APPEARANCE,
    RECONSTRUCTION,
    appearance_cached,
    reconstruct_cached,
)
from spatial_change.pipeline.videos import (
    WHOLE_IMAGE,
    ImageWindow,
    load_video_frames,
)
from spatial_change.ports import Describer
from spatial_change.semantics import (
    ChangeFact,
    LocalVlmDescriber,
    TemplateDescriber,
    build_change_fact,
    extract_landmarks,
    label_instances,
)
from spatial_change.visibility import CrossEvidence, vote_epochs

if TYPE_CHECKING:
    from spatial_change.capture import VideoFrame
    from spatial_change.reconstruction.joint import JointReconstruction

MetadataValue = str | int | float | bool


@dataclass(frozen=True, eq=False)
class RoomScene:
    """The posed frames and the objects behind the changes of a run.

    An evaluation projects the objects into the frames to score them in 2D.

    Attributes:
        before_frames: The BEFORE frames in the room frame, with their depth
            and confidence.
        after_frames: The AFTER frames in the room frame, registered onto
            the BEFORE epoch.
        sources: The objects each change was made from, by change id.
        after_window: The part of the AFTER image its frames show, which is
            all of it unless the frames were cropped to the BEFORE aspect.
    """

    before_frames: tuple[Frame, ...]
    after_frames: tuple[Frame, ...]
    sources: Mapping[str, ChangeSources]
    after_window: ImageWindow = WHOLE_IMAGE


@dataclass(frozen=True, eq=False)
class PipelineResult:
    """What a run produced.

    Attributes:
        report: The detected changes and unobserved regions, as written to
            ``changes.json``.
        facts: The structured facts behind each change description, in the
            order of ``report.changes``.
        files: The files written.
        seconds: Wall time of every stage that ran.
        room_from_world: The gravity alignment from the world frame of the
            joint reconstruction to the room frame of the report.
        scene: The frames and objects the changes were found in.
    """

    report: ChangeReport
    facts: tuple[ChangeFact, ...]
    files: OutputFiles
    seconds: dict[str, float]
    room_from_world: Sim3
    scene: RoomScene


def run_videos(
    before_path: Path,
    after_path: Path,
    out_dir: Path,
    config: RunConfig,
    models: Models,
    progress: ProgressCallback | None = None,
    *,
    overwrite: bool = False,
    cache_dir: Path | None = None,
) -> PipelineResult:
    """Runs the whole system on two videos of the same area.

    Args:
        before_path: The BEFORE video.
        after_path: The AFTER video.
        out_dir: Output folder, created if missing.
        config: The knobs of the run.
        models: Factories of the models.
        progress: Receives the start and the end of every stage.
        overwrite: Whether a folder that already holds files is accepted.
        cache_dir: Folder that keeps the reconstruction and the appearance
            comparison between runs, or ``None`` to compute both every
            time. Where the cache lives is not part of any key.

    Returns:
        The report, its facts, the files, and the stage times.

    Raises:
        FileExistsError: If ``out_dir`` holds files and ``overwrite`` is
            false.
        FileNotFoundError: If a video does not exist or is not a file. Both
            are checked before the folder is touched.
        ValueError: If a video cannot be decoded or a stage rejects its
            input.
    """
    _check_videos(before_path, after_path)
    prepare_out_dir(out_dir, overwrite)
    stages = build_stage_configs(config, resolve_device(config))
    clock = StageClock(progress)
    with clock.stage("keyframes"):
        videos = load_video_frames(
            before_path, after_path, stages, models.max_frames
        )
    extra: dict[str, MetadataValue] = {
        "video_before_width": videos.before_info.width,
        "video_before_height": videos.before_info.height,
        "video_before_fps": videos.before_info.fps,
        "video_before_frames": videos.before_info.frame_count,
        "video_after_width": videos.after_info.width,
        "video_after_height": videos.after_info.height,
        "video_after_fps": videos.after_info.fps,
        "video_after_frames": videos.after_info.frame_count,
        "after_cropped_to_before_aspect": videos.after_cropped,
    }
    if videos.strided_before is not None and videos.strided_after is not None:
        extra["strided_frames_before"] = videos.strided_before
        extra["strided_frames_after"] = videos.strided_after
    if videos.frame_durations is not None:
        extra["frame_durations"] = videos.frame_durations
    return _run(
        videos.before,
        videos.after,
        out_dir,
        config,
        stages,
        models,
        clock,
        extra,
        cache_dir,
        videos.after_window,
    )


def run_frames(
    before_frames: Sequence[VideoFrame],
    after_frames: Sequence[VideoFrame],
    out_dir: Path,
    config: RunConfig,
    models: Models,
    progress: ProgressCallback | None = None,
    *,
    overwrite: bool = False,
    cache_dir: Path | None = None,
) -> PipelineResult:
    """Runs everything after decoding on the keyframes of two videos.

    Args:
        before_frames: BEFORE keyframes, all of one size.
        after_frames: AFTER keyframes of the same size; together with the
            BEFORE ones at most as many as the reconstruction accepts.
        out_dir: Output folder, created if missing.
        config: The knobs of the run.
        models: Factories of the models.
        progress: Receives the start and the end of every stage.
        overwrite: Whether a folder that already holds files is accepted.
        cache_dir: Folder that keeps the reconstruction and the appearance
            comparison between runs, or ``None`` to compute both every
            time. Where the cache lives is not part of any key.

    Returns:
        The report, its facts, the files, and the stage times.

    Raises:
        FileExistsError: If ``out_dir`` holds files and ``overwrite`` is
            false.
        ValueError: If a capture has no keyframes or a stage rejects its
            input.
    """
    _check_frames(before_frames, after_frames, models)
    prepare_out_dir(out_dir, overwrite)
    stages = build_stage_configs(config, resolve_device(config))
    return _run(
        before_frames,
        after_frames,
        out_dir,
        config,
        stages,
        models,
        StageClock(progress),
        {},
        cache_dir,
        WHOLE_IMAGE,
    )


def _describer_kind(requested: str, device: str) -> str:
    """Resolves ``auto`` to ``vlm`` on a CUDA device, else ``template``."""
    if requested == "auto":
        return "vlm" if device.startswith("cuda") else "template"
    return requested


def _make_describer(kind: str, models: Models) -> Describer:
    """Builds the describer of a resolved kind (``vlm`` or ``template``)."""
    if kind == "template":
        return TemplateDescriber()
    return LocalVlmDescriber(models.vlm(), TemplateDescriber())


def _check_videos(before_path: Path, after_path: Path) -> None:
    """Rejects a video that is not a file, before anything is touched.

    Raises:
        FileNotFoundError: If a video does not exist or is not a file.
    """
    for path in (before_path, after_path):
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"not a video file: {path}")


def _check_frames(
    before: Sequence[VideoFrame],
    after: Sequence[VideoFrame],
    models: Models,
) -> None:
    """Rejects input a run cannot use, before anything is touched.

    Raises:
        ValueError: If a capture has no keyframes or both together exceed
            what the reconstruction accepts.
    """
    if not before or not after:
        raise ValueError("each epoch needs at least one keyframe")
    if len(before) + len(after) > models.max_frames:
        raise ValueError(
            f"{len(before)} + {len(after)} keyframes exceed the "
            f"reconstruction's max_frames of {models.max_frames}"
        )


def _metadata(
    config: RunConfig,
    stages: StageConfigs,
    device: str,
    describer: str,
    models: Models,
    before: EpochReconstruction,
    after: EpochReconstruction,
    aligned: AlignedEpochs,
    clock: StageClock,
    probe: CudaMemoryProbe | None,
    cache_hits: Sequence[str],
    extra: Mapping[str, MetadataValue],
) -> dict[str, MetadataValue]:
    """Gathers the flat run information of the report."""
    first = before.frames[0].intrinsics
    heights = [
        frame.world_from_camera.translation[2]
        for frame in (*aligned.before.frames, *aligned.after.frames)
    ]
    meta: dict[str, MetadataValue] = {
        "version": __version__,
        "device": device,
        "seed": config.seed,
        "sample_fps": stages.sample_fps,
        "describer": describer,
        "frames_before": len(before.frames),
        "frames_after": len(after.frames),
        "frame_width": first.width,
        "frame_height": first.height,
        "points_before": len(before.points),
        "points_after": len(after.points),
        "scale_to_meters": aligned.room_from_world.scale,
        "floor_observed": aligned.floor_observed,
        "camera_height_m": float(median(heights)),
        "refine_rounds": aligned.refinement.rounds,
        "refine_rmse": aligned.refinement.icp.rmse,
        "cache_hits": ",".join(cache_hits),
    }
    meta.update({f"model_{role}": name for role, name in models.ids.items()})
    meta.update({f"seconds_{n}": s for n, s in clock.timings.items()})
    if probe is not None:
        meta["peak_vram_mb"] = probe.peak_mb()
    meta.update(extra)
    return meta


def _run(
    before_frames: Sequence[VideoFrame],
    after_frames: Sequence[VideoFrame],
    out_dir: Path,
    config: RunConfig,
    stages: StageConfigs,
    models: Models,
    clock: StageClock,
    extra: Mapping[str, MetadataValue],
    cache_dir: Path | None,
    after_window: ImageWindow,
) -> PipelineResult:
    """Runs every stage after decoding."""
    from spatial_change.reconstruction import reconstruct_joint

    device = stages.visibility.device
    probe = CudaMemoryProbe(device) if device.startswith("cuda") else None
    if probe is not None:
        probe.reset()

    cache = StageCache(cache_dir) if cache_dir is not None else None
    cache_hits: list[str] = []

    def reconstruct() -> JointReconstruction:
        with loaded(models.reconstructor) as backbone:
            return reconstruct_joint(backbone, before_frames, after_frames)

    with clock.stage(RECONSTRUCTION):
        reconstruction = reconstruct_cached(
            cache, models, before_frames, after_frames, reconstruct
        )
    joint = reconstruction.value
    if reconstruction.hit:
        cache_hits.append(RECONSTRUCTION)

    with clock.stage("alignment"):
        aligned = align_epochs(joint.before, joint.after, stages)
        before, after = aligned.before, aligned.after

    with clock.stage("voting"):
        evidence: CrossEvidence = vote_epochs(before, after, stages.visibility)

    def compare() -> tuple[CrossAppearance, str | None]:
        with loaded(models.encoder) as encoder:
            result = compare_appearance(
                before, after, encoder, stages.appearance
            )
            return result, encoder.loaded_model_id

    with clock.stage(APPEARANCE):
        appearance = appearance_cached(
            cache, models, stages, before, after, compare
        )
        if appearance.hit:
            cache_hits.append(APPEARANCE)
        evidence = fuse_appearance(
            evidence, appearance.value, stages.appearance
        )

    with clock.stage("candidates"):
        candidates = find_candidates(before, after, evidence, stages.candidates)
        regions = find_unobserved_regions(
            before, after, evidence, stages.unobserved
        )

    with clock.stage("lifting"), loaded(models.segmenter) as segmenter:
        lifted = lift_change_candidates(
            before, after, candidates, segmenter, stages.lift
        )

    with clock.stage("descriptors"), loaded(models.encoder) as encoder:
        described = LiftedObjects(
            disappeared=describe_instances(
                before, lifted.disappeared, encoder, stages.descriptors
            ),
            appeared=describe_instances(
                after, lifted.appeared, encoder, stages.descriptors
            ),
        )

    with clock.stage("labeling"), loaded(models.labeler) as labeler:
        labeled = LiftedObjects(
            disappeared=label_instances(
                before, described.disappeared, labeler, config=stages.label
            ),
            appeared=label_instances(
                after, described.appeared, labeler, config=stages.label
            ),
        )
        landmarks = extract_landmarks(
            before, evidence, labeler, stages.landmarks
        )

    with clock.stage("matching"):
        matched = associate_with_sources(labeled, candidates, stages.match)
        changes = tuple(change for change, _ in matched)

    with clock.stage("describing"):
        views = crops_for_changes(changes, before, after)
        viewpoint = before.frames[0].world_from_camera
        kind = _describer_kind(config.describer, device)
        facts = []
        written = []
        if changes:
            with loaded(lambda: _make_describer(kind, models)) as describer:
                for change in changes:
                    fact = build_change_fact(
                        change, landmarks, viewpoint, stages.relation
                    )
                    text = describer.describe(
                        fact, views[change.change_id].describer
                    )
                    facts.append(fact)
                    written.append(
                        change.model_copy(update={"description": text})
                    )

    with clock.stage("outputs"):
        report = ChangeReport(
            changes=written,
            unobserved_regions=list(regions),
            metadata=_metadata(
                config,
                stages,
                device,
                kind,
                models,
                before,
                after,
                aligned,
                clock,
                probe,
                cache_hits,
                extra,
            ),
        )
        files = write_outputs(
            out_dir,
            report,
            before,
            after,
            evidence,
            {cid: view.report for cid, view in views.items()},
            stages.rerun,
        )
    return PipelineResult(
        report=report,
        facts=tuple(facts),
        files=files,
        seconds=dict(clock.timings),
        room_from_world=aligned.room_from_world,
        scene=RoomScene(
            before_frames=tuple(before.frames),
            after_frames=tuple(after.frames),
            sources={change.change_id: sources for change, sources in matched},
            after_window=after_window,
        ),
    )
