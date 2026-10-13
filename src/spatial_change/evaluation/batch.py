"""A resumable batch run of the pipeline over SceneDiff pairs.

Each pair gets a folder under ``<out>/pairs/<pair>``. It holds the outputs of
the run, the predicted masks, and ``result.json``, the record of the pair.
The record is written last and atomically, so a pair either has a complete
record or is run again. A record is reused only when it was made with the
same settings; a failed pair is recorded and the batch goes on.

``<out>/summary.json`` pools the pairs. It is a function of the records on
disk and of the ground truth, so ``summarize`` rebuilds it without models.

Importing this module does not load torch, transformers, open3d, or cv2.
"""

import gc
import hashlib
import json
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from spatial_change.diagnostics.errors import first_line
from spatial_change.domain import ChangeType
from spatial_change.evaluation._files import atomic_path
from spatial_change.evaluation.metrics import (
    Metrics,
    SceneEval,
    evaluate,
)
from spatial_change.evaluation.predictions import (
    Predictions,
    load_predictions,
    save_predictions,
)
from spatial_change.evaluation.scene import (
    PairEvalConfig,
    UncertainAs,
    build_scene,
    project_run,
)
from spatial_change.evaluation.scenediff import (
    AFTER_VIDEO,
    BEFORE_VIDEO,
    PairAnnotations,
    check_pair,
    eval_size,
    find_video,
    first_mask_shape,
    load_pair_annotations,
)
from spatial_change.pipeline import (
    MetadataValue,
    Models,
    PipelineResult,
    ProgressCallback,
    RunConfig,
    StageEvent,
    run_videos,
)
from spatial_change.pipeline.config import (
    MAX_VIDEO_KEYFRAMES,
    build_stage_configs,
    resolve_device,
)

RECORD_VERSION = 5
"""Format of ``result.json``; part of the settings, so a new format makes
every old record stale."""
PAIRS_DIR = "pairs"
RESULT_FILE = "result.json"
PREDICTIONS_FILE = "pred.npz"
RUN_DIR = "run"
SUMMARY_FILE = "summary.json"

_DEFAULT_EVAL = PairEvalConfig()
_FALLBACK_GRID = (1, 1)


class PairRunner(Protocol):
    """The seam between the batch and the pipeline: ``run_videos``."""

    def __call__(
        self,
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
        """Runs the pipeline on two videos."""


class PairRecord(BaseModel):
    """What one pair came to, as stored in its ``result.json``.

    Attributes:
        pair: Pair name.
        status: ``ok`` when the pair was run and scored, else ``failed``.
        settings: Digest of the settings the pair was run with.
        stride: Stride of the scored frames the pair was projected on.
        max_length: Long side of the evaluation grid the pair was projected
            on.
        seconds: Wall time of the pair, run and scoring.
        stage_seconds: Wall time of each stage that finished.
        cache_hits: The stages answered from the stage cache.
        n_gt: Ground-truth objects, ``None`` if the annotations were not
            read.
        n_changes: Changes the run reported.
        n_uncertain: Of those, the ones the run could not classify.
        uncertain_as: What the metrics of this record did with them:
            ``"drop"`` left them out, ``"moved"`` scored them as moved
            objects. The predictions hold them either way, so a summary may
            pool with the other setting; it then lists the pair with the
            metrics for that setting, and ``result.json`` stays as it was
            made.
        n_scored: Objects visible in a scored frame, which are the ones that
            can match; the others are left out by the projection.
        changes_by_type: Changes by type, uncertain ones included.
        floor_observed: Whether a floor was seen, as the run reported.
        camera_height_m: Median camera height of the run.
        after_cropped: Whether the AFTER frames were cropped to the aspect
            of the BEFORE frames.
        frames_before: BEFORE frames the reconstruction used.
        frames_after: AFTER frames the reconstruction used.
        frames_thinned: Whether a video had more frames on the stride than the
            reconstruction used, so some scored frames got no prediction;
            ``None`` if the run did not say.
        gt_matches_video: Whether the first ground-truth mask has the size of
            the BEFORE video; ``None`` without a mask or a video size. A
            ``False`` is recorded and the pair is still scored.
        metrics: The metrics of this pair alone.
        error_type: Class name of the error of a failed pair.
        error: First line of its message.
        failed_stage: A stage of the pipeline, or ``input``, ``annotations``,
            or ``setup`` before it, or ``scoring`` or ``record`` after it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair: str
    status: Literal["ok", "failed"]
    settings: str
    stride: int
    max_length: int
    seconds: float
    stage_seconds: dict[str, float]
    cache_hits: tuple[str, ...]
    n_gt: int | None
    n_changes: int
    n_uncertain: int
    uncertain_as: UncertainAs
    n_scored: int
    changes_by_type: dict[str, int]
    floor_observed: bool | None
    camera_height_m: float | None
    after_cropped: bool | None
    frames_before: int | None
    frames_after: int | None
    frames_thinned: bool | None
    gt_matches_video: bool | None
    metrics: Metrics | None
    error_type: str | None
    error: str | None
    failed_stage: str | None


class BatchSummary(BaseModel):
    """The pairs of a batch pooled.

    Attributes:
        total: Pairs asked for.
        ok: Pairs run and scored.
        failed: Pairs that failed.
        pending: Pairs without a usable record.
        uncertain_as: What the pooling did with uncertain objects, which
            may differ from what the records did.
        failure_rate: ``failed`` over ``ok + failed``, which is over
            ``total`` once no pair is pending.
        metrics: The headline: metrics pooled over all pairs, each failed
            pair scored as no predictions, so its ground truth counts as
            missed. ``None`` when no pair could be scored.
        metrics_ok_only: Metrics pooled over the ok pairs only, which leave
            out what the failures would have cost; ``None`` without one.
        floor_unobserved: Ok pairs in which no floor was seen.
        after_cropped: Ok pairs whose AFTER frames were cropped.
        frames_thinned: Ok pairs with more frames on the stride than the
            reconstruction used, so some scored frames have no prediction.
        gt_mismatch: Ok pairs whose ground truth is not the size of the
            BEFORE video.
        changes_by_type: Changes of the ok pairs by type.
        pairs: The records of the pairs that have one, in order, scored
            with ``uncertain_as`` like the pooled metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int
    ok: int
    failed: int
    pending: int
    uncertain_as: UncertainAs
    failure_rate: float
    metrics: Metrics | None
    metrics_ok_only: Metrics | None
    floor_unobserved: tuple[str, ...]
    after_cropped: tuple[str, ...]
    frames_thinned: tuple[str, ...]
    gt_mismatch: tuple[str, ...]
    changes_by_type: dict[str, int]
    pairs: tuple[PairRecord, ...]


@dataclass(frozen=True)
class PairEvent:
    """A pair is done.

    Attributes:
        position: Place of the pair in the batch, counting from 1.
        total: Pairs in the batch.
        record: Its record.
        resumed: Whether the record was reused rather than made now.
    """

    position: int
    total: int
    record: PairRecord
    resumed: bool


def _canonical(value: object) -> object:
    """Reduces a setting to plain JSON data, in a form that does not drift.

    Raises:
        TypeError: If the value has a type no setting should have, so an
            unstable ``repr`` never ends up in a digest.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return _canonical(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _canonical(getattr(value, f.name)) for f in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(_canonical(k)): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    raise TypeError(f"cannot digest a setting of type {type(value).__name__}")


def _effective(config: RunConfig, eval_config: PairEvalConfig) -> RunConfig:
    """Gives the run the frames that are scored, whatever the caller asked.

    The frame stride is that of ``eval_config``. The frame budget of each
    video is raised to the most the reconstruction takes from one video, so
    that no scored frame is thinned out, unless ``config`` sets it, as an
    ablation of the budget does.
    """
    update: dict[str, int] = {"frame_stride": eval_config.stride}
    if "max_keyframes" not in config.model_fields_set:
        update["max_keyframes"] = MAX_VIDEO_KEYFRAMES
    return config.model_copy(update=update)


def settings_digest(
    config: RunConfig, models: Models, eval_config: PairEvalConfig
) -> str:
    """Hashes everything that decides what a pair's record says.

    That is the knobs of the run, the config of every stage built from them
    and from the stage overrides, so an override equal to a default changes
    nothing, the weights and adapter settings of the models, how a run is
    projected, and the format of the record. The frame stride of ``config``
    is replaced by the stride of ``eval_config``, and a frame budget it
    leaves unset by the raised one the run uses. What scoring alone
    decides, the handling of uncertain objects, is not part of it, and
    neither is where the files live.

    Args:
        config: The knobs of the run.
        models: The models; only their ids and settings are read.
        eval_config: How a run is turned into a scored scene.

    Returns:
        The SHA-256 of their canonical JSON, in hex.
    """
    config = _effective(config, eval_config)
    stages = build_stage_configs(config, resolve_device(config))
    payload = {
        "record_version": RECORD_VERSION,
        "run": config.model_dump(mode="json", exclude={"stages"}),
        "stages": _canonical(stages),
        "models": {
            "ids": _canonical(models.ids),
            "settings": _canonical(models.settings),
        },
        "eval": _canonical(replace(eval_config, uncertain_as="drop")),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8", newline="\n")


def _intact_record(folder: Path, pair: str) -> PairRecord | None:
    """Reads the record of a pair, or ``None`` if it cannot be trusted.

    A record is trusted when it parses, names this pair rather than a folder
    copied from another, and, for an ok pair, has predictions that load.
    """
    try:
        text = (folder / RESULT_FILE).read_text(encoding="utf-8")
        record = PairRecord.model_validate_json(text)
        if record.pair != pair:
            return None
        if record.status == "ok":
            load_predictions(folder / PREDICTIONS_FILE)
    except (OSError, ValueError):
        return None
    return record


def _check_names(pairs: Sequence[str]) -> list[str]:
    """Rejects pair names that are unsafe as folder names or repeated.

    Raises:
        ValueError: If a name has a character other than letters, digits,
            ``_`` and ``-``, or a name appears twice.
    """
    names = list(pairs)
    for name in names:
        check_pair(name)
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"pair listed twice: {', '.join(repeated)}")
    return names


def _flag(metadata: Mapping[str, MetadataValue], key: str) -> bool | None:
    value = metadata.get(key)
    return value if isinstance(value, bool) else None


def _count(metadata: Mapping[str, MetadataValue], key: str) -> int | None:
    value = metadata.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _frames_thinned(metadata: Mapping[str, MetadataValue]) -> bool | None:
    """Tells whether a run used fewer frames than were on the stride."""
    thinned = []
    for epoch in ("before", "after"):
        strided = _count(metadata, f"strided_frames_{epoch}")
        used = _count(metadata, f"frames_{epoch}")
        if strided is None or used is None:
            return None
        thinned.append(strided > used)
    return any(thinned)


def _gt_matches_video(
    annotations: PairAnnotations, metadata: Mapping[str, MetadataValue]
) -> bool | None:
    mask = first_mask_shape(annotations)
    height = _count(metadata, "video_before_height")
    width = _count(metadata, "video_before_width")
    if mask is None or height is None or width is None:
        return None
    return mask == (height, width)


def _ok_record(
    pair: str,
    digest: str,
    seconds: float,
    result: PipelineResult,
    annotations: PairAnnotations,
    scene: SceneEval,
    metrics: Metrics,
    eval_config: PairEvalConfig,
) -> PairRecord:
    metadata = result.report.metadata
    hits = str(metadata.get("cache_hits", ""))
    height = metadata.get("camera_height_m")
    kinds = Counter(change.type.value for change in result.report.changes)
    return PairRecord(
        pair=pair,
        status="ok",
        settings=digest,
        stride=eval_config.stride,
        max_length=eval_config.max_length,
        seconds=seconds,
        stage_seconds=dict(result.seconds),
        cache_hits=tuple(hit for hit in hits.split(",") if hit),
        n_gt=len(scene.gt),
        n_changes=len(result.report.changes),
        n_uncertain=kinds[ChangeType.UNCERTAIN.value],
        uncertain_as=eval_config.uncertain_as,
        n_scored=len(scene.pred),
        changes_by_type=dict(sorted(kinds.items())),
        floor_observed=_flag(metadata, "floor_observed"),
        camera_height_m=float(height)
        if isinstance(height, int | float) and not isinstance(height, bool)
        else None,
        after_cropped=_flag(metadata, "after_cropped_to_before_aspect"),
        frames_before=_count(metadata, "frames_before"),
        frames_after=_count(metadata, "frames_after"),
        frames_thinned=_frames_thinned(metadata),
        gt_matches_video=_gt_matches_video(annotations, metadata),
        metrics=metrics,
        error_type=None,
        error=None,
        failed_stage=None,
    )


def _failed_record(
    pair: str,
    digest: str,
    seconds: float,
    stage_seconds: dict[str, float],
    n_gt: int | None,
    stage: str,
    error: Exception,
    eval_config: PairEvalConfig,
) -> PairRecord:
    return PairRecord(
        pair=pair,
        status="failed",
        settings=digest,
        stride=eval_config.stride,
        max_length=eval_config.max_length,
        seconds=seconds,
        stage_seconds=stage_seconds,
        cache_hits=(),
        n_gt=n_gt,
        n_changes=0,
        n_uncertain=0,
        uncertain_as=eval_config.uncertain_as,
        n_scored=0,
        changes_by_type={},
        floor_observed=None,
        camera_height_m=None,
        after_cropped=None,
        frames_before=None,
        frames_after=None,
        frames_thinned=None,
        gt_matches_video=None,
        metrics=None,
        error_type=type(error).__name__,
        error=first_line(error),
        failed_stage=stage,
    )


def _compute(
    pair: str,
    pair_dir: Path,
    pair_out: Path,
    digest: str,
    config: RunConfig,
    models: Models,
    eval_config: PairEvalConfig,
    runner: PairRunner,
    cache_dir: Path | None,
) -> PairRecord:
    """Runs and scores one pair, turning any failure into a record."""
    started = time.perf_counter()
    stage = "input"
    stage_seconds: dict[str, float] = {}
    n_gt: int | None = None

    def progress(event: StageEvent) -> None:
        nonlocal stage
        if event.seconds is None:
            stage = event.name
        else:
            stage_seconds[event.name] = event.seconds

    try:
        before = find_video(pair_dir, BEFORE_VIDEO)
        after = find_video(pair_dir, AFTER_VIDEO)
        stage = "annotations"
        annotations = load_pair_annotations(pair, pair_dir)
        n_gt = len(annotations.objects)
        stage = "setup"
        run_dir = pair_out / RUN_DIR
        shutil.rmtree(run_dir, ignore_errors=True)
        result = runner(
            before,
            after,
            run_dir,
            config,
            models,
            progress,
            overwrite=True,
            cache_dir=cache_dir,
        )
        stage = "scoring"
        predictions = project_run(result, annotations, eval_config)
        scene = build_scene(pair, annotations, predictions, eval_config)
        metrics = evaluate([scene])
        stage = "record"
        save_predictions(pair_out / PREDICTIONS_FILE, predictions)
        return _ok_record(
            pair,
            digest,
            time.perf_counter() - started,
            result,
            annotations,
            scene,
            metrics,
            eval_config,
        )
    except Exception as error:
        record = _failed_record(
            pair,
            digest,
            time.perf_counter() - started,
            stage_seconds,
            n_gt,
            stage,
            error,
            eval_config,
        )
    # Out of the except block the traceback no longer holds the tensors of
    # the stage that failed.
    gc.collect()
    return record


def _ok_scenes(
    records: Sequence[PairRecord],
    data_dir: Path,
    out_dir: Path,
    eval_config: PairEvalConfig,
) -> Iterator[SceneEval]:
    """Builds the scene of each ok pair from its files, one at a time."""
    for record in records:
        yield build_scene(
            record.pair,
            load_pair_annotations(record.pair, data_dir / record.pair),
            load_predictions(
                out_dir / PAIRS_DIR / record.pair / PREDICTIONS_FILE
            ),
            eval_config,
        )


def _missed_scenes(
    records: Sequence[PairRecord],
    data_dir: Path,
    eval_config: PairEvalConfig,
) -> Iterator[SceneEval]:
    """Builds the scene of each failed pair: its ground truth, no predictions.

    A pair whose annotations cannot be read has no ground truth to miss and
    is left out.
    """
    for record in records:
        try:
            annotations = load_pair_annotations(
                record.pair, data_dir / record.pair
            )
        except Exception:
            continue
        height, width = (
            eval_size(annotations, eval_config.max_length) or _FALLBACK_GRID
        )
        yield build_scene(
            record.pair,
            annotations,
            Predictions(height, width, [], frozenset()),
            eval_config,
        )


def _pooled(scenes: Iterator[SceneEval]) -> Metrics | None:
    """Scores scenes one at a time; ``None`` if there are none."""
    seen = 0

    def counted() -> Iterator[SceneEval]:
        nonlocal seen
        for scene in scenes:
            seen += 1
            yield scene

    metrics = evaluate(counted())
    return metrics if seen else None


def _same_frames(record: PairRecord, eval_config: PairEvalConfig) -> bool:
    return (record.stride, record.max_length) == (
        eval_config.stride,
        eval_config.max_length,
    )


def _as_asked(
    record: PairRecord,
    data_dir: Path,
    out_dir: Path,
    eval_config: PairEvalConfig,
) -> PairRecord:
    """Gives a record as a summary reports it, scored with its setting.

    A record made with the other ``uncertain_as`` has its metrics and scored
    count worked out again from its stored predictions, so a summary never
    mixes two settings. The record on disk is not touched.
    """
    if record.uncertain_as == eval_config.uncertain_as:
        return record
    update: dict[str, object] = {"uncertain_as": eval_config.uncertain_as}
    if record.status == "ok":
        (scene,) = _ok_scenes([record], data_dir, out_dir, eval_config)
        update.update(metrics=evaluate([scene]), n_scored=len(scene.pred))
    return record.model_copy(update=update)


def _collect(
    names: Sequence[str],
    data_dir: Path,
    out_dir: Path,
    eval_config: PairEvalConfig,
) -> BatchSummary:
    """Pools the records on disk into a summary.

    A record made on other frames than ``eval_config`` asks for is not
    pooled, as its predictions cannot be scored on that grid; it counts as
    pending. The other records are listed as scored with
    ``eval_config.uncertain_as``, like the pooled metrics.

    Raises:
        ValueError: If no pair is ok and some were made on other frames, so
            a mismatched setting cannot pass for a summary of pending pairs.
    """
    stored = []
    for pair in names:
        record = _intact_record(out_dir / PAIRS_DIR / pair, pair)
        if record is not None:
            stored.append(record)
    found = [
        _as_asked(r, data_dir, out_dir, eval_config)
        for r in stored
        if _same_frames(r, eval_config)
    ]
    ok = [r for r in found if r.status == "ok"]
    failed = [r for r in found if r.status == "failed"]
    if not ok and len(found) < len(stored):
        other = sorted(
            {(r.stride, r.max_length) for r in stored}
            - {(eval_config.stride, eval_config.max_length)}
        )
        raise ValueError(
            "pairs were made with "
            + " or ".join(f"stride {s}, max length {m}" for s, m in other)
            + f", not stride {eval_config.stride}, "
            f"max length {eval_config.max_length}"
        )
    metrics_ok_only = (
        _pooled(_ok_scenes(ok, data_dir, out_dir, eval_config)) if ok else None
    )
    metrics = _pooled(
        chain(
            _ok_scenes(ok, data_dir, out_dir, eval_config),
            _missed_scenes(failed, data_dir, eval_config),
        )
    )
    kinds: Counter[str] = Counter()
    for record in ok:
        kinds.update(record.changes_by_type)
    return BatchSummary(
        total=len(names),
        ok=len(ok),
        failed=len(failed),
        pending=len(names) - len(found),
        uncertain_as=eval_config.uncertain_as,
        failure_rate=len(failed) / len(found) if found else 0.0,
        metrics=metrics,
        metrics_ok_only=metrics_ok_only,
        floor_unobserved=tuple(r.pair for r in ok if r.floor_observed is False),
        after_cropped=tuple(r.pair for r in ok if r.after_cropped is True),
        frames_thinned=tuple(r.pair for r in ok if r.frames_thinned is True),
        gt_mismatch=tuple(r.pair for r in ok if r.gt_matches_video is False),
        changes_by_type=dict(sorted(kinds.items())),
        pairs=tuple(found),
    )


def _publish(out_dir: Path, summary: BatchSummary) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        out_dir / SUMMARY_FILE, summary.model_dump_json(indent=2) + "\n"
    )


def run_batch(
    pairs: Sequence[str],
    data_dir: Path,
    out_dir: Path,
    config: RunConfig,
    models: Models,
    *,
    eval_config: PairEvalConfig = _DEFAULT_EVAL,
    cache_dir: Path | None = None,
    runner: PairRunner = run_videos,
    on_pair: Callable[[PairEvent], None] | None = None,
) -> BatchSummary:
    """Runs and scores pairs, reusing what an earlier call finished.

    A pair whose record was made with the same settings and whose
    predictions are intact is not run again. A failed pair is recorded with
    the stage and the error, the batch goes on, and the pair is tried again
    by the next call. Only an interrupt of the process stops it, and the
    pairs finished by then are kept.

    Args:
        pairs: Pair names; each is a folder of ``data_dir`` holding
            ``segments.pkl``, ``video1.<mp4|mov>``, and ``video2.<mp4|mov>``.
        data_dir: Folder of the pairs.
        out_dir: Folder of the results, created if missing.
        config: The knobs of the run.
        models: Factories of the models.
        eval_config: How a run is turned into a scored scene.
        cache_dir: Stage cache shared by all pairs, or ``None``.
        runner: Runs the pipeline on two videos; a test passes a fake.
        on_pair: Called once per pair when it is done or reused.

    Returns:
        The pooled summary, also written to ``summary.json``.

    Raises:
        ValueError: If a pair name has characters that could leave the
            folder, or is repeated. Nothing is created then.
    """
    names = _check_names(pairs)
    digest = settings_digest(config, models, eval_config)
    config = _effective(config, eval_config)
    pairs_dir = out_dir / PAIRS_DIR
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for position, pair in enumerate(names, start=1):
        pair_out = pairs_dir / pair
        record = _intact_record(pair_out, pair)
        resumed = (
            record is not None
            and record.status == "ok"
            and record.settings == digest
        )
        if not resumed:
            pair_out.mkdir(parents=True, exist_ok=True)
            (pair_out / RESULT_FILE).unlink(missing_ok=True)
            (pair_out / PREDICTIONS_FILE).unlink(missing_ok=True)
            record = _compute(
                pair,
                data_dir / pair,
                pair_out,
                digest,
                config,
                models,
                eval_config,
                runner,
                cache_dir,
            )
            _write_atomic(
                pair_out / RESULT_FILE, record.model_dump_json(indent=2) + "\n"
            )
        if on_pair is not None and record is not None:
            on_pair(PairEvent(position, len(names), record, resumed))
    summary = _collect(names, data_dir, out_dir, eval_config)
    _publish(out_dir, summary)
    return summary


def summarize(
    pairs: Sequence[str],
    data_dir: Path,
    out_dir: Path,
    eval_config: PairEvalConfig = _DEFAULT_EVAL,
) -> BatchSummary:
    """Pools the records on disk again, without models or a GPU.

    The records are taken as they are, and the ground truth is read again at
    the stride of ``eval_config``. The stored predictions hold the uncertain
    objects whatever the run did with them, so ``eval_config.uncertain_as``
    may differ from the one the pairs were run with.

    Args:
        pairs: Pair names.
        data_dir: Folder of the pairs, for their ground truth.
        out_dir: Folder of the results.
        eval_config: How to pool the pairs. Its stride and max length must be
            the ones the pairs were run with.

    Returns:
        The pooled summary, also written to ``summary.json``. Pairs without
        a usable record count as pending.

    Raises:
        ValueError: If a pair name has characters that could leave the
            folder, or is repeated, or if no pair is ok and some were made
            with a stride or max length other than ``eval_config``'s.
            ``summary.json`` is left as it was then.
    """
    names = _check_names(pairs)
    summary = _collect(names, data_dir, out_dir, eval_config)
    _publish(out_dir, summary)
    return summary
