"""Fetched pairs and finished runs for the evaluation tests.

A pair folder holds ``segments.pkl`` and two videos, as a fetched SceneDiff
pair does. The segments are drawn from the run itself, on a square grid that
the evaluation maps onto itself, so a faithful adapter scores them 1.
"""

import dataclasses
import pickle
from pathlib import Path
from typing import Any

from rle_helpers import encode_mask

from spatial_change.domain import ChangeType
from spatial_change.evaluation.scene import (
    PairEvalConfig,
    run_pred_items,
    scored_items,
    uncertain_ids,
)
from spatial_change.pipeline import PipelineResult
from spatial_change.pipeline.videos import ImageWindow

SIZES = {
    "video_before_width": 32,
    "video_before_height": 32,
    "video_after_width": 32,
    "video_after_height": 32,
    "after_cropped_to_before_aspect": False,
}
"""Metadata of a run whose two videos are square and of one size."""

CROPPED = {
    "video_before_width": 32,
    "video_before_height": 32,
    "video_after_width": 64,
    "video_after_height": 32,
    "after_cropped_to_before_aspect": True,
}
"""Metadata of a run whose AFTER video was twice as wide and was cropped."""

CROP_WINDOW = ImageWindow(0.0, 0.25, 1.0, 0.75)
"""The part of the AFTER image that is left of the one in ``CROPPED``."""


def own_segments(
    result: PipelineResult, config: PairEvalConfig
) -> dict[str, Any]:
    """Annotates a pair with exactly what the run itself projects.

    Args:
        result: The run.
        config: Its evaluation settings; ``max_length`` is the side of the
            square grid the masks are drawn on.

    Returns:
        The content of a ``segments.pkl``.
    """
    items = scored_items(
        run_pred_items(result, config, config.max_length, config.max_length),
        uncertain_ids(result.report.changes),
        config.uncertain_as,
    )
    video1: dict[int, dict[str, object]] = {}
    video2: dict[int, dict[str, object]] = {}
    flags = []
    for item in items:
        sides = {1: video1, 2: video2}
        for (video, frame), mask in item.masks.items():
            sides[video].setdefault(item.obj_id, {})[str(frame)] = encode_mask(
                mask
            )
        flags.append(
            {
                "original_obj_idx": item.obj_id,
                "in_video1": any(v == 1 for v, _ in item.masks),
                "in_video2": any(v == 2 for v, _ in item.masks),
            }
        )
    return {
        "objects": flags,
        "video1_objects": video1,
        "video2_objects": video2,
    }


def with_metadata(
    result: PipelineResult, **extra: str | int | float | bool
) -> PipelineResult:
    """Copies a run with more entries in the metadata of its report."""
    report = result.report.model_copy(
        update={"metadata": {**result.report.metadata, **extra}}
    )
    return dataclasses.replace(result, report=report)


def with_after_window(
    result: PipelineResult, window: ImageWindow
) -> PipelineResult:
    """Copies a run whose AFTER frames show only a window of their image."""
    scene = dataclasses.replace(result.scene, after_window=window)
    return dataclasses.replace(result, scene=scene)


def with_uncertain_move(
    result: PipelineResult, confidence: float = 0.2
) -> PipelineResult:
    """Copies a run in which the first moved change is uncertain instead.

    The objects behind the change stay the same, so it can be scored as
    moved or dropped.

    Args:
        result: A run with at least one moved change.
        confidence: Confidence given to the uncertain change.

    Returns:
        The copy.
    """
    changes = list(result.report.changes)
    index = next(i for i, c in enumerate(changes) if c.type is ChangeType.MOVED)
    changes[index] = changes[index].model_copy(
        update={
            "type": ChangeType.UNCERTAIN,
            "rotation_deg": None,
            "confidence": confidence,
        }
    )
    report = result.report.model_copy(update={"changes": changes})
    return dataclasses.replace(result, report=report)


def write_pair(
    data_dir: Path,
    name: str,
    segments: object,
    videos: tuple[str, str] = ("video1.mp4", "video2.mov"),
) -> Path:
    """Writes a pair folder with empty videos and the given segments.

    Args:
        data_dir: Folder that receives the pair.
        name: Pair name.
        segments: Pickled as it is, with the standard library.
        videos: File names of the two videos; empty files are written.

    Returns:
        The pair folder.
    """
    folder = data_dir / name
    folder.mkdir(parents=True)
    for video in videos:
        (folder / video).write_bytes(b"")
    with (folder / "segments.pkl").open("wb") as handle:
        pickle.dump(segments, handle)
    return folder
