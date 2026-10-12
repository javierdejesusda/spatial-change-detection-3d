"""A synthetic scene pair turned into the inputs of the report and the GLB.

The changes come from the association pipeline with the oracle segmenter and
the fake encoder. The points take their colors from the rendered images, and
each crop is the tightest box around the true object in the frame that shows
it the most.
"""

import functools
import time
from dataclasses import dataclass

import numpy as np
from matching_pipeline import oracle_segmenter, run_pipeline
from numpy.typing import NDArray
from visibility_scenes import scene_evidence

from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeReport,
    Epoch,
    OrientedBox,
    UnobservedRegion,
)
from spatial_change.geometry import unproject_depth
from spatial_change.perception import FakeFeatureEncoder
from spatial_change.reporting import ChangeCrops, ReportInputs
from spatial_change.synthetic.scene import (
    SyntheticEpoch,
    SyntheticScenePair,
)
from spatial_change.visibility import VisibilityConfig
from spatial_change.visualization import ScenePoints

PATCH = 8
MARGIN_PX = 4
MAX_MATCH_M = 0.5


@dataclass(frozen=True, eq=False)
class SyntheticReport:
    """The inputs of both outputs for one scene pair.

    Attributes:
        inputs: The inputs of the HTML report.
        seconds: Wall time of the association pipeline.
    """

    inputs: ReportInputs
    seconds: float


def colored_cloud(data: SyntheticEpoch) -> ScenePoints:
    """Unprojects every valid pixel of an epoch with its image color.

    Args:
        data: Rendered frames and instance maps.

    Returns:
        The world points with the color of their pixel.
    """
    points, colors = [], []
    for frame in data.frames:
        assert frame.depth is not None and frame.image is not None
        valid = np.isfinite(frame.depth) & (frame.depth > 0)
        points.append(
            unproject_depth(
                frame.depth, frame.intrinsics, frame.world_from_camera
            )
        )
        colors.append(frame.image[valid])
    return ScenePoints(np.concatenate(points), np.concatenate(colors))


def crop_of(data: SyntheticEpoch, index: int) -> NDArray[np.uint8] | None:
    """Crops an object out of the frame that shows it the most.

    Args:
        data: Rendered frames and instance maps.
        index: Index of the object.

    Returns:
        The padded bounding box of the object, or ``None`` if no frame
        shows it.
    """
    areas = [int((m == index).sum()) for m in data.instance_maps]
    best = int(np.argmax(areas))
    if areas[best] == 0:
        return None
    image = data.frames[best].image
    assert image is not None
    rows, cols = np.nonzero(data.instance_maps[best] == index)
    top = max(0, int(rows.min()) - MARGIN_PX)
    left = max(0, int(cols.min()) - MARGIN_PX)
    bottom = int(rows.max()) + MARGIN_PX + 1
    right = int(cols.max()) + MARGIN_PX + 1
    return np.ascontiguousarray(image[top:bottom, left:right])


def _gap(
    change_boxes: list[OrientedBox], truth_boxes: list[OrientedBox]
) -> float:
    """Returns the smallest distance between two sets of box centers."""
    return min(
        (
            float(np.linalg.norm(a.center - b.center))
            for a in change_boxes
            for b in truth_boxes
        ),
        default=float("inf"),
    )


def crops_for(pair: SyntheticScenePair, change: Change) -> ChangeCrops:
    """Crops the true object that lies under the boxes of a change.

    The object is the one whose boxes have a center nearest to a box of the
    change, within ``MAX_MATCH_M``. A crop is made only for an epoch in
    which the change has a box.

    Args:
        pair: The scene pair with the true objects.
        change: The change to find crops for.

    Returns:
        The crops, both ``None`` if no object is near enough.
    """
    boxes = [b.to_box() for b in (change.before_box, change.after_box) if b]
    nearest = min(
        pair.objects,
        key=lambda o: _gap(
            boxes, [b for b in (o.box_before, o.box_after) if b is not None]
        ),
        default=None,
    )
    if nearest is None or not boxes:
        return ChangeCrops(None, None)
    truth = [b for b in (nearest.box_before, nearest.box_after) if b]
    if _gap(boxes, truth) > MAX_MATCH_M:
        return ChangeCrops(None, None)
    return ChangeCrops(
        before=(
            crop_of(pair.before, nearest.index)
            if change.before_box is not None
            else None
        ),
        after=(
            crop_of(pair.after, nearest.index)
            if change.after_box is not None
            else None
        ),
    )


@functools.cache
def synthetic_report(seed: int) -> SyntheticReport:
    """Runs the pipeline on a seeded scene pair and gathers the inputs.

    Args:
        seed: Scene seed.

    Returns:
        The report inputs with points, crops, and the pipeline time.
    """
    votes = scene_evidence(seed, VisibilityConfig())
    pair = votes.pair
    started = time.perf_counter()
    run = run_pipeline(votes, oracle_segmenter(pair), FakeFeatureEncoder(PATCH))
    seconds = time.perf_counter() - started
    report = ChangeReport(
        changes=list(run.changes),
        unobserved_regions=[
            UnobservedRegion(
                region_id="zone-1",
                box=BoxModel.from_box(pair.unobserved_zone),
                epoch_seen=pair.zone_seen_in,
            )
        ],
        metadata={"seed": seed, "frames_per_epoch": len(pair.before.frames)},
    )
    inputs = ReportInputs(
        report=report,
        summary=(
            f"Synthetic room, seed {seed}: {len(report.changes)} changes "
            f"between two captures of {len(pair.before.frames)} frames."
        ),
        crops={c.change_id: crops_for(pair, c) for c in report.changes},
        points={
            Epoch.BEFORE: colored_cloud(pair.before),
            Epoch.AFTER: colored_cloud(pair.after),
        },
    )
    return SyntheticReport(inputs, seconds)
