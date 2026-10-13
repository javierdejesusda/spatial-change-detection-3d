"""Projection of predicted 3D change objects to per-frame 2D masks.

SceneDiff scores 2D masks, while the pipeline outputs 3D objects. Each
object's points are projected into the posed frames of the epoch in which
the object is present, and a point is drawn only where it is not hidden by
the frame's observed depth. Video 1 is the ``BEFORE`` epoch and video 2 is
the ``AFTER`` epoch.

Visibility rule: a point at camera depth ``z`` lands on pixel ``p`` and is
drawn when the pixel has a valid depth ``d`` with enough confidence and
``z <= d + abs_tolerance + rel_tolerance * z``. A point behind the observed
surface is occluded and dropped; a point in front of it is kept, because
depth noise puts object points slightly in front of their own surface.
Pixels without a usable depth never show a point, since occlusion cannot be
ruled out there. Each visible point is then splatted into a disk of
``splat_radius`` pixels to turn a sparse point cloud into a connected mask;
the splat is not depth tested and can spill a few pixels past an occluding
edge.

The frames of video 2 may show only a window of the image the ground truth
was drawn on, because the pipeline center-crops them to the aspect of video
1. Their masks are then placed at that window of the evaluation grid and the
rest of the grid stays empty, since nothing was observed there.

An object with no visible point in a frame gets no mask there. Because the
metrics derive the moved type from the presence of masks in both videos, a
moved object seen in only one video is scored as static.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import ChangeType, Epoch, Frame
from spatial_change.evaluation._masks import resize_nearest, splat
from spatial_change.evaluation.metrics import PredItem
from spatial_change.evaluation.scenediff import DEFAULT_FRAME_STRIDE
from spatial_change.geometry import depth_residual, usable_pixels
from spatial_change.geometry.projection import DEFAULT_MIN_CONFIDENCE
from spatial_change.pipeline.videos import WHOLE_IMAGE, ImageWindow

_REQUIRED_EPOCHS = {
    ChangeType.ADDED: frozenset({Epoch.AFTER}),
    ChangeType.REMOVED: frozenset({Epoch.BEFORE}),
    ChangeType.MOVED: frozenset({Epoch.BEFORE, Epoch.AFTER}),
}


@dataclass(frozen=True)
class ProjectionConfig:
    """Parameters of the depth-tested projection.

    Attributes:
        abs_tolerance: Depth slack at zero depth, in reconstruction units.
        rel_tolerance: Extra depth slack per unit of depth.
        splat_radius: Radius in pixels of the disk drawn per visible point.
        min_confidence: Minimum per-pixel depth confidence.
    """

    abs_tolerance: float = 0.0
    rel_tolerance: float = 0.05
    splat_radius: int = 2
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a tolerance or the splat radius is negative.
        """
        if self.abs_tolerance < 0 or self.rel_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if self.splat_radius < 0:
            raise ValueError("splat_radius must be non-negative")


@dataclass(frozen=True, eq=False)
class ChangeObject3D:
    """A predicted change object with its points per epoch.

    Attributes:
        obj_id: Object id.
        change_type: Added, removed or moved.
        score: Confidence used for every mask of the object.
        points: World points of shape ``(N, 3)`` per epoch in which the
            object is present: ``AFTER`` for added, ``BEFORE`` for removed,
            both for moved.
    """

    obj_id: int
    change_type: ChangeType
    score: float
    points: Mapping[Epoch, NDArray[np.float64]]

    def __post_init__(self) -> None:
        """Checks the type against the epochs and the point arrays.

        Raises:
            ValueError: If the type is not added, removed or moved, the
                epochs do not match the type, or a point array is not
                ``(N, 3)``.
        """
        required = _REQUIRED_EPOCHS.get(self.change_type)
        if required is None:
            raise ValueError(
                f"{self.change_type.value} objects cannot be scored"
            )
        if set(self.points) != required:
            raise ValueError(
                f"{self.change_type.value} objects need points in exactly "
                f"{sorted(e.value for e in required)}"
            )
        for pts in self.points.values():
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError("points must have shape (N, 3)")


def _render(
    obj: ChangeObject3D,
    frame: Frame,
    usable: NDArray[np.bool_],
    config: ProjectionConfig,
) -> NDArray[np.bool_] | None:
    points = obj.points.get(frame.epoch)
    if points is None or len(points) == 0:
        return None
    res = depth_residual(points, frame, usable)
    seen = res.residual <= config.abs_tolerance + config.rel_tolerance * res.z
    if not seen.any():
        return None
    intr = frame.intrinsics
    return splat(
        res.pixels[seen, 1],
        res.pixels[seen, 0],
        intr.height,
        intr.width,
        config.splat_radius,
    )


def _place_in_window(
    mask: NDArray[np.bool_], height: int, width: int, window: ImageWindow
) -> NDArray[np.bool_]:
    """Resizes a mask into a window of an otherwise empty grid.

    Args:
        mask: The mask of a frame that shows only the window.
        height: Grid height.
        width: Grid width.
        window: The part of the grid the frame shows.

    Returns:
        The ``(height, width)`` mask.
    """
    rows, cols = window.slices(height, width)
    placed = np.zeros((height, width), dtype=bool)
    placed[rows, cols] = resize_nearest(
        mask, rows.stop - rows.start, cols.stop - cols.start
    )
    return placed


def project_to_pred_items(
    objects: Sequence[ChangeObject3D],
    before_frames: Sequence[Frame],
    after_frames: Sequence[Frame],
    height: int,
    width: int,
    *,
    config: ProjectionConfig | None = None,
    stride: int = DEFAULT_FRAME_STRIDE,
    after_window: ImageWindow = WHOLE_IMAGE,
) -> list[PredItem]:
    """Projects objects into the sampled frames of both videos.

    Only frames whose index is a multiple of ``stride`` are used, keyed as
    ``index // stride`` like the ground truth. Masks are resized to the
    evaluation grid with nearest sampling.

    Args:
        objects: Predicted objects.
        before_frames: Posed frames with depth of video 1.
        after_frames: Posed frames with depth of video 2.
        height: Evaluation grid height.
        width: Evaluation grid width.
        config: Projection parameters.
        stride: Frame sampling stride of the evaluation.
        after_window: The part of the image of video 2 that its frames show;
            its masks are placed at this window of the grid. Video 1 always
            fills the grid.

    Returns:
        One item per object visible in at least one frame, in input order.

    Raises:
        ValueError: If ``stride`` is not positive, a frame is in the wrong
            epoch list, or a frame has no depth.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    config = config or ProjectionConfig()
    if any(f.epoch is not Epoch.BEFORE for f in before_frames):
        raise ValueError("before_frames must all be BEFORE frames")
    if any(f.epoch is not Epoch.AFTER for f in after_frames):
        raise ValueError("after_frames must all be AFTER frames")
    sampled = [
        (video, frame, usable_pixels(frame, config.min_confidence))
        for video, frames in ((1, before_frames), (2, after_frames))
        for frame in frames
        if frame.index % stride == 0
    ]
    items = []
    for obj in objects:
        masks: dict[tuple[int, int], NDArray[np.bool_]] = {}
        for video, frame, usable in sampled:
            mask = _render(obj, frame, usable, config)
            if mask is not None:
                key = (video, frame.index // stride)
                if video == 2:
                    masks[key] = _place_in_window(
                        mask, height, width, after_window
                    )
                else:
                    masks[key] = resize_nearest(mask, height, width)
        if masks:
            items.append(
                PredItem(obj.obj_id, masks, dict.fromkeys(masks, obj.score))
            )
    return items
