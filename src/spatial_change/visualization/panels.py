"""Image panels: keyframes with the changed objects projected into them.

Lifting does not keep the segmentation masks, so a panel recovers each changed
object from its box instead: the epoch's points inside the change's box are
projected into the frame and drawn as 2D points inside a labelled 2D box that
bounds them. Where a frame has depth, a point counts only if the frame sees it
on its depth surface, so objects behind nearer surfaces are not drawn.

Entity tree, below ``world/<epoch>/cameras/frame_<index>``, logged at the
step of the epoch::

    (the entity)     camera pose and pinhole, drawn as a frustum
    image            the downscaled keyframe
    overlay/<id>     points and bounding box of one changed object
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import (
    Change,
    Epoch,
    EpochReconstruction,
    Frame,
    Intrinsics,
)
from spatial_change.geometry import (
    in_image,
    project,
    usable_pixels,
    visible_points,
)
from spatial_change.geometry.projection import DEFAULT_MIN_CONFIDENCE
from spatial_change.visibility import VisibilityConfig
from spatial_change.visualization.config import RerunConfig
from spatial_change.visualization.layers import (
    at_step,
    epoch_path,
    rgba,
    subsample,
)

if TYPE_CHECKING:
    import rerun as rr

_DEPTH = VisibilityConfig()
_RERUN_PIXEL_SHIFT = 0.5


@dataclass(frozen=True)
class Panel:
    """A logged image panel.

    Attributes:
        name: Title of the panel's view.
        origin: Entity path of the panel's camera, without a leading slash.
    """

    name: str
    origin: str


@dataclass(frozen=True, eq=False)
class _Tracked:
    """The points of one changed object in one epoch.

    Attributes:
        change: The change.
        points: Up to ``panel_points`` of the epoch's points inside the
            change's box, shape ``(K, 3)``.
    """

    change: Change
    points: NDArray[np.float64]


class _View(NamedTuple):
    """How many points of each changed object a frame sees.

    Attributes:
        frame: A frame with an image.
        counts: Visible point count by change id, only for objects the frame
            sees.
    """

    frame: Frame
    counts: dict[str, int]


def rerun_pixels(pixels: NDArray[np.float64]) -> NDArray[np.float64]:
    """Converts pixel coordinates to the Rerun convention.

    The repo puts pixel centers at integer coordinates; Rerun puts the corner
    of the top-left pixel at (0, 0). Everything logged in pixels, the
    principal point included, goes through this shift of half a pixel.

    Args:
        pixels: Repo pixel coordinates of any shape.

    Returns:
        The same positions in Rerun pixel coordinates.
    """
    return pixels + _RERUN_PIXEL_SHIFT


def rerun_matrix(intrinsics: Intrinsics) -> NDArray[np.float64]:
    """Builds the calibration matrix in Rerun's pixel coordinates.

    Args:
        intrinsics: Intrinsics in the repo's convention.

    Returns:
        The 3x3 matrix with the principal point converted by
        ``rerun_pixels``.
    """
    matrix = intrinsics.matrix
    matrix[:2, 2] = rerun_pixels(matrix[:2, 2])
    return matrix


def _resized_pixels(
    uv: NDArray[np.float64], sx: float, sy: float
) -> NDArray[np.float64]:
    """Maps pixel coordinates through a resize, like ``Intrinsics.scaled``."""
    return np.array([sx, sy]) * (uv + 0.5) - 0.5


def _tracked(
    recon: EpochReconstruction, changes: list[Change], config: RerunConfig
) -> list[_Tracked]:
    found = []
    for change in changes:
        model = (
            change.before_box
            if recon.epoch is Epoch.BEFORE
            else change.after_box
        )
        if model is None:
            continue
        inside = recon.points[model.to_box().contains(recon.points)]
        keep = subsample(len(inside), config.panel_points, config.seed)
        if len(keep):
            found.append(_Tracked(change, inside[keep]))
    return found


def _project(
    tracked: list[_Tracked], frame: Frame
) -> dict[str, NDArray[np.float64]]:
    """Projects the tracked objects into a frame.

    Args:
        tracked: The changed objects of the frame's epoch.
        frame: A frame; with depth, a point counts only if the frame sees it
            on its depth surface.

    Returns:
        The pixel coordinates of the points it sees, by change id, only for
        objects it sees.
    """
    usable = (
        None
        if frame.depth is None
        else usable_pixels(frame, DEFAULT_MIN_CONFIDENCE)
    )
    seen = {}
    for item in tracked:
        if usable is not None:
            uv = visible_points(
                item.points,
                frame,
                usable,
                _DEPTH.abs_tolerance,
                _DEPTH.rel_tolerance,
            ).uv
        else:
            uv, depth = project(
                item.points, frame.intrinsics, frame.world_from_camera
            )
            uv = uv[in_image(uv, depth, frame.intrinsics)]
        if len(uv):
            seen[item.change.change_id] = uv
    return seen


def _choose(views: list[_View], limit: int) -> list[Frame]:
    """Picks the frames that get a panel.

    Frames that see a changed object are preferred; if none does, every frame
    is a candidate. More candidates than ``limit`` are thinned to evenly
    spaced ones, keeping the first and the last.

    Args:
        views: What each frame sees, in frame order.
        limit: Largest number of panels.

    Returns:
        At most ``limit`` frames, in frame order.
    """
    pool = [v.frame for v in views if v.counts] or [v.frame for v in views]
    if len(pool) <= limit:
        return pool
    picks = np.unique(np.round(np.linspace(0, len(pool) - 1, limit)))
    return [pool[int(i)] for i in picks]


def log_panels(
    stream: rr.RecordingStream,
    recon: EpochReconstruction,
    changes: list[Change],
    config: RerunConfig,
) -> list[Panel]:
    """Logs the image panels of an epoch at its step.

    Args:
        stream: The recording stream.
        recon: The epoch's reconstruction.
        changes: The detected changes.
        config: Panel cap, downscale factor, point cap, and colors.

    Returns:
        The panels that were logged, in frame order; empty if the cap is 0.
    """
    if config.max_image_panels == 0:
        return []
    tracked = _tracked(recon, changes, config)
    views = [
        _View(f, {k: len(v) for k, v in _project(tracked, f).items()})
        for f in recon.frames
        if f.image is not None
    ]
    at_step(stream, recon.epoch)
    return [
        _log_panel(stream, frame, _project(tracked, frame), changes, config)
        for frame in _choose(views, config.max_image_panels)
    ]


def _log_panel(
    stream: rr.RecordingStream,
    frame: Frame,
    seen: dict[str, NDArray[np.float64]],
    changes: list[Change],
    config: RerunConfig,
) -> Panel:
    import rerun as rr

    from spatial_change.capture.resize import resize_image

    assert frame.image is not None
    height, width = frame.image.shape[:2]
    new_width = max(1, round(width / config.image_downscale))
    new_height = max(1, round(height / config.image_downscale))
    sx, sy = new_width / width, new_height / height
    pose = frame.world_from_camera
    origin = epoch_path(frame.epoch, f"cameras/frame_{frame.index}")
    stream.log(
        origin,
        rr.Transform3D(translation=pose.translation, mat3x3=pose.rotation),
        rr.Pinhole(
            image_from_camera=rerun_matrix(frame.intrinsics.scaled(sx, sy)),
            resolution=[new_width, new_height],
        ),
    )
    stream.log(
        f"{origin}/image",
        rr.Image(resize_image(frame.image, new_width, new_height)),
    )
    for change in changes:
        uv = seen.get(change.change_id)
        if uv is not None:
            pixels = rerun_pixels(_resized_pixels(uv, sx, sy))
            _log_overlay(stream, origin, change, pixels, config)
    return Panel(
        f"{frame.epoch.value.capitalize()} frame {frame.index}", origin
    )


def _log_overlay(
    stream: rr.RecordingStream,
    origin: str,
    change: Change,
    pixels: NDArray[np.float64],
    config: RerunConfig,
) -> None:
    import rerun as rr

    color = rgba(config.change_colors[change.type])
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    stream.log(
        f"{origin}/overlay/{rr.escape_entity_path_part(change.change_id)}",
        rr.Points2D(pixels, colors=color),
        rr.Boxes2D(
            mins=[low],
            sizes=[high - low],
            colors=[color],
            labels=[change.label],
            show_labels=True,
        ),
    )
