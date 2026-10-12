"""Views of an object: the frames that see its points best, and their crops.

Lifting and labeling both look at an object in the frames of its own epoch
that see the most of its points on their depth surface.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import EpochReconstruction, Frame
from spatial_change.geometry.projection import (
    DEFAULT_MIN_CONFIDENCE,
    VisiblePoints,
    usable_pixels,
    visible_points,
)


@dataclass(frozen=True)
class ViewConfig:
    """How the views of an object are picked.

    Attributes:
        max_views: Most views to pick, at least 1.
        min_visible_points: Points a frame must see to give a view, at
            least 1.
        abs_tolerance: Depth tolerance at zero depth of the visibility
            test, positive and finite.
        rel_tolerance: Extra tolerance per unit of depth, at least 0 and
            finite.
        min_confidence: Pixels with a lower confidence give no view, in
            ``[0, 1]``.
    """

    max_views: int = 5
    min_visible_points: int = 10
    abs_tolerance: float = 0.05
    rel_tolerance: float = 0.02
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a count is below 1, a tolerance is out of range
                or not finite, or the confidence is outside ``[0, 1]``.
        """
        if self.max_views < 1:
            raise ValueError("max_views must be >= 1")
        if self.min_visible_points < 1:
            raise ValueError("min_visible_points must be >= 1")
        if not 0.0 < self.abs_tolerance < math.inf:
            raise ValueError("abs_tolerance must be positive and finite")
        if not 0.0 <= self.rel_tolerance < math.inf:
            raise ValueError("rel_tolerance must be >= 0 and finite")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")


@dataclass(frozen=True, eq=False)
class UsableFrame:
    """A frame that can give views.

    Attributes:
        frame: The frame, with image and depth.
        usable: The frame's usable pixels.
    """

    frame: Frame
    usable: NDArray[np.bool_]

    @property
    def image(self) -> NDArray[np.uint8]:
        """The frame's ``(H, W, 3)`` image."""
        assert self.frame.image is not None
        return self.frame.image

    @property
    def depth(self) -> NDArray[np.float32]:
        """The frame's ``(H, W)`` depth map."""
        assert self.frame.depth is not None
        return self.frame.depth


@dataclass(frozen=True, eq=False)
class ObjectView:
    """One frame's sight of an object.

    Attributes:
        position: Index of the frame in the reconstruction.
        seen: The object's points that the frame sees.
    """

    position: int
    seen: VisiblePoints


def usable_frames(
    reconstruction: EpochReconstruction, min_confidence: float
) -> dict[int, UsableFrame]:
    """Lists the frames that have an image and a depth map.

    Args:
        reconstruction: The epoch's reconstruction.
        min_confidence: Pixels with a lower confidence are not usable.

    Returns:
        The frames by their index in the reconstruction, in that order.
    """
    return {
        position: UsableFrame(frame, usable_pixels(frame, min_confidence))
        for position, frame in enumerate(reconstruction.frames)
        if frame.image is not None and frame.depth is not None
    }


def select_views(
    points: NDArray[np.float64],
    frames: Mapping[int, UsableFrame],
    config: ViewConfig,
) -> list[ObjectView]:
    """Picks the frames that see an object's points best.

    Args:
        points: ``(N, 3)`` world points of the object.
        frames: The usable frames of the object's epoch.
        config: How many views, how many points each must see, and the
            visibility tolerances.

    Returns:
        Up to ``max_views`` views in frame order, the frames that see the
        most points first chosen, ties to the earlier frame.
    """
    positions = list(frames)

    def see(position: int) -> VisiblePoints:
        item = frames[position]
        return visible_points(
            points,
            item.frame,
            item.usable,
            config.abs_tolerance,
            config.rel_tolerance,
        )

    counts = np.array([len(see(position).indices) for position in positions])
    ranked = np.argsort(-counts, kind="stable")
    chosen = sorted(
        int(i)
        for i in ranked[: config.max_views]
        if counts[i] >= config.min_visible_points
    )
    return [ObjectView(positions[i], see(positions[i])) for i in chosen]


def pixel_box(
    pixels: NDArray[np.intp],
    margin: float | NDArray[np.float64],
    width: int,
    height: int,
) -> NDArray[np.float64]:
    """Boxes pixels, grows the box by a margin, and clips it to the image.

    Args:
        pixels: ``(M, 2)`` pixels as (column, row), at least one.
        margin: Pixels added on every side, or per axis as (column, row).
        width: Image width.
        height: Image height.

    Returns:
        The ``(4,)`` box ``(x0, y0, x1, y1)`` in pixel-center coordinates,
        inside ``[0, width - 1] x [0, height - 1]``.
    """
    low = pixels.min(axis=0) - margin
    high = pixels.max(axis=0) + margin
    limit = np.array([width - 1, height - 1])
    return np.concatenate([np.maximum(low, 0.0), np.minimum(high, limit)])
