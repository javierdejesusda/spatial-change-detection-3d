"""Line segments that draw boxes, moves, and unobserved regions.

Everything here is in the z-up room frame of the report. The HTML view
and the GLB export convert to their own axes at the edge.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import (
    Change,
    ChangeType,
    OrientedBox,
    UnobservedRegion,
)

# Pairs of corner indices that differ in exactly one local axis sign, with
# the corner order of ``OrientedBox.corners`` (x sign changes slowest).
_BOX_EDGES = tuple(
    (i, i | bit) for i in range(8) for bit in (4, 2, 1) if not i & bit
)
# Arrow head length as a share of the shaft, and its largest absolute size.
_HEAD_SHARE = 0.25
_HEAD_MAX = 0.25
_HEAD_SPREAD = 0.4
_PARALLEL = 0.99


class Role(StrEnum):
    """What a group of segments draws."""

    BEFORE_BOX = "before_box"
    AFTER_BOX = "after_box"
    ARROW = "arrow"
    REGION_BOX = "region_box"


def box_edges(box: OrientedBox) -> NDArray[np.float64]:
    """Returns the twelve edges of a box as line segments.

    Args:
        box: The box.

    Returns:
        Array of shape ``(12, 2, 3)`` of segment endpoints.
    """
    corners = box.corners()
    return np.array([[corners[a], corners[b]] for a, b in _BOX_EDGES])


def arrow_segments(
    start: NDArray[np.float64], end: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Returns the shaft and two head barbs of an arrow.

    Args:
        start: Tail of the arrow, shape ``(3,)``.
        end: Tip of the arrow, shape ``(3,)``.

    Returns:
        Array of shape ``(3, 2, 3)`` of segment endpoints: the shaft, then
        the two barbs that run from the tip back along the shaft. Shape
        ``(0, 2, 3)`` when the arrow has no length.
    """
    shaft = np.asarray(end, dtype=np.float64) - start
    length = float(np.linalg.norm(shaft))
    if length == 0.0:
        return np.zeros((0, 2, 3))
    direction = shaft / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ helper)) > _PARALLEL:
        helper = np.array([0.0, 1.0, 0.0])
    side = np.cross(direction, helper)
    side /= np.linalg.norm(side)
    head = min(_HEAD_SHARE * length, _HEAD_MAX)
    base = end - head * direction
    barbs = [base + sign * _HEAD_SPREAD * head * side for sign in (1, -1)]
    return np.array(
        [[start, end], [end, barbs[0]], [end, barbs[1]]], dtype=np.float64
    )


def change_segments(
    change: Change,
) -> list[tuple[Role, NDArray[np.float64]]]:
    """Returns the line segments that draw a change, by role.

    Args:
        change: The change to draw.

    Returns:
        ``(role, segments)`` pairs, each segments array of shape
        ``(n, 2, 3)`` in room coordinates: ``BEFORE_BOX`` and
        ``AFTER_BOX`` for the boxes the change has, then ``ARROW`` for a
        move whose box centers differ. Empty when the change has no box.
    """
    parts: list[tuple[Role, NDArray[np.float64]]] = []
    if change.before_box is not None:
        parts.append((Role.BEFORE_BOX, box_edges(change.before_box.to_box())))
    if change.after_box is not None:
        parts.append((Role.AFTER_BOX, box_edges(change.after_box.to_box())))
    if (
        change.type is ChangeType.MOVED
        and change.before_box is not None
        and change.after_box is not None
    ):
        start = np.array(change.before_box.center)
        end = np.array(change.after_box.center)
        parts.append((Role.ARROW, arrow_segments(start, end)))
    return [(role, seg) for role, seg in parts if len(seg) > 0]


def region_segments(
    region: UnobservedRegion,
) -> list[tuple[Role, NDArray[np.float64]]]:
    """Returns the line segments that draw an unobserved region.

    Args:
        region: The region to draw.

    Returns:
        One ``(Role.REGION_BOX, segments)`` pair of shape ``(12, 2, 3)``.
    """
    return [(Role.REGION_BOX, box_edges(region.box.to_box()))]
