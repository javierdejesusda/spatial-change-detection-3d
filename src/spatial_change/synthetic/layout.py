"""Footprint geometry for rejection-sampled object placement.

Footprints are circles on the floor plane that circumscribe an object's
box, so a clear gap between two circles guarantees the boxes do not touch
whatever their yaw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import OrientedBox


@dataclass(frozen=True, eq=False)
class Disk:
    """A circular floor footprint.

    Attributes:
        center: World ``(x, y)`` of the center.
        radius: Radius in meters.
    """

    center: NDArray[np.float64]
    radius: float


@dataclass(frozen=True)
class Rect:
    """An axis-aligned floor rectangle.

    Attributes:
        x0: Minimum x.
        x1: Maximum x.
        y0: Minimum y.
        y1: Maximum y.
    """

    x0: float
    x1: float
    y0: float
    y1: float


def footprint(box: OrientedBox) -> Disk:
    """Returns the circle circumscribing a box on the floor plane.

    Args:
        box: A box with a vertical axis (yaw-only rotation).

    Returns:
        The disk centered on the box center.
    """
    radius = 0.5 * float(np.hypot(box.extent[0], box.extent[1]))
    return Disk(np.array(box.center[:2]), radius)


def disks_clear(a: Disk, b: Disk, gap: float) -> bool:
    """Tests whether two disks are separated by at least ``gap``.

    Args:
        a: First disk.
        b: Second disk.
        gap: Required clearance between the circles, in meters.

    Returns:
        True if the circles are at least ``gap`` apart.
    """
    return float(np.linalg.norm(a.center - b.center)) >= (
        a.radius + b.radius + gap
    )


def disk_clear_of_rect(disk: Disk, rect: Rect) -> bool:
    """Tests whether a disk stays out of a rectangle.

    Args:
        disk: The footprint.
        rect: The forbidden rectangle.

    Returns:
        True if the disk does not intersect the rectangle.
    """
    nearest_x = min(max(float(disk.center[0]), rect.x0), rect.x1)
    nearest_y = min(max(float(disk.center[1]), rect.y0), rect.y1)
    gap = np.hypot(disk.center[0] - nearest_x, disk.center[1] - nearest_y)
    return bool(gap >= disk.radius)


def disk_in_room(
    disk: Disk, room_size: tuple[float, float, float], margin: float
) -> bool:
    """Tests whether a disk lies inside the room walls.

    Args:
        disk: The footprint.
        room_size: Room extent ``(x, y, z)``.
        margin: Required distance from every wall.

    Returns:
        True if the disk is at least ``margin`` from all four walls.
    """
    low = disk.radius + margin
    return bool(
        low <= disk.center[0] <= room_size[0] - low
        and low <= disk.center[1] <= room_size[1] - low
    )
