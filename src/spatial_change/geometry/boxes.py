"""Gravity-aligned boxes around point sets in the z-up room frame."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import OrientedBox
from spatial_change.domain.transforms import as_finite_points

# Relative margin by which the side across must exceed the side along the
# edge before ``+x`` of the box turns a quarter to follow it. It keeps a
# square footprint, whose two sides differ only by rounding, on one yaw.
_SIDE_TIE = 1e-9
# Padding of the extents, in ulps of the largest coordinate, so that the
# rounding of the rotation round trip in ``OrientedBox.contains`` never
# excludes a point on the boundary.
_PAD_ULPS = 16.0
# Entries of the edge-by-vertex projection matrix handled at once.
_BLOCK_ENTRIES = 1 << 20
# Margin, in ulps of the area of the bounding box, by which a point must lie
# inside the quadrilateral of the extremes to be dropped. It exceeds the
# rounding error of the side test, so a point on the boundary is kept.
_INTERIOR_ULPS = 16.0


def _drop_interior(xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Drops the points strictly inside the quadrilateral of the extremes.

    The points with the lowest and highest x and y lie on the convex hull,
    so a point strictly inside the quadrilateral they span is no hull
    vertex. A point on its boundary, or closer to it than the rounding of
    the test, is kept. Fewer than three distinct extremes enclose nothing
    and keep every point.

    Args:
        xy: Points of shape ``(N, 2)``, at least one row.

    Returns:
        The remaining points, shape ``(M, 2)``, in their order.
    """
    x, y = xy[:, 0], xy[:, 1]
    ring = xy[[x.argmin(), y.argmin(), x.argmax(), y.argmax()]].tolist()
    corners = [p for i, p in enumerate(ring) if p != ring[i - 1]]
    if len(corners) < 3:
        return xy
    margin = _INTERIOR_ULPS * np.finfo(np.float64).eps * np.ptp(x) * np.ptp(y)
    inside = np.ones(len(xy), dtype=bool)
    for i, (ax, ay) in enumerate(corners):
        bx, by = corners[(i + 1) % len(corners)]
        inside &= (bx - ax) * (y - ay) - (by - ay) * (x - ax) > margin
    return xy[~inside]


def _convex_hull(xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Finds the convex hull of planar points.

    Andrew's monotone chain on the lexicographically sorted distinct
    points that remain after ``_drop_interior``, which leaves a blob of
    thousands of points with a few dozen. Collinear points on the boundary
    are dropped.

    Args:
        xy: Points of shape ``(N, 2)``, at least one row.

    Returns:
        The hull vertices counter-clockwise, starting at the smallest point
        in lexicographic order, shape ``(H, 2)``. One point gives that
        point and collinear points give their two extremes.
    """
    rows = np.unique(_drop_interior(xy), axis=0).tolist()
    if len(rows) < 3:
        return np.array(rows)
    halves = []
    for sequence in (rows, rows[::-1]):
        chain: list[list[float]] = []
        for x, y in sequence:
            while len(chain) >= 2:
                (ax, ay), (bx, by) = chain[-2], chain[-1]
                if (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0.0:
                    break
                chain.pop()
            chain.append([x, y])
        halves.append(chain[:-1])
    return np.array(halves[0] + halves[1])


def _rectangle_sides(
    hull: NDArray[np.float64], angles: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Measures the bounding rectangle of a hull at several angles.

    Args:
        hull: Hull vertices, shape ``(H, 2)``.
        angles: Directions of the first rectangle side in radians, shape
            ``(E,)``.

    Returns:
        Array of shape ``(E, 2)``: for every angle the extent of the hull
        along that direction and along the one a quarter turn after it.
    """
    sides = np.empty((len(angles), 2))
    step = max(1, _BLOCK_ENTRIES // len(hull))
    for start in range(0, len(angles), step):
        cos = np.cos(angles[start : start + step])
        sin = np.sin(angles[start : start + step])
        along = hull @ np.stack([cos, sin])
        across = hull @ np.stack([-sin, cos])
        sides[start : start + step, 0] = np.ptp(along, axis=0)
        sides[start : start + step, 1] = np.ptp(across, axis=0)
    return sides


def _footprint_yaw(xy: NDArray[np.float64]) -> float:
    """Finds the yaw of the smallest-area rectangle around a footprint.

    The smallest rectangle has a side collinear with an edge of the convex
    hull, so the direction of every hull edge is tried and the one with
    the smallest area wins (the first in hull order on a tie). The edge
    angle is taken modulo a quarter turn into ``(-45, 45]`` degrees, so that
    equal footprints give equal yaws. ``+x`` then follows the longer side:
    the yaw is a quarter turn further when the side across is longer.

    Args:
        xy: Footprint points of shape ``(N, 2)``, at least one row.

    Returns:
        The yaw in radians, within ``(-45, 135]`` degrees. A footprint of a
        single position has yaw 0.
    """
    hull = _convex_hull(xy)
    if len(hull) == 1:
        return 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    quarter = math.pi / 2.0
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = quarter / 2.0 - (quarter / 2.0 - angles) % quarter
    sides = _rectangle_sides(hull, angles)
    best = int(np.argmin(sides[:, 0] * sides[:, 1]))
    along, across = sides[best]
    turn = quarter if across > along * (1.0 + _SIDE_TIE) else 0.0
    return float(angles[best]) + turn


def yaw_rotation(yaw: float) -> NDArray[np.float64]:
    """Builds a counterclockwise rotation about the z axis.

    Args:
        yaw: Angle in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_upright_box(points: ArrayLike, min_extent: float) -> OrientedBox:
    """Fits a box that is upright in the z-up room frame around points.

    The box is rotated about ``+z`` only. Its yaw is the one of the
    smallest-area rectangle around the xy footprint of the points, found
    exactly from the edges of the convex hull. ``+x`` of the box follows the
    longer side; a square footprint takes the yaw within ``(-45, 45]``
    degrees, and a single xy position has yaw 0. The same footprint gives the
    same rotation whatever the order or the offset of the points. The
    extents are the ranges of all the points in the yawed frame and the
    center is the middle of those ranges. The extents are padded by a few
    ulps of the largest coordinate, so rounding never leaves a boundary
    point outside.

    Args:
        points: Room-frame points of shape ``(N, 3)``.
        min_extent: Smallest extent along every box axis, positive. A flat,
            thin, or single-point set gets a box of at least this size
            around its middle.

    Returns:
        The box, which contains every input point.

    Raises:
        ValueError: If ``points`` is not a finite ``(N, 3)`` array with at
            least one row, or ``min_extent`` is not positive and finite.
    """
    cloud = as_finite_points(points, "points", 1)
    if not 0.0 < min_extent < math.inf:
        raise ValueError("min_extent must be positive and finite")
    yaw = _footprint_yaw(cloud[:, :2])
    rotation = yaw_rotation(yaw)
    local = cloud @ rotation
    low, high = local.min(axis=0), local.max(axis=0)
    pad = _PAD_ULPS * np.finfo(np.float64).eps * max(1.0, np.abs(cloud).max())
    extent = np.maximum(high - low, min_extent) + pad
    return OrientedBox(rotation @ ((low + high) / 2.0), extent, rotation)
