"""Vertical walls found in a point cloud by a seeded RANSAC."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.geometry.voxels import pack_cells, voxel_cells

_REFIT_ROUNDS = 10
"""Most least-squares refits of a wall plane."""

_WALL_PAIR_SHARE = 0.5
"""Two sampled points define a wall plane only when they are at least this
share of the minimum wall width apart, so a near pair cannot tilt it."""


@dataclass(frozen=True)
class WallConfig:
    """Parameters of the wall search. The defaults are provisional.

    Lengths are in the units of the world frame, with z up.

    Attributes:
        plane_distance: Distance from a vertical plane within which a
            point is on the wall, positive and finite.
        min_width: Extent along the plane that a wall needs, positive and
            finite.
        min_height: Extent in z that a wall needs, positive and finite.
        max_gap: Largest gap along the plane inside one wall, positive and
            finite; furniture in line with a wall beyond a larger gap is
            not part of it.
        min_fill: Share of the wall's rectangle (along the plane by z) that
            its inliers must occupy, in ``(0, 1]``; it tells a continuous
            wall from a plane that slices through separate pieces of
            furniture.
        max_planes: Walls found at most, at least 0.
        ransac_iterations: Planes tried per wall, at least 1.
        seed: Seed of the plane sampling.
    """

    plane_distance: float = 0.03
    min_width: float = 1.5
    min_height: float = 1.2
    max_gap: float = 0.3
    min_fill: float = 0.75
    max_planes: int = 8
    ransac_iterations: int = 300
    seed: int = 0

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a length is out of range or not finite, or a
                count or share is out of range.
        """
        for name in ("plane_distance", "min_width", "min_height", "max_gap"):
            if not 0.0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.min_fill <= 1.0:
            raise ValueError("min_fill must be within (0, 1]")
        if self.max_planes < 0:
            raise ValueError("max_planes must be >= 0")
        if self.ransac_iterations < 1:
            raise ValueError("ransac_iterations must be >= 1")


@dataclass(frozen=True, eq=False)
class Wall:
    """A vertical plane and the stretch of it that is wall.

    Attributes:
        normal: ``(2,)`` unit normal in the xy plane.
        offset: Plane position along the normal.
        along: ``(2,)`` unit direction of the plane.
        low: Smallest position of the inliers along the plane.
        high: Largest position of the inliers along the plane.
    """

    normal: NDArray[np.float64]
    offset: float
    along: NDArray[np.float64]
    low: float
    high: float

    def contains(
        self, points: NDArray[np.float64], distance: float
    ) -> NDArray[np.bool_]:
        """Marks the points on the wall.

        Args:
            points: ``(N, 3)`` points.
            distance: Distance from the plane that still counts, also
                allowed beyond both ends of the stretch.

        Returns:
            ``(N,)`` mask.
        """
        xy = points[:, :2]
        position = xy @ self.along
        return (
            (np.abs(xy @ self.normal - self.offset) <= distance)
            & (position >= self.low - distance)
            & (position <= self.high + distance)
        )


def _downsample(
    points: NDArray[np.float64], voxel_size: float
) -> NDArray[np.float64]:
    """Keeps the first point of every occupied voxel.

    Args:
        points: ``(N, 3)`` points.
        voxel_size: Voxel edge.

    Returns:
        The kept points in their original order.
    """
    if len(points) == 0:
        return points
    _, first = np.unique(
        pack_cells(voxel_cells(points, voxel_size)), return_index=True
    )
    return points[np.sort(first)]


def _is_wall(
    position: NDArray[np.float64],
    height: NDArray[np.float64],
    voxel_size: float,
    config: WallConfig,
) -> bool:
    """Tells whether the inliers of a plane form a continuous wall.

    Args:
        position: Position of each inlier along the plane.
        height: ``z`` of each inlier.
        voxel_size: Edge of the cells that measure the fill.
        config: Wall size and fill thresholds.

    Returns:
        True when the inliers span the minimum width and height and fill
        enough of the cells of their rectangle.
    """
    width = np.ptp(position)
    tall = np.ptp(height)
    if width < config.min_width or tall < config.min_height:
        return False
    cells = np.floor(
        np.stack([position - position.min(), height - height.min()], axis=1)
        / voxel_size
    ).astype(np.int64)
    keys = np.ravel_multi_index(cells.T, tuple(cells.max(axis=0) + 1))
    occupied = len(np.unique(keys))
    rectangle = (int(width / voxel_size) + 1) * (int(tall / voxel_size) + 1)
    return occupied >= config.min_fill * rectangle


def _wall_on(
    points: NDArray[np.float64],
    normal: NDArray[np.float64],
    offset: float,
    along: NDArray[np.float64],
    voxel_size: float,
    config: WallConfig,
) -> tuple[Wall, NDArray[np.bool_]] | None:
    """Finds the wall on a vertical plane.

    The points within ``plane_distance`` of the plane are split along it at
    gaps larger than ``max_gap``, and the run with the most points is the
    candidate, so furniture in line with a wall stays apart from it.

    Args:
        points: ``(N, 3)`` points.
        normal: ``(2,)`` unit normal of the plane in the xy plane.
        offset: Plane position along the normal.
        along: ``(2,)`` unit direction of the plane.
        voxel_size: Edge of the cells that measure the fill.
        config: Distance, gap, and wall thresholds.

    Returns:
        The wall and the mask of its points, or None when the candidate is
        not a wall (see ``_is_wall``).
    """
    near = np.flatnonzero(
        np.abs(points[:, :2] @ normal - offset) <= config.plane_distance
    )
    if len(near) == 0:
        return None
    position = points[near, :2] @ along
    order = np.argsort(position, kind="stable")
    breaks = np.flatnonzero(np.diff(position[order]) > config.max_gap) + 1
    run = max(np.split(order, breaks), key=len)
    members = near[run]
    if not _is_wall(position[run], points[members, 2], voxel_size, config):
        return None
    mask = np.zeros(len(points), np.bool_)
    mask[members] = True
    wall = Wall(
        normal,
        offset,
        along,
        float(position[run].min()),
        float(position[run].max()),
    )
    return wall, mask


def _refine(
    points: NDArray[np.float64],
    found: tuple[Wall, NDArray[np.bool_]],
    voxel_size: float,
    config: WallConfig,
) -> tuple[Wall, NDArray[np.bool_]]:
    """Refits a wall plane to its points.

    The plane through two sampled points tilts toward whatever else is near
    it, so the plane is refit to the wall's points by least squares in the
    xy plane and the wall is found again, while it keeps growing.

    Args:
        points: The sampled points the wall was found in.
        found: The wall of the best sampled pair and its points.
        voxel_size: Edge of the cells that measure the fill.
        config: Distance, gap, and wall thresholds.

    Returns:
        The refit wall and its points, or the input when the refit does not
        grow it.
    """
    best = found
    for _ in range(_REFIT_ROUNDS):
        xy = points[best[1], :2]
        center = xy.mean(axis=0)
        along = np.linalg.svd(xy - center, full_matrices=False)[2][0]
        normal = np.array([-along[1], along[0]])
        grown = _wall_on(
            points, normal, float(normal @ center), along, voxel_size, config
        )
        if grown is None or grown[1].sum() <= best[1].sum():
            break
        best = grown
    return best


def find_walls(
    points: NDArray[np.float64], voxel_size: float, config: WallConfig
) -> list[Wall]:
    """Finds the vertical walls with a seeded RANSAC.

    Each round samples pairs of points, each pair defining a vertical plane
    through both. The plane with the most points in a continuous wall
    becomes a wall and its points leave the sample, and the next round
    starts, until ``max_planes`` walls are found or no plane qualifies.

    Args:
        points: ``(N, 3)`` points of the static scene, z up.
        voxel_size: Edge of the voxels that thin the points and measure the
            fill of a wall.
        config: Plane and wall thresholds.

    Returns:
        The walls, in the order they were found.
    """
    rng = np.random.default_rng(config.seed)
    remaining = _downsample(points, voxel_size)
    walls: list[Wall] = []
    while len(walls) < config.max_planes and len(remaining) >= 2:
        pairs = rng.integers(
            0, len(remaining), size=(config.ransac_iterations, 2)
        )
        best: tuple[Wall, NDArray[np.bool_]] | None = None
        for first, second in pairs:
            step = remaining[second, :2] - remaining[first, :2]
            length = float(np.linalg.norm(step))
            if length < _WALL_PAIR_SHARE * config.min_width:
                continue
            along = step / length
            normal = np.array([-along[1], along[0]])
            offset = float(normal @ remaining[first, :2])
            found = _wall_on(
                remaining, normal, offset, along, voxel_size, config
            )
            if found is not None and (
                best is None or found[1].sum() > best[1].sum()
            ):
                best = found
        if best is None:
            break
        wall, members = _refine(remaining, best, voxel_size, config)
        walls.append(wall)
        remaining = remaining[~members]
    return walls
