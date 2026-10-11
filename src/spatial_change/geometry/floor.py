"""Floor plane estimation and the gravity-aligned room frame.

A reconstruction's world frame has an arbitrary origin and orientation.
The floor is found as a near-horizontal plane (relative to an up direction
taken from the cameras) and turned into a room frame that is z-up with the
floor at ``z = 0``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3
from spatial_change.domain.transforms import (
    as_finite_points,
    as_frozen_array,
    as_points,
)

_MIN_NORM = 1e-9
# Smallest sine of the angle between the two edges of a hypothesis triple.
# It is relative, so the test does not depend on the scale of the cloud.
_MIN_EDGE_SIN = 1e-6
# Hypotheses are built from nearby points: a random triple from the whole
# cloud lies on one plane with probability (plane share)^3, which is hopeless
# for a floor that is a small part of the scene. The pool bounds the cost of
# the neighbour search.
_POOL_SIZE = 2000
_NEIGHBOURS = 100
# Hypotheses are scored on a random subsample of at most this many points,
# in column chunks whose float64 temporaries stay under the byte budget.
_SCORE_SAMPLE = 50_000
_SCORE_BUDGET_BYTES = 16 * 2**20
# sin of the smallest angle between the forward hint and the floor normal
# that still defines a heading.
_MIN_HEADING_SIN = 1e-3


@dataclass(frozen=True)
class FloorConfig:
    """Parameters of the floor plane search.

    Attributes:
        distance_threshold: Inlier distance to a plane, in meters of the
            room. It is the only length in this config, so a cloud that is
            not in meters needs it scaled by the same factor.
        max_tilt_deg: Largest angle between a candidate normal and the up
            hint, in degrees.
        iterations: RANSAC hypotheses scored per extracted plane.
        max_planes: Number of horizontal planes extracted sequentially.
        min_support_fraction: Fraction of all points a plane needs to be a
            floor candidate.
        seed: Seed of the hypothesis sampler.
    """

    distance_threshold: float = 0.02
    max_tilt_deg: float = 25.0
    iterations: int = 500
    max_planes: int = 4
    min_support_fraction: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a parameter is outside its valid range.
        """
        if not self.distance_threshold > 0.0:
            raise ValueError("distance_threshold must be positive")
        if not 0.0 < self.max_tilt_deg < 90.0:
            raise ValueError("max_tilt_deg must be in (0, 90)")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.max_planes < 1:
            raise ValueError("max_planes must be at least 1")
        if not 0.0 < self.min_support_fraction <= 1.0:
            raise ValueError("min_support_fraction must be in (0, 1]")


@dataclass(frozen=True, eq=False)
class FloorPlane:
    """A plane whose height function is ``normal @ p + offset``.

    Attributes:
        normal: Unit normal pointing up (toward the up hint).
        offset: Plane offset; the height of ``p`` above the plane is
            ``normal @ p + offset``.
        inlier_count: Number of points within the distance threshold.
    """

    normal: NDArray[np.float64]
    offset: float
    inlier_count: int

    def height(self, points: ArrayLike) -> NDArray[np.float64]:
        """Returns the signed height of points above the plane.

        Args:
            points: Points of shape ``(N, 3)``.

        Returns:
            Heights of shape ``(N,)``; positive on the up side.
        """
        return as_points(points) @ self.normal + self.offset


def camera_up_hint(poses: Sequence[SE3]) -> NDArray[np.float64]:
    """Returns the mean up direction of cameras in the world frame.

    Cameras follow the OpenCV convention (y down), so a camera's up vector
    is ``-R[:, 1]`` of its ``world_from_camera`` rotation.

    Args:
        poses: Camera-to-world poses.

    Returns:
        A unit vector of shape ``(3,)``.

    Raises:
        ValueError: If there are no poses or the mean up vector is (nearly)
            zero.
    """
    if len(poses) == 0:
        raise ValueError("need at least one camera pose")
    mean = -np.mean([pose.rotation[:, 1] for pose in poses], axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        raise ValueError("camera up vectors cancel out to zero")
    return mean / norm


def _unit(vector: ArrayLike, name: str) -> NDArray[np.float64]:
    array = as_frozen_array(vector, (3,), name)
    norm = float(np.linalg.norm(array))
    if norm < _MIN_NORM:
        raise ValueError(f"{name} must be nonzero")
    return array / norm


def _hypotheses(
    pool: NDArray[np.float64],
    up: NDArray[np.float64],
    config: FloorConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Builds horizontal plane hypotheses from local point triples."""
    m = len(pool)
    anchors = rng.integers(0, m, config.iterations)
    k = min(_NEIGHBOURS, m - 1)
    sq = ((pool[anchors, None, :] - pool[None, :, :]) ** 2).sum(axis=2)
    sq[np.arange(config.iterations), anchors] = np.inf
    near = np.argpartition(sq, k - 1, axis=1)[:, :k]
    picks = rng.random((config.iterations, 2)) * k
    b = near[np.arange(config.iterations), picks[:, 0].astype(int)]
    c = near[np.arange(config.iterations), picks[:, 1].astype(int)]
    a_pts = pool[anchors]
    edge_b = pool[b] - a_pts
    edge_c = pool[c] - a_pts
    normals = np.cross(edge_b, edge_c)
    norms = np.linalg.norm(normals, axis=1)
    lengths = np.linalg.norm(edge_b, axis=1) * np.linalg.norm(edge_c, axis=1)
    sin = np.divide(norms, lengths, out=np.zeros_like(norms), where=lengths > 0)
    normals = np.divide(
        normals,
        norms[:, None],
        out=np.zeros_like(normals),
        where=norms[:, None] > 0,
    )
    normals *= np.where(normals @ up < 0.0, -1.0, 1.0)[:, None]
    ok = (sin > _MIN_EDGE_SIN) & (
        normals @ up >= np.cos(np.radians(config.max_tilt_deg))
    )
    normals = normals[ok]
    offsets = -np.einsum("ij,ij->i", normals, a_pts[ok])
    return normals, offsets


def _best_plane(
    points: NDArray[np.float64],
    up: NDArray[np.float64],
    config: FloorConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], float] | None:
    """Returns the hypothesis with the most inliers on a subsample, or None."""
    pool = points
    if len(points) > _POOL_SIZE:
        pool = points[rng.choice(len(points), _POOL_SIZE, replace=False)]
    normals, offsets = _hypotheses(pool, up, config, rng)
    sample = points
    if len(points) > _SCORE_SAMPLE:
        sample = points[rng.choice(len(points), _SCORE_SAMPLE, replace=False)]
    chunk = max(1, _SCORE_BUDGET_BYTES // (8 * len(sample)))
    best_count, best = -1, None
    for start in range(0, len(normals), chunk):
        n_chunk = normals[start : start + chunk]
        o_chunk = offsets[start : start + chunk]
        dist = sample @ n_chunk.T
        dist += o_chunk
        np.abs(dist, out=dist)
        counts = np.count_nonzero(dist < config.distance_threshold, axis=0)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best_count, best = int(counts[i]), (n_chunk[i], float(o_chunk[i]))
    return best


def _refine(
    inliers: NDArray[np.float64], up: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Least-squares plane through ``inliers`` with an up-facing normal."""
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[2]
    if normal @ up < 0.0:
        normal = -normal
    return normal, float(-normal @ centroid)


_DEFAULT_CONFIG = FloorConfig()


def fit_floor(
    points: ArrayLike,
    up_hint: ArrayLike,
    config: FloorConfig = _DEFAULT_CONFIG,
) -> FloorPlane:
    """Finds the floor as the lowest well-supported horizontal plane.

    Sequential RANSAC extracts up to ``config.max_planes`` planes whose
    normals lie within ``config.max_tilt_deg`` of the up hint (so walls are
    never candidates), removing each plane's inliers before the next. Among
    planes supported by at least ``config.min_support_fraction`` of all
    points, the lowest along the up hint wins, so a large table top or a
    ceiling never beats a smaller visible floor. The winner is refined by a
    least-squares fit on its inliers.

    Hypotheses come from triples of nearby points, which keeps small planes
    findable; the sampler is seeded, so the result is deterministic.

    Args:
        points: Scene points of shape ``(N, 3)``.
        up_hint: Approximate up direction, for example from
            ``camera_up_hint``.
        config: Search parameters.

    Returns:
        The floor plane with an up-facing unit normal.

    Raises:
        ValueError: If the inputs are invalid or no horizontal plane has
            enough support.
    """
    cloud = as_finite_points(points, "points", 3)
    up = _unit(up_hint, "up_hint")

    rng = np.random.default_rng(config.seed)
    min_support = max(3, int(np.ceil(config.min_support_fraction * len(cloud))))
    remaining = np.ones(len(cloud), dtype=np.bool_)
    candidates: list[tuple[float, NDArray[np.float64], float, int]] = []
    for _ in range(config.max_planes):
        active = cloud[remaining]
        if len(active) < min_support:
            break
        best = _best_plane(active, up, config, rng)
        if best is None:
            continue
        normal, offset = best
        near = np.abs(active @ normal + offset) < config.distance_threshold
        if near.sum() < min_support:
            break
        normal, offset = _refine(active[near], up)
        inside = np.abs(cloud @ normal + offset) < config.distance_threshold
        inside &= remaining
        if inside.sum() >= min_support:
            level = float(cloud[inside].mean(axis=0) @ up)
            candidates.append((level, normal, offset, int(inside.sum())))
        remaining &= ~inside
    if not candidates:
        raise ValueError("no horizontal plane with enough support")
    _, normal, offset, count = min(candidates, key=lambda c: c[0])
    return FloorPlane(normal, offset, count)


def floor_frame(floor: FloorPlane, forward_hint: ArrayLike) -> SE3:
    """Builds the z-up room frame with the floor at ``z = 0``.

    The room's ``+z`` is the floor normal and ``+y`` is the forward hint
    projected onto the floor, so ``+x = y x z`` points to the right of a
    viewer looking along ``+y``. The xy origin is the projection of the
    world origin onto the floor. If the hint is (nearly) parallel to the
    normal, the world x axis is used, then the world y axis.

    Args:
        floor: The fitted floor plane.
        forward_hint: World direction that should become ``+y``, for
            example the first camera's viewing direction ``R[:, 2]``.

    Returns:
        The rigid transform that maps world points into the room frame.

    Raises:
        ValueError: If ``forward_hint`` is not a finite 3-vector.
    """
    z = floor.normal
    hint = as_frozen_array(forward_hint, (3,), "forward_hint")
    y = None
    for candidate in (
        hint,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ):
        flat = candidate - (candidate @ z) * z
        scale = np.linalg.norm(candidate)
        if scale > 0.0 and np.linalg.norm(flat) > _MIN_HEADING_SIN * scale:
            y = flat / np.linalg.norm(flat)
            break
    assert y is not None  # x and y axes cannot both be parallel to z
    x = np.cross(y, z)
    rotation = np.stack([x, y, z])
    origin = -floor.offset * z
    return SE3(rotation, -rotation @ origin)
