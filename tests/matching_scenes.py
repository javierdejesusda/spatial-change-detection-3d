"""Hand-built objects for the registration and matching tests.

Objects are sampled on the surface of upright boxes and of an L-shaped solid,
so the tests know the exact transform between two copies of one shape.
"""

import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.change import Candidate, MatchConfig
from spatial_change.domain import (
    Epoch,
    EvidenceCounts,
    ObjectInstance,
)
from spatial_change.geometry import fit_upright_box, yaw_rotation

_Vec3 = tuple[float, float, float]
SPACING = 0.02
CHAIR = (0.5, 0.4, 0.9)
NOISE = 0.002
EVIDENCE = EvidenceCounts(consistent=1, free=20, occluded=0, unobserved=3)
# The default gates turned off, so that a test sees the changes as they are
# made and not as the gates demote them.
UNGATED_GATES: dict[str, Any] = {
    "max_occluded_fraction": None,
    "max_box_volume_m3": None,
}
UNGATED = MatchConfig(**UNGATED_GATES)


def _face_points(
    low: NDArray[np.float64], high: NDArray[np.float64], spacing: float
) -> list[NDArray[np.float64]]:
    """Samples the six faces of an axis-aligned box on cell centers."""
    faces = []
    for axis in range(3):
        a, b = (k for k in range(3) if k != axis)
        grid_a = low[a] + spacing * (
            np.arange(max(1, round((high[a] - low[a]) / spacing))) + 0.5
        )
        grid_b = low[b] + spacing * (
            np.arange(max(1, round((high[b] - low[b]) / spacing))) + 0.5
        )
        grid = np.array(list(itertools.product(grid_a, grid_b)))
        for side in (low[axis], high[axis]):
            face = np.empty((len(grid), 3))
            face[:, axis] = side
            face[:, a] = grid[:, 0]
            face[:, b] = grid[:, 1]
            faces.append(face)
    return faces


def _inside(
    points: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Tells which points lie in a box, its boundary included."""
    return np.all((points >= low - 1e-6) & (points <= high + 1e-6), axis=1)


def _place(
    local: NDArray[np.float64],
    center: _Vec3,
    yaw_deg: float,
    seed: int,
    noise: float,
) -> NDArray[np.float64]:
    """Rotates about z, moves to ``center``, and adds Gaussian noise."""
    rotation = yaw_rotation(np.radians(yaw_deg))
    points = local @ rotation.T + np.asarray(center)
    if noise > 0.0:
        points = points + np.random.default_rng(seed).normal(
            scale=noise, size=points.shape
        )
    return points


def box_surface(
    center: _Vec3,
    extent: _Vec3,
    yaw_deg: float = 0.0,
    seed: int = 0,
    spacing: float = SPACING,
    noise: float = NOISE,
) -> NDArray[np.float64]:
    """Samples the surface of a box turned about its vertical axis.

    Args:
        center: Box center.
        extent: Full side lengths along the box axes.
        yaw_deg: Turn about the vertical axis through the center.
        seed: Seed of the noise.
        spacing: Distance between samples.
        noise: Standard deviation of the Gaussian noise.

    Returns:
        Points of shape ``(N, 3)``.
    """
    half = np.asarray(extent) / 2.0
    local = np.concatenate(_face_points(-half, half, spacing))
    return _place(local, center, yaw_deg, seed, noise)


def chair(
    x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0, seed: int = 1
) -> NDArray[np.float64]:
    """Samples a chair-sized box standing on the floor.

    Args:
        x: Box center along x.
        y: Box center along y.
        yaw_deg: Turn about the vertical axis through the center.
        seed: Seed of the noise.

    Returns:
        Points of shape ``(N, 3)``.
    """
    return box_surface((x, y, CHAIR[2] / 2.0), CHAIR, yaw_deg, seed)


def chair_pair(
    dx: float = 0.0, dy: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Samples a chair at the origin and again after a move.

    Args:
        dx: Move along x.
        dy: Move along y.

    Returns:
        The points before and after, each with its own noise.
    """
    return chair(seed=1), chair(dx, dy, seed=2)


def l_surface(
    center: _Vec3,
    yaw_deg: float = 0.0,
    height: float = 0.4,
    seed: int = 0,
    spacing: float = SPACING,
    noise: float = NOISE,
) -> NDArray[np.float64]:
    """Samples the surface of an L-shaped solid.

    The L fits a 0.6 x 0.6 m footprint centered on ``center``, with arms 0.2
    m wide, so no turn about the vertical axis maps it to itself.

    Args:
        center: Center of the bounding box.
        yaw_deg: Turn about the vertical axis through the center.
        height: Height of the solid.
        seed: Seed of the noise.
        spacing: Distance between samples.
        noise: Standard deviation of the Gaussian noise.

    Returns:
        Points of shape ``(N, 3)``.
    """
    z = (-height / 2.0, height / 2.0)
    arms = [
        (np.array([-0.3, -0.3, z[0]]), np.array([0.3, -0.1, z[1]])),
        (np.array([-0.3, -0.1, z[0]]), np.array([-0.1, 0.3, z[1]])),
    ]
    parts = []
    for index, (low, high) in enumerate(arms):
        other_low, other_high = arms[1 - index]
        points = np.concatenate(_face_points(low, high, spacing))
        parts.append(points[~_inside(points, other_low, other_high)])
    return _place(np.concatenate(parts), center, yaw_deg, seed, noise)


def instance(
    points: NDArray[np.float64],
    object_id: str,
    epoch: Epoch,
    label: str | None = None,
    feature: NDArray[np.float32] | None = None,
    label_score: float | None = None,
) -> ObjectInstance:
    """Wraps points in an instance with their upright box.

    Args:
        points: Object points.
        object_id: Identifier.
        epoch: Epoch of the instance.
        label: Optional class name.
        feature: Optional unit appearance feature.
        label_score: Optional score of the label.

    Returns:
        The instance.
    """
    return ObjectInstance(
        object_id=object_id,
        epoch=epoch,
        points=points,
        box=fit_upright_box(points, 0.05),
        label=label,
        label_score=label_score,
        feature=feature,
    )


def unit(*values: float) -> NDArray[np.float32]:
    """Builds an L2-normalized feature vector."""
    vector = np.array(values, np.float32)
    return vector / np.linalg.norm(vector)


def candidate_for(
    source: ObjectInstance,
    evidence: EvidenceCounts = EVIDENCE,
    changed_fraction: float = 0.9,
) -> Candidate:
    """Builds a candidate whose points are those of an instance.

    Args:
        source: The instance to mirror.
        evidence: Vote counts of the candidate.
        changed_fraction: Its changed fraction.

    Returns:
        A candidate with the instance's points and box.
    """
    return Candidate(
        candidate_id=source.object_id,
        epoch=source.epoch,
        point_indices=np.arange(len(source.points), dtype=np.int64),
        points=source.points,
        box=source.box,
        evidence=evidence,
        voxel_count=1,
        changed_fraction=changed_fraction,
    )
