"""Virtual handheld camera paths and frustum tests.

Poses are ``world_from_camera`` in the OpenCV convention: x right, y down,
z forward, with world z up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3, Intrinsics
from spatial_change.geometry.projection import in_image, project

_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, eq=False)
class Waypoint:
    """A camera position and the world point it looks at.

    Attributes:
        position: Camera center in world coordinates, shape ``(3,)``.
        target: World point on the optical axis, shape ``(3,)``.
    """

    position: NDArray[np.float64]
    target: NDArray[np.float64]


def look_at(position: ArrayLike, target: ArrayLike) -> SE3:
    """Builds a roll-free pose whose optical axis points at a target.

    Args:
        position: Camera center in world coordinates.
        target: World point to look at; must not lie straight above or
            below the camera.

    Returns:
        The ``world_from_camera`` pose.

    Raises:
        ValueError: If ``position`` and ``target`` coincide or the view
            direction is vertical.
    """
    eye = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    length = np.linalg.norm(forward)
    if length < 1e-9:
        raise ValueError("position and target must differ")
    forward /= length
    right = np.cross(forward, _UP)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        raise ValueError("view direction must not be vertical")
    right /= norm
    down = np.cross(forward, right)
    return SE3(np.stack([right, down, forward], axis=1), eye)


def sweep(
    waypoints: Sequence[Waypoint],
    n_frames: int,
    rng: np.random.Generator,
    position_jitter_m: float,
    target_jitter_m: float,
) -> list[SE3]:
    """Interpolates a smooth path through waypoints with seeded jitter.

    Frames are spread evenly over the path and eased with a smoothstep in
    each segment. Gaussian noise on the camera center and on the look-at
    target imitates hand shake.

    Args:
        waypoints: At least two waypoints, in travel order.
        n_frames: Number of poses to produce, at least 2.
        rng: Source of the jitter.
        position_jitter_m: Standard deviation of the camera-center noise.
        target_jitter_m: Standard deviation of the look-at target noise.

    Returns:
        ``n_frames`` camera poses.

    Raises:
        ValueError: If there are fewer than two waypoints or two frames.
    """
    if len(waypoints) < 2 or n_frames < 2:
        raise ValueError("need at least two waypoints and two frames")
    positions = np.stack([w.position for w in waypoints])
    targets = np.stack([w.target for w in waypoints])
    s = np.linspace(0.0, len(waypoints) - 1.0, n_frames)
    segment = np.minimum(np.floor(s).astype(int), len(waypoints) - 2)
    t = (s - segment)[:, None]
    t = t * t * (3.0 - 2.0 * t)
    eye = positions[segment] * (1 - t) + positions[segment + 1] * t
    aim = targets[segment] * (1 - t) + targets[segment + 1] * t
    eye = eye + rng.normal(scale=position_jitter_m, size=eye.shape)
    aim = aim + rng.normal(scale=target_jitter_m, size=aim.shape)
    return [look_at(p, q) for p, q in zip(eye, aim, strict=True)]


def sees_points(
    points: ArrayLike, intrinsics: Intrinsics, world_from_camera: SE3
) -> NDArray[np.bool_]:
    """Tests which points fall inside a camera frustum, ignoring occlusion.

    Args:
        points: World points of shape ``(N, 3)``.
        intrinsics: Camera intrinsics.
        world_from_camera: Camera-to-world pose.

    Returns:
        Boolean mask: the point is in front of the camera and projects
        inside the image.
    """
    uv, depth = project(points, intrinsics, world_from_camera)
    return in_image(uv, depth, intrinsics)
