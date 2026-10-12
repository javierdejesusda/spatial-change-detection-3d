"""Rotation matrix helpers shared by the tests.

Plain functions rather than fixtures, so Hypothesis tests can call them.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3, Sim3
from spatial_change.synthetic.trajectory import look_at


def random_rotation(rng: np.random.Generator) -> NDArray[np.float64]:
    """Draws a random proper rotation matrix.

    Args:
        rng: Source of randomness.

    Returns:
        A 3x3 orthonormal matrix with determinant +1.
    """
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotation_z(deg: float) -> NDArray[np.float64]:
    """Builds a counterclockwise rotation about the z axis.

    Args:
        deg: Rotation angle in degrees.

    Returns:
        A 3x3 rotation matrix.
    """
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def perturbation(axis: ArrayLike, deg: float, translation: ArrayLike) -> SE3:
    """Builds a rigid transform from an axis-angle rotation and a shift.

    Args:
        axis: Rotation axis of shape ``(3,)``; it is normalized.
        deg: Rotation angle in degrees.
        translation: Translation of shape ``(3,)``.

    Returns:
        The transform ``p -> R p + translation``.
    """
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    angle = np.radians(deg)
    k = np.array(
        [[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]]
    )
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    return SE3(rotation, np.asarray(translation, dtype=np.float64))


def rotation_angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Measures the angle of the rotation between two rotation matrices.

    Args:
        a: A 3x3 rotation matrix.
        b: A 3x3 rotation matrix.

    Returns:
        The angle of ``a.T @ b`` in degrees, in ``[0, 180]``.
    """
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Measures the angle between two vectors.

    Args:
        a: A nonzero vector.
        b: A nonzero vector of the same length.

    Returns:
        The angle in degrees, in ``[0, 180]``.
    """
    cos = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pitched_cameras(
    pitch_deg: float,
    yaws_deg: ArrayLike,
    center: tuple[float, float, float] = (0.0, 0.0, 1.5),
    spread: float = 0.0,
    world_from_room: Sim3 | None = None,
) -> list[SE3]:
    """Builds roll-free cameras pitched down, one per yaw.

    The optical axis is ``(cos p cos y, cos p sin y, -sin p)`` in the z-up
    room frame.

    Args:
        pitch_deg: Downward pitch of every camera in degrees.
        yaws_deg: Yaw of each camera in degrees.
        center: Camera center in the room frame.
        spread: Distance by which each center moves along its yaw direction.
        world_from_room: Optional map of the poses into another world frame.

    Returns:
        One ``world_from_camera`` pose per yaw.
    """
    p = np.radians(pitch_deg)
    poses = []
    for yaw in np.radians(np.asarray(yaws_deg, dtype=np.float64)):
        forward = np.array(
            [np.cos(p) * np.cos(yaw), np.cos(p) * np.sin(yaw), -np.sin(p)]
        )
        eye = np.array(center) + spread * np.array(
            [np.cos(yaw), np.sin(yaw), 0]
        )
        pose = look_at(eye, eye + forward)
        poses.append(
            pose
            if world_from_room is None
            else world_from_room.transform_pose(pose)
        )
    return poses


def pose_error(estimate: SE3, truth: SE3) -> tuple[float, float]:
    """Measures the residual between two rigid transforms.

    Args:
        estimate: The estimated transform.
        truth: The reference transform.

    Returns:
        The rotation error in degrees and the translation error in the
        transform's units.
    """
    return (
        rotation_angle_deg(estimate.rotation, truth.rotation),
        float(np.linalg.norm(truth.translation - estimate.translation)),
    )
