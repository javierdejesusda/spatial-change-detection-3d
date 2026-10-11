"""Rotation matrix helpers shared by the tests.

Plain functions rather than fixtures, so Hypothesis tests can call them.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3


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
