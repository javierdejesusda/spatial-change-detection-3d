"""Measures of a rotation matrix."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain.transforms import check_array


def rotation_angle_rad(rotation: NDArray[np.float64]) -> float:
    """Measures the angle of a rotation matrix without checking it.

    The angle comes from ``arctan2`` of the skew and the trace parts, which
    keeps its precision near 0 and near 180 degrees where ``arccos`` of the
    trace does not. A matrix that is not a finite 3x3 array gives garbage,
    and a NaN gives NaN.

    Args:
        rotation: A 3x3 rotation matrix.

    Returns:
        The angle about the rotation axis, in ``[0, pi]`` radians.
    """
    sin = np.linalg.norm(rotation - rotation.T) / (2.0 * np.sqrt(2.0))
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.arctan2(sin, cos))


def rotation_angle_deg(rotation: ArrayLike) -> float:
    """Measures the angle of a rotation.

    See ``rotation_angle_rad`` for how the angle is computed.

    Args:
        rotation: A 3x3 rotation matrix.

    Returns:
        The angle about the rotation axis, in ``[0, 180]`` degrees.

    Raises:
        ValueError: If ``rotation`` is not a finite 3x3 array.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    check_array(matrix, (3, 3), np.float64, "rotation")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be finite")
    return float(np.degrees(rotation_angle_rad(matrix)))
