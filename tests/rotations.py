"""Rotation matrix helpers shared by the tests.

Plain functions rather than fixtures, so Hypothesis tests can call them.
"""

import numpy as np
from numpy.typing import NDArray


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
