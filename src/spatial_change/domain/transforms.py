"""Rigid (SE3) and similarity (Sim3) transforms between 3D frames.

Conventions:
    * Transforms act on row-stacked points of shape ``(N, 3)``, float64.
    * A pose named ``world_from_camera`` maps camera coordinates to world
      coordinates: ``p_w = R @ p_c + t``. Its translation is therefore the
      camera center in world coordinates.
    * ``a.compose(b)`` is the matrix product ``a @ b``: it applies ``b``
      first, then ``a``.
    * Units are meters when metric scale is known, otherwise arbitrary but
      consistent within one reconstruction.

Both types are frozen, and their arrays are private read-only copies, so a
transform can be shared freely without defensive copying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

ROTATION_ATOL = 1e-6


def as_frozen_array(
    value: ArrayLike, shape: tuple[int, ...], name: str
) -> NDArray[np.float64]:
    """Returns a finite, read-only float64 copy of ``value``.

    Args:
        value: Array-like input.
        shape: Required shape.
        name: Field name used in error messages.

    Returns:
        A new read-only array with the requested shape.

    Raises:
        ValueError: If the shape differs or any entry is not finite.
    """
    array = np.array(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    return array


def check_array(
    array: NDArray[np.generic],
    shape: tuple[int | None, ...],
    dtype: type[np.generic],
    name: str,
) -> None:
    """Checks that ``array`` has the required shape and dtype.

    Args:
        array: Array to check. It is not modified or copied.
        shape: Required shape; ``None`` accepts any size along that axis.
        dtype: Required numpy scalar type.
        name: Field name used in error messages.

    Raises:
        ValueError: If the number of axes, a fixed axis size, or the dtype
            differs from the requirement.
    """
    shape_ok = array.ndim == len(shape) and all(
        want is None or want == got
        for want, got in zip(shape, array.shape, strict=True)
    )
    if not shape_ok or array.dtype != dtype:
        dims = ["N" if dim is None else str(dim) for dim in shape]
        expected = f"({dims[0]},)" if len(dims) == 1 else f"({', '.join(dims)})"
        raise ValueError(
            f"{name} must be {np.dtype(dtype)} with shape {expected}, "
            f"got {array.dtype} with shape {array.shape}"
        )


def as_rotation(
    value: ArrayLike, name: str = "rotation"
) -> NDArray[np.float64]:
    """Returns a read-only copy of ``value`` checked to be a proper rotation.

    Args:
        value: A 3x3 array-like.
        name: Field name used in error messages.

    Returns:
        A read-only 3x3 float64 rotation matrix.

    Raises:
        ValueError: If ``value`` is not orthonormal with determinant +1
            within ``ROTATION_ATOL``.
    """
    rotation = as_frozen_array(value, (3, 3), name)
    orthonormal = np.allclose(
        rotation @ rotation.T, np.eye(3), rtol=0.0, atol=ROTATION_ATOL
    )
    if not orthonormal or abs(np.linalg.det(rotation) - 1.0) > ROTATION_ATOL:
        raise ValueError(f"{name} must be orthonormal with determinant +1")
    return rotation


def rotation_to_quaternion(
    rotation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Converts a rotation matrix to a unit quaternion.

    Args:
        rotation: Proper rotation of shape ``(3, 3)``.

    Returns:
        The unit quaternion ``(x, y, z, w)``.
    """
    r = rotation
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = (
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
            s / 4.0,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = (
            s / 4.0,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[2, 1] - r[1, 2]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = (
            (r[0, 1] + r[1, 0]) / s,
            s / 4.0,
            (r[1, 2] + r[2, 1]) / s,
            (r[0, 2] - r[2, 0]) / s,
        )
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = (
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            s / 4.0,
            (r[1, 0] - r[0, 1]) / s,
        )
    return np.array(q)


def as_points(points: ArrayLike, name: str = "points") -> NDArray[np.float64]:
    """Returns ``points`` as a float64 ``(N, 3)`` array, copying only if needed.

    Args:
        points: Array-like of row-stacked 3D points.
        name: Field name used in error messages.

    Returns:
        The points as a float64 array.

    Raises:
        ValueError: If ``points`` does not have shape ``(N, 3)``.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    return array


def as_finite_points(
    points: ArrayLike, name: str, min_count: int
) -> NDArray[np.float64]:
    """Returns ``points`` as a finite float64 ``(N, 3)`` array of enough rows.

    Args:
        points: Array-like of row-stacked 3D points.
        name: Field name used in error messages.
        min_count: Smallest accepted number of points.

    Returns:
        The points as a float64 array.

    Raises:
        ValueError: If ``points`` does not have shape ``(N, 3)``, has fewer
            than ``min_count`` rows, or holds a non-finite value.
    """
    array = as_points(points, name)
    if len(array) < min_count:
        raise ValueError(
            f"{name} needs at least {min_count} points, got {len(array)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _split_affine(
    m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = as_frozen_array(m, (4, 4), "matrix")
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=ROTATION_ATOL
    ):
        raise ValueError("matrix bottom row must be [0, 0, 0, 1]")
    return matrix[:3, :3], matrix[:3, 3]


@dataclass(frozen=True, eq=False)
class SE3:
    """A rigid transform ``p -> R @ p + t``.

    Attributes:
        rotation: 3x3 rotation matrix, orthonormal with determinant +1.
        translation: Translation vector of shape ``(3,)``.
    """

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Copies, validates, and freezes the arrays.

        Raises:
            ValueError: If the rotation is not a proper rotation or the
                translation is not a finite ``(3,)`` vector.
        """
        object.__setattr__(self, "rotation", as_rotation(self.rotation))
        object.__setattr__(
            self,
            "translation",
            as_frozen_array(self.translation, (3,), "translation"),
        )

    @classmethod
    def identity(cls) -> SE3:
        """Returns the identity transform."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, m: ArrayLike) -> SE3:
        """Builds a transform from a 4x4 homogeneous matrix.

        Args:
            m: Homogeneous matrix ``[[R, t], [0, 1]]``.

        Returns:
            The equivalent transform.

        Raises:
            ValueError: If ``m`` is not a 4x4 rigid homogeneous matrix.
        """
        rotation, translation = _split_affine(m)
        return cls(rotation, translation)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """The 4x4 homogeneous matrix ``[[R, t], [0, 1]]``."""
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def inverse(self) -> SE3:
        """Returns the inverse transform."""
        rotation_t = self.rotation.T
        return SE3(rotation_t, -rotation_t @ self.translation)

    def compose(self, other: SE3) -> SE3:
        """Returns ``self @ other``, which applies ``other`` first.

        Args:
            other: The transform applied before ``self``.

        Returns:
            The composed transform.
        """
        return SE3(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
        )

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transforms row-stacked points.

        Args:
            points: Points of shape ``(N, 3)``.

        Returns:
            Transformed points of shape ``(N, 3)``, float64.

        Raises:
            ValueError: If ``points`` does not have shape ``(N, 3)``.
        """
        out = as_points(points) @ self.rotation.T
        out += self.translation
        return out

    def apply_inverse(self, points: ArrayLike) -> NDArray[np.float64]:
        """Applies the inverse transform to row-stacked points.

        Equivalent to ``self.inverse().apply(points)``, i.e.
        ``R.T @ (p - t)`` per point, without building the inverse pose. For
        a ``world_from_camera`` pose this maps world points into the camera
        frame.

        Args:
            points: Points of shape ``(N, 3)``.

        Returns:
            Transformed points of shape ``(N, 3)``, float64.

        Raises:
            ValueError: If ``points`` does not have shape ``(N, 3)``.
        """
        out = as_points(points) @ self.rotation
        out -= self.translation @ self.rotation
        return out


@dataclass(frozen=True, eq=False)
class Sim3:
    """A similarity transform ``p -> s * R @ p + t``.

    Attributes:
        scale: Positive, finite uniform scale.
        rotation: 3x3 rotation matrix, orthonormal with determinant +1.
        translation: Translation vector of shape ``(3,)``.
    """

    scale: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Copies, validates, and freezes the fields.

        Raises:
            ValueError: If the scale is not positive and finite, the rotation
                is not a proper rotation, or the translation is not a finite
                ``(3,)`` vector.
        """
        scale = float(self.scale)
        if not (np.isfinite(scale) and scale > 0.0):
            raise ValueError(f"scale must be positive and finite, got {scale}")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rotation", as_rotation(self.rotation))
        object.__setattr__(
            self,
            "translation",
            as_frozen_array(self.translation, (3,), "translation"),
        )

    @classmethod
    def identity(cls) -> Sim3:
        """Returns the identity transform."""
        return cls(1.0, np.eye(3), np.zeros(3))

    @classmethod
    def from_se3(cls, se3: SE3) -> Sim3:
        """Returns the rigid transform ``se3`` as a similarity with scale 1."""
        return cls(1.0, se3.rotation, se3.translation)

    @classmethod
    def from_matrix(cls, m: ArrayLike) -> Sim3:
        """Builds a transform from a 4x4 homogeneous matrix.

        Args:
            m: Homogeneous matrix ``[[s * R, t], [0, 1]]``.

        Returns:
            The equivalent transform.

        Raises:
            ValueError: If ``m`` is not a 4x4 similarity homogeneous matrix.
        """
        linear, translation = _split_affine(m)
        det = np.linalg.det(linear)
        if det <= 0.0:
            raise ValueError("matrix must have a positive determinant")
        scale = float(np.cbrt(det))
        return cls(scale, linear / scale, translation)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """The 4x4 homogeneous matrix ``[[s * R, t], [0, 1]]``."""
        m = np.eye(4)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m

    def inverse(self) -> Sim3:
        """Returns the inverse transform."""
        rotation_t = self.rotation.T
        return Sim3(
            1.0 / self.scale,
            rotation_t,
            -(rotation_t @ self.translation) / self.scale,
        )

    def compose(self, other: Sim3) -> Sim3:
        """Returns ``self @ other``, which applies ``other`` first.

        Args:
            other: The transform applied before ``self``.

        Returns:
            The composed transform.
        """
        return Sim3(
            self.scale * other.scale,
            self.rotation @ other.rotation,
            self.scale * self.rotation @ other.translation + self.translation,
        )

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transforms row-stacked points.

        Args:
            points: Points of shape ``(N, 3)``.

        Returns:
            Transformed points of shape ``(N, 3)``, float64.

        Raises:
            ValueError: If ``points`` does not have shape ``(N, 3)``.
        """
        linear = self.scale * self.rotation
        out = as_points(points) @ linear.T
        out += self.translation
        return out

    def transform_pose(self, world_from_camera: SE3) -> SE3:
        """Re-expresses a camera pose in the world frame this map leads to.

        The camera center moves with the full similarity and the orientation
        is rotated by ``R``. Scale changes only the translation, so the
        result stays a rigid pose. Camera-frame quantities such as depth are
        then in the new world's units only after multiplying them by
        ``scale``.

        Args:
            world_from_camera: Camera-to-world pose in the source world.

        Returns:
            The camera-to-world pose in the target world.
        """
        moved = self.compose(Sim3.from_se3(world_from_camera))
        return SE3(moved.rotation, moved.translation)
