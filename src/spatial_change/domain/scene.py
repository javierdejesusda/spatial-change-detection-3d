"""Reconstructed scenes, oriented boxes, and segmented object instances.

Conventions:
    * All geometry is in the world frame of its reconstruction. Units are
      meters when metric scale is known, otherwise arbitrary but consistent.
    * Point clouds are ``(N, 3)`` float64, colors ``(N, 3)`` uint8 RGB, and
      per-point confidences ``(N,)`` float32.
    * An ``OrientedBox`` rotation maps box-local axes to world axes, so a
      world point ``p`` has local coordinates ``R.T @ (p - center)``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain.camera import Epoch, Frame
from spatial_change.domain.transforms import (
    SE3,
    as_frozen_array,
    as_rotation,
    check_array,
)

_CORNER_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


@dataclass(eq=False)
class EpochReconstruction:
    """The posed frames and fused point cloud of one capture.

    Arrays are stored by reference, not copied, because they can be large.

    Attributes:
        epoch: Capture this reconstruction belongs to.
        frames: Posed frames, all from ``epoch``.
        points: World points of shape ``(N, 3)``, float64.
        colors: Point colors of shape ``(N, 3)``, uint8 RGB.
        confidence: Point confidences of shape ``(N,)``, float32.
    """

    epoch: Epoch
    frames: list[Frame]
    points: NDArray[np.float64]
    colors: NDArray[np.uint8]
    confidence: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Checks array shapes, dtypes, and frame epochs.

        Raises:
            ValueError: If an array has the wrong shape or dtype, the arrays
                disagree on N, or a frame belongs to another epoch.
        """
        check_array(self.points, (None, 3), np.float64, "points")
        check_array(self.colors, (None, 3), np.uint8, "colors")
        check_array(self.confidence, (None,), np.float32, "confidence")
        counts = {len(self.points), len(self.colors), len(self.confidence)}
        if len(counts) != 1:
            raise ValueError("points, colors, and confidence lengths differ")
        if any(frame.epoch != self.epoch for frame in self.frames):
            raise ValueError(f"all frames must belong to epoch {self.epoch}")


@dataclass(frozen=True, eq=False)
class OrientedBox:
    """A box with arbitrary orientation in 3D.

    Attributes:
        center: Box center of shape ``(3,)``.
        extent: Full side lengths along the box axes, shape ``(3,)``, all
            positive.
        rotation: 3x3 rotation whose columns are the box axes in world
            coordinates.
    """

    center: NDArray[np.float64]
    extent: NDArray[np.float64]
    rotation: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Copies, validates, and freezes the arrays.

        Raises:
            ValueError: If an array has the wrong shape or is not finite, an
                extent is not positive, or the rotation is not proper.
        """
        object.__setattr__(
            self, "center", as_frozen_array(self.center, (3,), "center")
        )
        extent = as_frozen_array(self.extent, (3,), "extent")
        if not np.all(extent > 0.0):
            raise ValueError("extent must be positive")
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "rotation", as_rotation(self.rotation))

    @property
    def volume(self) -> float:
        """The box volume."""
        return float(np.prod(self.extent))

    def corners(self) -> NDArray[np.float64]:
        """Returns the eight corners in world coordinates.

        Returns:
            Array of shape ``(8, 3)``. Row ``i`` is the corner whose local
            signs are ``itertools.product((-1, 1), repeat=3)[i]``, so the
            local x sign changes slowest.
        """
        local = _CORNER_SIGNS * (self.extent / 2.0)
        return local @ self.rotation.T + self.center

    def contains(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Tests which points lie inside the box, boundary included.

        Args:
            points: World points of shape ``(N, 3)``.

        Returns:
            Boolean mask of shape ``(N,)``.

        Raises:
            ValueError: If ``points`` does not have shape ``(N, 3)``.
        """
        local = SE3(self.rotation, self.center).apply_inverse(points)
        np.abs(local, out=local)
        return np.all(local <= self.extent / 2.0, axis=1)


@dataclass(eq=False)
class ObjectInstance:
    """A segmented object observed in one epoch.

    Attributes:
        object_id: Identifier, unique within its epoch.
        epoch: Capture the object was observed in.
        points: Object points in world coordinates, shape ``(N, 3)``.
        box: Oriented bounding box of the object.
        label: Optional open-vocabulary class name.
        label_score: Optional score of ``label``.
        feature: Optional appearance descriptor used for matching.
    """

    object_id: str
    epoch: Epoch
    points: NDArray[np.float64]
    box: OrientedBox
    label: str | None = None
    label_score: float | None = None
    feature: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        """Checks the point array.

        Raises:
            ValueError: If ``points`` is not a float64 ``(N, 3)`` array.
        """
        check_array(self.points, (None, 3), np.float64, "points")
