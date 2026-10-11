"""Camera intrinsics, capture epochs, and posed frames.

Conventions:
    * OpenCV pinhole model: x points right, y points down, z points forward.
      A camera-frame point ``(x, y, z)`` with ``z > 0`` projects to
      ``u = fx * x / z + cx`` and ``v = fy * y / z + cy``.
    * Pixel centers sit at integer coordinates, so pixel ``(0, 0)`` covers
      ``[-0.5, 0.5) x [-0.5, 0.5)``.
    * ``world_from_camera`` maps camera coordinates to world coordinates.
    * Images are ``(H, W, 3)`` uint8 RGB. Depth and confidence maps are
      ``(H, W)`` float32; a depth of 0 or a non-finite depth means invalid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain.transforms import SE3, Sim3, check_array

_WHOLE_PIXEL_ATOL = 1e-6


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for an image of a given size.

    Attributes:
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Principal point column in pixels.
        cy: Principal point row in pixels.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a focal length is not positive and finite, the
                principal point is not finite, or a dimension is not
                positive.
        """
        if not all(math.isfinite(f) and f > 0 for f in (self.fx, self.fy)):
            raise ValueError("focal lengths must be positive and finite")
        if not all(math.isfinite(c) for c in (self.cx, self.cy)):
            raise ValueError("principal point must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def matrix(self) -> NDArray[np.float64]:
        """The 3x3 calibration matrix ``K``."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    def scaled(self, sx: float, sy: float) -> Intrinsics:
        """Returns the intrinsics after resizing the image.

        With pixel centers at integer coordinates, a resize by ``s`` maps a
        coordinate ``u`` to ``s * (u + 0.5) - 0.5``, so the principal point
        shifts by half a pixel on top of the plain scaling.

        Args:
            sx: Horizontal resize factor, new width over old width.
            sy: Vertical resize factor, new height over old height.

        Returns:
            Intrinsics for the ``(width * sx, height * sy)`` image.

        Raises:
            ValueError: If a factor is not positive or the new size is not a
                whole number of pixels.
        """
        if sx <= 0 or sy <= 0:
            raise ValueError("resize factors must be positive")
        width, height = self.width * sx, self.height * sy
        new_width, new_height = round(width), round(height)
        if (
            abs(width - new_width) > _WHOLE_PIXEL_ATOL
            or abs(height - new_height) > _WHOLE_PIXEL_ATOL
        ):
            raise ValueError(
                f"resize to ({width}, {height}) is not a whole pixel size"
            )
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=sx * (self.cx + 0.5) - 0.5,
            cy=sy * (self.cy + 0.5) - 0.5,
            width=new_width,
            height=new_height,
        )


class Epoch(StrEnum):
    """Which of the two captures a piece of data comes from."""

    BEFORE = "before"
    AFTER = "after"


@dataclass(eq=False)
class Frame:
    """One posed video frame with optional pixel data.

    Pixel arrays are stored by reference, not copied, because they can be
    large.

    Attributes:
        epoch: Capture the frame belongs to.
        index: Frame index in the source video.
        timestamp_s: Timestamp in the source video, in seconds.
        intrinsics: Intrinsics at the resolution of the pixel arrays.
        world_from_camera: Camera-to-world pose.
        image: Optional ``(H, W, 3)`` uint8 RGB image.
        depth: Optional ``(H, W)`` float32 depth along the camera z axis.
            A value of 0 or a non-finite value means invalid.
        confidence: Optional ``(H, W)`` float32 per-pixel depth confidence.
    """

    epoch: Epoch
    index: int
    timestamp_s: float
    intrinsics: Intrinsics
    world_from_camera: SE3
    image: NDArray[np.uint8] | None = None
    depth: NDArray[np.float32] | None = None
    confidence: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        """Checks that pixel arrays match the intrinsics.

        Raises:
            ValueError: If an array's shape or dtype does not match
                ``(intrinsics.height, intrinsics.width)``.
        """
        hw = (self.intrinsics.height, self.intrinsics.width)
        if self.image is not None:
            check_array(self.image, (*hw, 3), np.uint8, "image")
        if self.depth is not None:
            check_array(self.depth, hw, np.float32, "depth")
        if self.confidence is not None:
            check_array(self.confidence, hw, np.float32, "confidence")

    def transformed(self, world_from_world: Sim3) -> Frame:
        """Re-expresses the frame in another world frame.

        The pose moves through ``Sim3.transform_pose`` and the depth is
        multiplied by the scale, because depth is a camera-frame quantity
        that the pose no longer carries once the world changes units.

        Args:
            world_from_world: Similarity from the current world to the
                target world.

        Returns:
            A new frame in the target world; intrinsics, image, confidence,
            epoch, index, and timestamp are shared with this frame.
        """
        depth = self.depth
        if depth is not None:
            depth = depth * np.float32(world_from_world.scale)
        return replace(
            self,
            world_from_camera=world_from_world.transform_pose(
                self.world_from_camera
            ),
            depth=depth,
        )
