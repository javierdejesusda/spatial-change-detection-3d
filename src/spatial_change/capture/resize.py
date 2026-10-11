"""Resizing to the grid a patch-based backbone expects."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_change.domain.transforms import check_array


@dataclass(frozen=True)
class ResizePolicy:
    """Target size rules for a backbone.

    Attributes:
        long_side: Maximum length of the longer image side, in pixels.
        multiple_of: Both sides are rounded to a multiple of this value,
            for example 14 for ViT backbones with 14 px patches.

    Raises:
        ValueError: If ``multiple_of`` is not positive or ``long_side`` is not
            a multiple of it (rounding could otherwise exceed the cap).
    """

    long_side: int
    multiple_of: int

    def __post_init__(self) -> None:
        """Validates the policy."""
        if (
            self.multiple_of < 1
            or self.long_side < self.multiple_of
            or self.long_side % self.multiple_of
        ):
            raise ValueError(
                f"long_side {self.long_side} must be a positive multiple of "
                f"multiple_of {self.multiple_of}"
            )

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        """Returns the output size for an input image size.

        The long side is capped at ``long_side`` (small images are not scaled
        up beyond rounding), then both sides are rounded to the nearest
        multiple of ``multiple_of`` with a minimum of one multiple.

        Args:
            height: Input height in pixels.
            width: Input width in pixels.

        Returns:
            The output ``(height, width)`` in pixels.
        """
        scale = min(1.0, self.long_side / max(height, width))
        return (
            _round_side(height * scale, self.multiple_of),
            _round_side(width * scale, self.multiple_of),
        )


def _round_side(size: float, multiple_of: int) -> int:
    return max(multiple_of, round(size / multiple_of) * multiple_of)


def resize_for_backbone(
    image: NDArray[np.uint8], policy: ResizePolicy
) -> tuple[NDArray[np.uint8], float, float]:
    """Resizes an image to the backbone grid, keeping the aspect ratio.

    The long side is capped at ``policy.long_side`` (small images are not
    scaled up beyond rounding), then both sides are rounded to the nearest
    multiple of ``policy.multiple_of`` with a minimum of one multiple.

    Args:
        image: RGB image of shape (H, W, 3), dtype uint8.
        policy: Target size rules.

    Returns:
        The resized image, and the scale factors ``sx = new_w / old_w`` and
        ``sy = new_h / old_h`` for adjusting camera intrinsics.
    """
    height, width = image.shape[:2]
    new_h, new_w = policy.target_size(height, width)
    return _resize(image, new_w, new_h), new_w / width, new_h / height


def resize_image(
    image: NDArray[np.uint8], width: int, height: int
) -> NDArray[np.uint8]:
    """Resizes an RGB image, averaging when shrinking.

    Area averaging avoids aliasing when the image shrinks in both
    directions; any enlargement uses bilinear interpolation, because area
    averaging degrades to nearest-neighbor there.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        The resized image, or ``image`` itself if it already has the target
        size.

    Raises:
        ValueError: If the target size is not positive or ``image`` is not
            an ``(H, W, 3)`` uint8 array.
    """
    check_array(image, (None, None, 3), np.uint8, "image")
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")
    old_height, old_width = image.shape[:2]
    if (old_width, old_height) == (width, height):
        return image
    return _resize(image, width, height)


def _resize(
    image: NDArray[np.uint8], width: int, height: int
) -> NDArray[np.uint8]:
    old_height, old_width = image.shape[:2]
    shrinking = width <= old_width and height <= old_height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return np.asarray(
        cv2.resize(image, (width, height), interpolation=interpolation),
        dtype=np.uint8,
    )
