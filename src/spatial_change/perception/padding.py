"""Image validation and padding shared by the feature encoders."""

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain.transforms import check_array


def check_image(image: NDArray[np.uint8]) -> None:
    """Checks that an image is a non-empty RGB uint8 array.

    Args:
        image: The array to check.

    Raises:
        ValueError: If it is not ``(H, W, 3)`` uint8 with ``H, W >= 1``.
    """
    check_array(image, (None, None, 3), np.uint8, "image")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"image must not be empty, got shape {image.shape}")


def pad_to_patches(
    image: NDArray[np.uint8], patch_size: int
) -> NDArray[np.uint8]:
    """Pads an image at the bottom and right to whole patches.

    The border pixels are replicated, so feature ``(i, j)`` of the padded
    image covers rows ``i * patch_size`` to ``(i + 1) * patch_size - 1`` of
    the input exactly.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        patch_size: Patch side in pixels, at least 1.

    Returns:
        The image itself when it already fits, otherwise a padded copy of
        shape ``(ceil(H / p) * p, ceil(W / p) * p, 3)``.
    """
    height, width = image.shape[:2]
    pad_h = -height % patch_size
    pad_w = -width % patch_size
    if not pad_h and not pad_w:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
