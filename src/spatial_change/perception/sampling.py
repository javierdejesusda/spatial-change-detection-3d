"""Sampling a patch feature grid at pixel positions."""

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain.transforms import check_array


def sample_features(
    features: NDArray[np.float32],
    uv: NDArray[np.float64],
    patch_size: int,
) -> NDArray[np.float32]:
    """Bilinearly interpolates a patch feature grid at pixel positions.

    Pixel centers sit at integer coordinates (x right, y down). Feature
    ``(i, j)`` describes the patch whose center is the pixel position
    ``((j + 0.5) * patch_size - 0.5, (i + 0.5) * patch_size - 0.5)``.
    Positions beyond the outer patch centers take the border patches.

    Args:
        features: ``(h, w, C)`` float32 grid with ``h, w >= 1``.
        uv: Pixel positions of shape ``(N, 2)`` as ``(x, y)``.
        patch_size: Side of a patch in pixels, at least 1.

    Returns:
        ``(N, C)`` float32 interpolated features.

    Raises:
        ValueError: If an array has the wrong shape or dtype, the grid is
            empty, or the patch size is below 1.
    """
    check_array(features, (None, None, None), np.float32, "features")
    check_array(uv, (None, 2), np.float64, "uv")
    if features.size == 0:
        raise ValueError("features must not be empty")
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    height, width = features.shape[:2]
    # Grid coordinates: patch (i, j) sits at integer (j, i).
    x = np.clip((uv[:, 0] + 0.5) / patch_size - 0.5, 0.0, width - 1.0)
    y = np.clip((uv[:, 1] + 0.5) / patch_size - 0.5, 0.0, height - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.intp), width - 2).clip(0)
    y0 = np.minimum(np.floor(y).astype(np.intp), height - 2).clip(0)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0).astype(np.float32)[:, None]
    fy = (y - y0).astype(np.float32)[:, None]
    flat = features.reshape(-1, features.shape[2])
    result = flat[y0 * width + x0] * ((1 - fx) * (1 - fy))
    result += flat[y0 * width + x1] * (fx * (1 - fy))
    result += flat[y1 * width + x0] * ((1 - fx) * fy)
    result += flat[y1 * width + x1] * (fx * fy)
    return np.asarray(result, dtype=np.float32)
