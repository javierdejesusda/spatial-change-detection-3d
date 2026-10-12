"""Mask helpers shared by the SceneDiff loader, projection and metrics."""

import numpy as np
from numpy.typing import NDArray

_MAX_SCATTER = 1 << 22


def _nearest_index(src: int, dst: int) -> NDArray[np.intp]:
    scale = np.float32(src) / np.float32(dst)
    index = np.floor(np.arange(dst, dtype=np.float32) * scale)
    return np.minimum(index.astype(np.intp), src - 1)


def resize_nearest(
    mask: NDArray[np.bool_], height: int, width: int
) -> NDArray[np.bool_]:
    """Resizes a mask with nearest-neighbor sampling.

    Destination pixel ``i`` reads source pixel
    ``min(floor(i * scale), src - 1)`` with ``scale = src / dst`` in float32,
    the rule of ``torch.nn.functional.interpolate(mode="nearest")`` that
    the official evaluation uses. Integer arithmetic differs from it for
    some sizes, so the float32 rule is kept.

    Args:
        mask: ``(h, w)`` boolean mask.
        height: Target height.
        width: Target width.

    Returns:
        The ``(height, width)`` mask.
    """
    src_h, src_w = mask.shape
    if (src_h, src_w) == (height, width):
        return mask
    rows = _nearest_index(src_h, height)
    cols = _nearest_index(src_w, width)
    return mask[np.ix_(rows, cols)]


def splat(
    rows: NDArray[np.intp],
    cols: NDArray[np.intp],
    height: int,
    width: int,
    radius: int,
) -> NDArray[np.bool_]:
    """Draws a disk around every point, clipped to the image.

    Args:
        rows: Point rows, all inside the image.
        cols: Point columns, all inside the image.
        height: Image height.
        width: Image width.
        radius: Disk radius in pixels; a pixel is drawn when its offset
            ``(dy, dx)`` has ``dy * dy + dx * dx <= radius * radius``.

    Returns:
        The ``(height, width)`` boolean mask.
    """
    out = np.zeros((height, width), dtype=bool)
    if rows.size == 0:
        return out
    reach = min(radius, max(height, width) - 1)
    span = np.arange(-reach, reach + 1)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    inside = dy * dy + dx * dx <= radius * radius
    dy, dx = dy[inside], dx[inside]
    pixels = np.unique(rows * width + cols)
    row0, col0 = pixels // width, pixels % width
    step = max(1, _MAX_SCATTER // dy.size)
    for start in range(0, pixels.size, step):
        r = row0[start : start + step, None] + dy
        c = col0[start : start + step, None] + dx
        ok = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        out[r[ok], c[ok]] = True
    return out
