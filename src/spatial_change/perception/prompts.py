"""Validation of box and point prompts shared by the mask segmenters."""

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain.transforms import check_array


def check_prompts(
    boxes_xyxy: NDArray[np.float64], points_xy: NDArray[np.float64] | None
) -> None:
    """Checks the box and point prompts.

    Args:
        boxes_xyxy: The ``(K, 4)`` boxes.
        points_xy: The optional ``(K, 2)`` points.

    Raises:
        ValueError: If a shape, dtype, or value is invalid, or the point
            count differs from the box count.
    """
    check_array(boxes_xyxy, (None, 4), np.float64, "boxes_xyxy")
    if not np.isfinite(boxes_xyxy).all():
        raise ValueError("boxes_xyxy must be finite")
    if (boxes_xyxy[:, 0] > boxes_xyxy[:, 2]).any() or (
        boxes_xyxy[:, 1] > boxes_xyxy[:, 3]
    ).any():
        raise ValueError("boxes_xyxy must satisfy x0 <= x1 and y0 <= y1")
    if points_xy is None:
        return
    check_array(points_xy, (len(boxes_xyxy), 2), np.float64, "points_xy")
    if not np.isfinite(points_xy).all():
        raise ValueError("points_xy must be finite")
