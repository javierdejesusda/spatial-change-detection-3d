"""Colored points of one epoch, thinned for display.

Everything here is in the z-up room frame of the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain.transforms import check_array


@dataclass(frozen=True, eq=False)
class ScenePoints:
    """Colored points of one epoch.

    Attributes:
        points: World points of shape ``(N, 3)``, float64.
        colors: Point colors of shape ``(N, 3)``, uint8 RGB.
    """

    points: NDArray[np.float64]
    colors: NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Checks array shapes and dtypes.

        Raises:
            ValueError: If an array has the wrong shape or dtype, or the
                two lengths differ.
        """
        check_array(self.points, (None, 3), np.float64, "points")
        check_array(self.colors, (None, 3), np.uint8, "colors")
        if len(self.points) != len(self.colors):
            raise ValueError("points and colors lengths differ")

    @classmethod
    def empty(cls) -> ScenePoints:
        """Returns a cloud with no points."""
        return cls(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
        )


def subsample(cloud: ScenePoints, cap: int) -> ScenePoints:
    """Keeps the finite points, thinned evenly to at most ``cap``.

    The choice depends only on the order of the points, so equal input
    gives equal output.

    Args:
        cloud: The points to thin.
        cap: Largest number of points to keep, at least 1.

    Returns:
        The kept points in their original order.

    Raises:
        ValueError: If ``cap`` is below 1.
    """
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    finite = np.isfinite(cloud.points).all(axis=1)
    points, colors = cloud.points[finite], cloud.colors[finite]
    count = len(points)
    if count <= cap:
        return ScenePoints(points, colors)
    keep = (np.arange(cap, dtype=np.int64) * count) // cap
    return ScenePoints(points[keep], colors[keep])
