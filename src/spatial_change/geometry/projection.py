"""Pinhole projection between world points and pixels.

Uses the OpenCV convention documented in ``spatial_change.domain.camera``:
x right, y down, z forward, pixel centers at integer coordinates, and
``world_from_camera`` poses that map camera to world coordinates.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3, Intrinsics
from spatial_change.domain.transforms import check_array

DEFAULT_MIN_CONFIDENCE = 0.1
"""Default per-pixel confidence threshold; Pi3's demo uses 0.1 as well."""


def project(
    points_world: ArrayLike,
    intrinsics: Intrinsics,
    world_from_camera: SE3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Projects world points into a camera.

    No visibility filtering happens here; combine the outputs with
    ``in_image`` or a depth test as needed. Points with zero depth get
    non-finite pixel coordinates.

    Args:
        points_world: World points of shape ``(N, 3)``.
        intrinsics: Camera intrinsics.
        world_from_camera: Camera-to-world pose.

    Returns:
        A tuple ``(uv, depth)``: pixel coordinates of shape ``(N, 2)`` and
        camera-frame z of shape ``(N,)``, both float64. Points behind the
        camera have negative depth.

    Raises:
        ValueError: If ``points_world`` does not have shape ``(N, 3)``.
    """
    points_camera = world_from_camera.apply_inverse(points_world)
    depth = points_camera[:, 2].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics.fx * points_camera[:, 0] / depth + intrinsics.cx
        v = intrinsics.fy * points_camera[:, 1] / depth + intrinsics.cy
    return np.stack([u, v], axis=1), depth


def in_image(
    uv: NDArray[np.float64],
    depth: NDArray[np.float64],
    intrinsics: Intrinsics,
) -> NDArray[np.bool_]:
    """Tests which projections land in front of the camera and in the image.

    Pixel centers sit at integer coordinates, so the image covers
    ``[-0.5, W - 0.5) x [-0.5, H - 0.5)``: every pixel center, including
    those on the first row and column, is inside, and ``np.rint`` of an
    inside coordinate is always a valid index.

    Args:
        uv: Pixel coordinates of shape ``(N, 2)``.
        depth: Camera-frame z of shape ``(N,)``.
        intrinsics: Camera intrinsics giving ``W`` and ``H``.

    Returns:
        Boolean mask of shape ``(N,)``: ``depth > 0`` and inside the image.
        Non-finite coordinates are outside.
    """
    u, v = uv[:, 0], uv[:, 1]
    return (
        (depth > 0.0)
        & (u >= -0.5)
        & (u < intrinsics.width - 0.5)
        & (v >= -0.5)
        & (v < intrinsics.height - 0.5)
    )


def confident_pixels(
    depth: NDArray[np.float32],
    confidence: NDArray[np.float32],
    threshold: float,
) -> NDArray[np.bool_]:
    """Marks the pixels with a valid depth and enough confidence.

    Args:
        depth: ``(H, W)`` depth map; 0 and non-finite values are invalid.
        confidence: ``(H, W)`` confidence map.
        threshold: Minimum confidence, inclusive.

    Returns:
        ``(H, W)`` boolean mask.
    """
    return np.isfinite(depth) & (depth > 0) & (confidence >= threshold)


def unproject_depth(
    depth: NDArray[np.float32],
    intrinsics: Intrinsics,
    world_from_camera: SE3,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Lifts valid depth pixels to world points.

    Args:
        depth: Depth map of shape ``(H, W)``, z along the camera axis. A
            pixel is valid when its depth is finite and positive.
        intrinsics: Intrinsics at the depth map's resolution.
        world_from_camera: Camera-to-world pose.
        mask: Optional ``(H, W)`` boolean mask; only pixels where it is True
            are kept.

    Returns:
        World points of shape ``(M, 3)``, float64, one per valid pixel in
        row-major pixel order.

    Raises:
        ValueError: If ``depth`` is not float32 or ``mask`` is not bool, or
            either does not have shape ``(H, W)``.
    """
    hw = (intrinsics.height, intrinsics.width)
    check_array(depth, hw, np.float32, "depth")
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        check_array(mask, hw, np.bool_, "mask")
        valid &= mask
    rows, cols = np.nonzero(valid)
    z_valid = depth[rows, cols].astype(np.float64)
    points_camera = np.stack(
        [
            (cols - intrinsics.cx) / intrinsics.fx * z_valid,
            (rows - intrinsics.cy) / intrinsics.fy * z_valid,
            z_valid,
        ],
        axis=1,
    )
    return world_from_camera.apply(points_camera)
