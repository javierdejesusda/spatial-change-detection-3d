"""Pure numpy conversion of Pi3X outputs into pipeline geometry.

Shapes follow the model: ``N`` views at a working resolution of ``H x W``
pixels. Camera-frame points use the OpenCV axes of
``spatial_change.domain.camera``.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import SE3, Intrinsics
from spatial_change.ports import ReconstructionResult

PATCH_SIZE = 14
_Floats = NDArray[np.floating[Any]]


def working_size(width: int, height: int, pixel_limit: int) -> tuple[int, int]:
    """Returns the Pi3 working resolution for an input image size.

    Ports the rule of Pi3's ``load_images_as_tensor``: scale to about
    ``pixel_limit`` pixels keeping the aspect ratio, round both sides to a
    multiple of the 14 px patch, and drop patch rows or columns until the
    grid fits. Each side keeps at least one patch, so an extreme aspect ratio
    or a limit below one patch can exceed ``pixel_limit``.

    Args:
        width: Input width in pixels.
        height: Input height in pixels.
        pixel_limit: Target pixel count of the working image.

    Returns:
        The working ``(width, height)``, both multiples of 14.

    Raises:
        ValueError: If an argument is not positive.
    """
    if width <= 0 or height <= 0 or pixel_limit <= 0:
        raise ValueError("width, height, and pixel_limit must be positive")
    scale = math.sqrt(pixel_limit / (width * height))
    target_w, target_h = width * scale, height * scale
    cols, rows = round(target_w / PATCH_SIZE), round(target_h / PATCH_SIZE)
    while (cols * PATCH_SIZE) * (rows * PATCH_SIZE) > pixel_limit:
        if cols / rows > target_w / target_h:
            cols -= 1
        else:
            rows -= 1
    return max(1, cols) * PATCH_SIZE, max(1, rows) * PATCH_SIZE


def intrinsics_from_local_points(
    local_points: _Floats,
    confidence: _Floats,
    width: int,
    height: int,
) -> Intrinsics:
    """Fits pinhole intrinsics to a camera-frame point map.

    With the principal point fixed at the image center, ``u - cx`` is
    linear in ``x / z`` (and likewise for ``v`` and ``y``), so each focal
    length is a one-parameter weighted least-squares fit. Points that are
    non-finite or lie at ``z <= 0`` are ignored, and so are points without a
    positive finite confidence, unless no point has one: a view whose
    confidences all underflowed to zero still fits, with every valid point
    weighted equally.

    Args:
        local_points: ``(H, W, 3)`` camera-frame points, one per pixel.
        confidence: ``(H, W)`` non-negative weight per point.
        width: Image width ``W`` in pixels.
        height: Image height ``H`` in pixels.

    Returns:
        Intrinsics with the principal point at ``((W - 1) / 2, (H - 1) / 2)``.

    Raises:
        ValueError: If an array shape does not match ``(height, width)``,
            there are no valid points to fit, or the points do not fit a
            pinhole camera with positive focal lengths.
    """
    if local_points.shape != (height, width, 3):
        raise ValueError(
            f"local_points must have shape {(height, width, 3)}, "
            f"got {local_points.shape}"
        )
    if confidence.shape != (height, width):
        raise ValueError(
            f"confidence must have shape {(height, width)}, "
            f"got {confidence.shape}"
        )
    cx, cy = (width - 1) / 2, (height - 1) / 2
    points = local_points.astype(np.float64)
    weight = confidence.astype(np.float64)
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    if not valid.any():
        raise ValueError("no valid points to fit the intrinsics")
    confident = valid & np.isfinite(weight) & (weight > 0)
    if confident.any():
        valid = confident
    else:
        weight = np.ones_like(weight)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    z = points[..., 2][valid]
    w = weight[valid]
    fx = _focal_length(cols[valid] - cx, points[..., 0][valid] / z, w)
    fy = _focal_length(rows[valid] - cy, points[..., 1][valid] / z, w)
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _focal_length(
    offset: _Floats,
    slope: _Floats,
    weight: _Floats,
) -> float:
    """Fits ``slope = offset / f`` by weighted least squares.

    Pixel offsets are exact and the predicted slopes carry the noise, so the
    slope is the dependent variable: regressing the other way would shrink
    ``f`` in proportion to the noise.
    """
    covariance = np.sum(weight * slope * offset)
    variance = np.sum(weight * offset * offset)
    if not (covariance > 0 and variance > 0):
        raise ValueError("points are inconsistent with a pinhole camera")
    return float(variance / covariance)


def nearest_rotation(matrix: _Floats) -> NDArray[np.float64]:
    """Projects a matrix onto the nearest proper rotation.

    Pi3 predicts rotations in float32 (bfloat16 on CPU), so they come back
    slightly off orthonormal, beyond the tolerance ``SE3`` accepts. The SVD
    projection ``U @ Vt`` is the closest orthogonal matrix in the Frobenius
    norm; if it is a reflection, the direction of the smallest singular
    value is flipped.

    Args:
        matrix: A ``(3, 3)`` finite matrix.

    Returns:
        A ``(3, 3)`` float64 rotation with determinant +1.

    Raises:
        ValueError: If ``matrix`` is not a finite ``(3, 3)`` array.
    """
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be a finite (3, 3) array")
    u, _, vt = np.linalg.svd(matrix.astype(np.float64))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1.0
    return np.asarray(u @ vt)


def to_reconstruction_result(
    camera_poses: _Floats,
    local_points: _Floats,
    conf_logits: _Floats,
) -> ReconstructionResult:
    """Converts the outputs of one Pi3X batch into a reconstruction result.

    Depth is the camera-frame z of the point map, with invalid entries set
    to 0, and confidence is the sigmoid of the logits (0 where the logit is
    not finite). The intrinsics are fitted per view at the working
    resolution of the point map.

    Args:
        camera_poses: ``(N, 4, 4)`` camera-to-world matrices in OpenCV axes.
        local_points: ``(N, H, W, 3)`` camera-frame points.
        conf_logits: ``(N, H, W, 1)`` confidence logits.

    Returns:
        One entry per view, in input order.

    Raises:
        ValueError: If the shapes are inconsistent or a view has no valid
            points.
    """
    if local_points.ndim != 4 or local_points.shape[-1] != 3:
        raise ValueError(
            f"local_points must have shape (N, H, W, 3), got "
            f"{local_points.shape}"
        )
    n, height, width, _ = local_points.shape
    if camera_poses.shape != (n, 4, 4):
        raise ValueError(
            f"camera_poses must have shape {(n, 4, 4)}, "
            f"got {camera_poses.shape}"
        )
    if conf_logits.shape != (n, height, width, 1):
        raise ValueError(
            f"conf_logits must have shape {(n, height, width, 1)}, "
            f"got {conf_logits.shape}"
        )
    intrinsics, poses, depth, confidence = [], [], [], []
    for pose, points, logits in zip(
        camera_poses, local_points, conf_logits, strict=True
    ):
        view_confidence = _sigmoid(logits[..., 0])
        z = points[..., 2]
        intrinsics.append(
            intrinsics_from_local_points(points, view_confidence, width, height)
        )
        poses.append(
            SE3(nearest_rotation(pose[:3, :3]), pose[:3, 3].astype(np.float64))
        )
        depth.append(
            np.where(np.isfinite(z) & (z > 0), z, 0).astype(np.float32)
        )
        confidence.append(view_confidence)
    return ReconstructionResult(intrinsics, poses, depth, confidence)


def _sigmoid(logits: _Floats) -> NDArray[np.float32]:
    """Numerically stable sigmoid that maps non-finite logits to 0."""
    finite = np.isfinite(logits)
    safe = np.where(finite, logits, 0.0)
    return np.where(finite, 0.5 * (1.0 + np.tanh(0.5 * safe)), 0.0).astype(
        np.float32
    )
