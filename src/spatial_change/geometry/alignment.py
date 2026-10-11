"""Closed-form alignment of corresponding 3D point sets."""

import numpy as np
from numpy.typing import ArrayLike

from spatial_change.domain import Sim3
from spatial_change.domain.transforms import as_finite_points

# The rotation is unique only when the cross-covariance has rank >= 2. A
# second singular value this small relative to the first means one of the
# point sets is collinear or coincident.
_DEGENERACY_RTOL = 1e-9
_MIN_POINTS = 3


def umeyama(src: ArrayLike, dst: ArrayLike, with_scale: bool = True) -> Sim3:
    """Finds the similarity that best maps ``src`` onto ``dst``.

    Minimizes ``sum ||dst_i - (s * R @ src_i + t)||^2`` in closed form
    (Umeyama 1991, "Least-squares estimation of transformation parameters
    between two point patterns"), including the correction that keeps ``R``
    a proper rotation when the best orthogonal fit is a reflection.

    Args:
        src: Source points of shape ``(N, 3)``.
        dst: Corresponding target points of shape ``(N, 3)``.
        with_scale: Whether to estimate a scale. When False, the scale is
            fixed to 1 and only a rigid transform is fitted.

    Returns:
        The transform ``T`` for which ``T.apply(src)`` best fits ``dst``.

    Raises:
        ValueError: If there are fewer than 3 points, the shapes differ, a
            value is not finite, or the points do not determine a rotation
            (either set is collinear or coincident).
    """
    src_points = as_finite_points(src, "src", _MIN_POINTS)
    dst_points = as_finite_points(dst, "dst", _MIN_POINTS)
    if src_points.shape != dst_points.shape:
        raise ValueError(
            f"src and dst shapes differ: {src_points.shape} vs "
            f"{dst_points.shape}"
        )
    n = len(src_points)

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_c = src_points - src_mean
    covariance = (dst_points - dst_mean).T @ src_c / n
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] <= _DEGENERACY_RTOL * singular[0]:
        raise ValueError("points are collinear or coincident")

    signs = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        signs[2] = -1.0
    rotation = (u * signs) @ vt

    scale = 1.0
    if with_scale:
        src_variance = np.vdot(src_c, src_c) / n
        scale = float(singular @ signs / src_variance)

    translation = dst_mean - scale * rotation @ src_mean
    return Sim3(scale, rotation, translation)
