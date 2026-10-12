"""Scale policy and gravity alignment into a metric z-up room frame."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3, Sim3
from spatial_change.domain.transforms import as_finite_points
from spatial_change.geometry.floor import (
    FloorConfig,
    FloorNotFoundError,
    FloorPlane,
    estimate_up,
    fit_floor,
    floor_frame,
)


class ScalePolicy(StrEnum):
    """How the reconstruction's units are mapped to meters.

    Attributes:
        MODEL: Trust the reconstruction's units. This is the default for
            backbones that predict approximately metric depth, such as Pi3X.
        CAMERA_HEIGHT: Heuristic for non-metric backbones. Scales the scene
            so the median camera height above the floor equals the assumed
            camera height. It assumes handheld video at roughly a constant
            height (about 1.4 m for a phone held at chest to eye level) and
            fails when the camera is crouched, held overhead, mounted on a
            tripod or gimbal at another height, or when the floor fit is
            wrong. Its error is the relative error of that assumed height.
    """

    MODEL = "model"
    CAMERA_HEIGHT = "camera_height"


@dataclass(frozen=True)
class RoomAlignment:
    """The transform from a reconstruction's world frame into the room frame.

    Attributes:
        room_from_world: Rigid map into the z-up floor frame followed by the
            uniform scale about the room origin, so the floor stays at
            ``z = 0``.
        floor: The floor plane, in world units.
        policy: The scale policy that produced the scale.
        floor_observed: Whether ``floor`` was fitted (True) or assumed
            (False).
    """

    room_from_world: Sim3
    floor: FloorPlane
    policy: ScalePolicy
    floor_observed: bool

    @property
    def scale(self) -> float:
        """Uniform scale applied by ``room_from_world``."""
        return self.room_from_world.scale


def estimate_scale(
    camera_heights: ArrayLike,
    policy: ScalePolicy | str,
    assumed_camera_height_m: float = 1.4,
) -> float:
    """Returns the factor that converts reconstruction units to meters.

    Args:
        camera_heights: Camera heights above the floor, in reconstruction
            units, shape ``(K,)``.
        policy: The scale policy, or its string value.
        assumed_camera_height_m: Median camera height assumed by
            ``ScalePolicy.CAMERA_HEIGHT``, in meters.

    Returns:
        1 for ``MODEL``; ``assumed_camera_height_m / median(heights)`` for
        ``CAMERA_HEIGHT``.

    Raises:
        ValueError: If the policy is unknown, the heights are empty or not a
            finite 1D array, (for ``CAMERA_HEIGHT``) their median is not
            positive, or the assumed height is not positive and finite.
    """
    policy = ScalePolicy(policy)
    heights = np.asarray(camera_heights, dtype=np.float64)
    if heights.ndim != 1 or len(heights) == 0 or not np.isfinite(heights).all():
        raise ValueError("camera_heights must be a non-empty finite 1D array")
    if not (
        np.isfinite(assumed_camera_height_m) and assumed_camera_height_m > 0.0
    ):
        raise ValueError("assumed_camera_height_m must be positive and finite")
    if policy == ScalePolicy.MODEL:
        return 1.0
    median = float(np.median(heights))
    if median <= 0.0:
        raise ValueError(
            f"median camera height must be positive, got {median}; the "
            "floor fit may be upside down"
        )
    return assumed_camera_height_m / median


_DEFAULT_FLOOR_CONFIG = FloorConfig()
# Percentile of the point heights taken as the floor level: it sizes the
# coarse scale and places the assumed floor.
_FLOOR_PERCENTILE = 2.0
# The fitted floor must lie at least half way from the cameras down to the
# low end of the points to count as observed.
_OBSERVED_FLOOR_FRACTION = 0.5


def _choose_floor(
    cloud: NDArray[np.float64],
    up: NDArray[np.float64],
    centers: NDArray[np.float64],
    point_heights: NDArray[np.float64],
    low: float,
    floor_config: FloorConfig,
) -> tuple[FloorPlane, bool]:
    """Returns the fitted floor, or the assumed level floor, and which.

    Args:
        cloud: Scene points of shape ``(N, 3)``.
        up: Unit up direction.
        centers: Camera centers of shape ``(K, 3)``.
        point_heights: Heights of the points along ``up``, shape ``(N,)``.
        low: The ``_FLOOR_PERCENTILE`` of ``point_heights``.
        floor_config: Floor search parameters, in the units of ``cloud``.

    Returns:
        The floor and whether it was observed.

    Raises:
        ValueError: If the cameras are not above the low end of the points,
            or a fitted floor lies far above the cameras.
    """
    depth = float(np.median(centers @ up)) - low
    if depth <= 0.0:
        raise ValueError(
            "median camera height above the low end of the points must be "
            "positive; the up direction is wrong"
        )
    try:
        fitted = fit_floor(cloud, up, floor_config)
    except FloorNotFoundError:
        fitted = None
    if fitted is not None:
        ratio = float(np.median(fitted.height(centers))) / depth
        if ratio >= _OBSERVED_FLOOR_FRACTION:
            return fitted, True
        if ratio <= -_OBSERVED_FLOOR_FRACTION:
            raise ValueError(
                "median camera height is far below a fitted floor; the up "
                "direction is wrong"
            )
    near = np.abs(point_heights - low) < floor_config.distance_threshold
    return FloorPlane(up, -low, int(np.count_nonzero(near))), False


def gravity_align(
    points: ArrayLike,
    camera_poses: Sequence[SE3],
    policy: ScalePolicy | str = ScalePolicy.MODEL,
    floor_config: FloorConfig = _DEFAULT_FLOOR_CONFIG,
    assumed_camera_height_m: float = 1.4,
) -> RoomAlignment:
    """Computes the transform from the world frame to a metric room frame.

    The up direction comes from the cameras (``estimate_up``), the floor
    from ``fit_floor``, and the heading from the first camera's viewing
    direction. Scale follows ``policy``, with camera heights measured above
    the floor. Only the transform is returned; apply it to frames and clouds
    separately.

    The floor counts as observed if ``fit_floor`` finds a horizontal plane
    that lies at least half way from the median camera center down to the
    low end of the points (a low percentile of their heights along up). The
    rule has no length in it, so it holds in any units. A scene that never
    shows its floor (a closet full of coats) yields a shelf or a wall base
    near camera height instead. Then the floor is assumed: a level plane
    perpendicular to up at the low end of the points, and
    ``RoomAlignment.floor_observed`` is False.

    Call this ONCE on the union of both epochs' points and cameras, and apply
    the single returned ``Sim3`` to both epochs. Separate per-epoch calls fit
    different floors and headings, so the epochs end up with different yaws,
    and a refinement started from the identity then fails silently once the
    yaw difference reaches about 10 degrees.

    ``floor_config.distance_threshold`` is in meters, but the floor is
    fitted in world units. Under ``CAMERA_HEIGHT`` it is divided by a coarse
    scale first. That scale uses the median camera height above a low
    percentile of the point heights along the up direction, which
    approximates the floor level. The final scale then comes from the floor.
    The returned floor is expressed in world units.

    Args:
        points: Scene points of shape ``(N, 3)`` in the world frame.
        camera_poses: Camera-to-world poses of the epoch.
        policy: The scale policy, or its string value.
        floor_config: Floor search parameters, in meters.
        assumed_camera_height_m: Camera height assumed by
            ``ScalePolicy.CAMERA_HEIGHT``, in meters.

    Returns:
        The alignment into the z-up room frame.

    Raises:
        ValueError: If the policy is unknown, there are no poses, the points
            are not finite, the median camera center is not above the low
            end of the points, or a fitted floor lies far above the cameras
            (the up direction is wrong, for example upside down).
    """
    policy = ScalePolicy(policy)
    cloud = as_finite_points(points, "points", 3)
    up = estimate_up(camera_poses)
    centers = np.array([pose.translation for pose in camera_poses])
    point_heights = cloud @ up
    low = float(np.percentile(point_heights, _FLOOR_PERCENTILE))
    if policy == ScalePolicy.CAMERA_HEIGHT:
        coarse = estimate_scale(
            centers @ up - low, policy, assumed_camera_height_m
        )
        floor_config = replace(
            floor_config,
            distance_threshold=floor_config.distance_threshold / coarse,
        )
    floor, observed = _choose_floor(
        cloud, up, centers, point_heights, low, floor_config
    )
    frame = floor_frame(floor, camera_poses[0].rotation[:, 2])
    heights = floor.height(centers)
    scale = estimate_scale(heights, policy, assumed_camera_height_m)
    room_from_world = Sim3(scale, frame.rotation, scale * frame.translation)
    return RoomAlignment(room_from_world, floor, policy, observed)
