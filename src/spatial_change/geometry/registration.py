"""Rigid registration of one object's points onto another's.

Two objects of one scene are registered to tell whether they are the same
object seen before and after a move. Objects stand upright in the z-up room
frame, so the starts are a global feature match (FPFH and RANSAC) and a
ring of turns about the vertical axis around the centroid shift. Each is
refined by trimmed ICP and the best fit wins, preferring an upright pose
and a small rotation among equal fits. A feature match cannot tell a shape
from its flip about a horizontal axis when the shape has that symmetry, and
nor can the fitness, hence the preference.

Importing this module does not load open3d.
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3
from spatial_change.domain.transforms import as_finite_points
from spatial_change.geometry.boxes import yaw_rotation
from spatial_change.geometry.icp import IcpConfig, TargetIndex, trimmed_icp
from spatial_change.geometry.rotation import rotation_angle_deg

_NORMAL_NEIGHBORS = 30
_FEATURE_NEIGHBORS = 100


@dataclass(frozen=True)
class RegistrationConfig:
    """Parameters of ``register_objects``. The defaults are provisional.

    They are placeholders until T22 calibrates them on real captures.
    Lengths are in the units of the world frame, meters after the alignment
    stage.

    Attributes:
        voxel_size: Edge of the voxels both clouds are downsampled to,
            positive and finite.
        normal_radius: Search radius of the normals, positive and finite.
        feature_radius: Search radius of the FPFH features, positive and
            finite.
        inlier_distance: A point of the smaller cloud is an inlier when the
            larger cloud has a point within this distance; also the
            correspondence distance of the RANSAC, positive and finite.
        ransac_iterations: Maximum RANSAC iterations, at least 1.
        ransac_confidence: Confidence at which the RANSAC stops, in
            ``(0, 1]``.
        edge_length_ratio: Smallest ratio of the lengths of two matched
            edges for the RANSAC to keep a sample, in ``(0, 1]``.
        icp: Settings of the refinement.
        yaw_starts: Turns about the vertical axis, evenly spaced from 0, that
            start an ICP around the centroid shift, at least 1.
        max_tilt_deg: Angle between the object's up axis before and after
            above which a pose counts as tilted and loses to an upright one
            of equal fit, in ``[0, 180]``.
        tie_tolerance: Two fits whose fitness differs by at most this count
            as equal and the upright one with the smaller rotation wins, in
            ``[0, 1]``.
        min_points: Points each cloud needs after downsampling, at least
            3.
        min_voxels_across: Voxels the largest extent of the smaller object
            must span. A smaller object shrinks the voxel and every length
            above in proportion, so it keeps enough points; a larger one
            keeps the lengths as given. At least 3 and finite.
        seed: Seed of the RANSAC sampling, which makes runs repeat.
    """

    voxel_size: float = 0.03
    normal_radius: float = 0.06
    feature_radius: float = 0.15
    inlier_distance: float = 0.03
    ransac_iterations: int = 3000
    ransac_confidence: float = 0.999
    edge_length_ratio: float = 0.9
    icp: IcpConfig = field(
        default_factory=lambda: IcpConfig(max_iterations=200)
    )
    yaw_starts: int = 4
    max_tilt_deg: float = 30.0
    tie_tolerance: float = 0.02
    min_points: int = 30
    min_voxels_across: float = 12.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a length is not positive and finite, a count or
                share is out of range.
        """
        for name in (
            "voxel_size",
            "normal_radius",
            "feature_radius",
            "inlier_distance",
        ):
            if not 0.0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be positive and finite")
        if self.ransac_iterations < 1:
            raise ValueError("ransac_iterations must be >= 1")
        if not 0.0 < self.ransac_confidence <= 1.0:
            raise ValueError("ransac_confidence must be within (0, 1]")
        if not 0.0 < self.edge_length_ratio <= 1.0:
            raise ValueError("edge_length_ratio must be within (0, 1]")
        if self.yaw_starts < 1:
            raise ValueError("yaw_starts must be >= 1")
        if not 0.0 <= self.max_tilt_deg <= 180.0:
            raise ValueError("max_tilt_deg must be within [0, 180]")
        if not 0.0 <= self.tie_tolerance <= 1.0:
            raise ValueError("tie_tolerance must be within [0, 1]")
        if self.min_points < 3:
            raise ValueError("min_points must be >= 3")
        if not 3.0 <= self.min_voxels_across < math.inf:
            raise ValueError("min_voxels_across must be >= 3 and finite")


_DEFAULT_CONFIG = RegistrationConfig()


@dataclass(frozen=True, eq=False)
class Registration:
    """Outcome of ``register_objects``.

    Attributes:
        transform: Estimated ``target_from_source``; the identity when the
            registration failed.
        fitness: Share of the smaller cloud's points that have a point of
            the larger cloud within ``inlier_distance``, in ``[0, 1]``; 0
            when the registration failed.
        rmse: Root mean squared distance of those inliers, infinite when
            there is none.
        succeeded: Whether a transform could be estimated at all.
    """

    transform: SE3
    fitness: float
    rmse: float
    succeeded: bool


def _downsample(points: NDArray[np.float64], voxel_size: float) -> Any:
    """Downsamples points into an open3d cloud.

    Args:
        points: ``(N, 3)`` points.
        voxel_size: Voxel edge.

    Returns:
        An open3d point cloud with one point per occupied voxel.
    """
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud.voxel_down_sample(voxel_size)


def _evaluate(
    transform: SE3,
    source: NDArray[np.float64],
    index: TargetIndex,
    config: RegistrationConfig,
) -> Registration:
    """Measures how well a transform puts the source on the target.

    Args:
        transform: Candidate ``target_from_source``.
        source: ``(N, 3)`` source points.
        index: Nearest-neighbour index of the target.
        config: Supplies ``inlier_distance``.

    Returns:
        The transform with its inlier share and inlier RMSE, as a
        successful registration.
    """
    _, distance = index.nearest(transform.apply(source))
    inliers = distance <= config.inlier_distance
    rmse = (
        float(np.sqrt(np.mean(distance[inliers] ** 2)))
        if inliers.any()
        else math.inf
    )
    return Registration(transform, float(inliers.mean()), rmse, True)


def _feature_match(
    source: Any, target: Any, config: RegistrationConfig
) -> SE3 | None:
    """Finds a global pose by matching FPFH features with RANSAC.

    Args:
        source: Open3d cloud, the smaller one.
        target: Open3d cloud, the larger one.
        config: Search radii, RANSAC settings, and seed.

    Returns:
        The pose, or None when no sample survived the checks.
    """
    import open3d as o3d

    reg = o3d.pipelines.registration
    features = []
    for cloud in (source, target):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.normal_radius, max_nn=_NORMAL_NEIGHBORS
            )
        )
        # The stub lists the optional ``indices`` argument as required.
        features.append(
            reg.compute_fpfh_feature(  # type: ignore[call-arg]
                cloud,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=config.feature_radius, max_nn=_FEATURE_NEIGHBORS
                ),
            )
        )
    # The parallel RANSAC draws its samples per thread, so even a fixed seed
    # gives different poses from run to run; one thread makes it repeat.
    threads = o3d.utility.get_max_threads()
    o3d.utility.set_max_threads(1)
    try:
        o3d.utility.random.seed(config.seed)
        result = reg.registration_ransac_based_on_feature_matching(
            source,
            target,
            features[0],
            features[1],
            True,
            config.inlier_distance,
            reg.TransformationEstimationPointToPoint(False),
            3,
            [
                reg.CorrespondenceCheckerBasedOnEdgeLength(
                    config.edge_length_ratio
                ),
                reg.CorrespondenceCheckerBasedOnDistance(
                    config.inlier_distance
                ),
            ],
            reg.RANSACConvergenceCriteria(
                config.ransac_iterations, config.ransac_confidence
            ),
        )
    finally:
        o3d.utility.set_max_threads(threads)
    if len(result.correspondence_set) == 0:
        return None
    return SE3.from_matrix(np.asarray(result.transformation))


def _refine(
    start: SE3,
    source: NDArray[np.float64],
    index: TargetIndex,
    config: RegistrationConfig,
) -> Registration | None:
    """Refines a start pose with trimmed ICP and measures it.

    Args:
        start: Initial ``target_from_source``.
        source: ``(N, 3)`` source points.
        index: Nearest-neighbour index of the target.
        config: Supplies the ICP settings and ``inlier_distance``.

    Returns:
        The refined fit, or None when too few pairs survive the ICP gate.
    """
    try:
        result = trimmed_icp(source, index, start, config.icp)
    except ValueError:
        return None
    return _evaluate(result.transform, source, index, config)


def _tilt_deg(transform: SE3) -> float:
    """Measures how far a transform tips the vertical axis."""
    return float(
        np.degrees(np.arccos(np.clip(transform.rotation[2, 2], -1, 1)))
    )


def _choose(
    fits: list[Registration], config: RegistrationConfig
) -> Registration:
    """Picks the best fit, preferring an upright small rotation among equals.

    Args:
        fits: At least one fit.
        config: Supplies ``tie_tolerance`` and ``max_tilt_deg``.

    Returns:
        The fit with the highest fitness; among the fits within
        ``tie_tolerance`` of it, an upright one over a tilted one, and then
        the one that turns least.
    """
    top = max(fit.fitness for fit in fits)
    close = [f for f in fits if top - f.fitness <= config.tie_tolerance]
    return min(
        close,
        key=lambda f: (
            _tilt_deg(f.transform) > config.max_tilt_deg,
            rotation_angle_deg(f.transform.rotation),
        ),
    )


def _yaw_starts(
    small: NDArray[np.float64],
    large: NDArray[np.float64],
    count: int,
) -> list[SE3]:
    """Builds the starts that turn the smaller cloud about its centroid.

    Args:
        small: ``(N, 3)`` points of the smaller cloud.
        large: ``(M, 3)`` points of the larger cloud.
        count: Number of evenly spaced turns, from 0.

    Returns:
        The poses that turn the smaller cloud about the vertical axis
        through its centroid and then put the centroid on the larger one's.
    """
    source_center, target_center = small.mean(axis=0), large.mean(axis=0)
    starts = []
    for step in range(count):
        rotation = yaw_rotation(2.0 * math.pi * step / count)
        starts.append(SE3(rotation, target_center - rotation @ source_center))
    return starts


def _fitted(
    config: RegistrationConfig, clouds: list[NDArray[np.float64]]
) -> RegistrationConfig:
    """Shrinks the lengths of a config to fit the smaller object.

    Args:
        config: The settings, for objects of ``min_voxels_across`` voxels or
            more.
        clouds: ``(N, 3)`` points of the two objects.

    Returns:
        The config itself when the smaller object's largest extent spans at
        least ``min_voxels_across`` voxels, else one whose voxel and other
        lengths are scaled down until it does.
    """
    extent = min(
        (float(np.ptp(cloud, axis=0).max()) for cloud in clouds if len(cloud)),
        default=math.inf,
    )
    scale = min(1.0, extent / (config.min_voxels_across * config.voxel_size))
    if not 0.0 < scale < 1.0:
        return config
    return dataclasses.replace(
        config,
        voxel_size=config.voxel_size * scale,
        normal_radius=config.normal_radius * scale,
        feature_radius=config.feature_radius * scale,
        inlier_distance=config.inlier_distance * scale,
        icp=dataclasses.replace(
            config.icp,
            max_correspondence_distance=(
                config.icp.max_correspondence_distance * scale
            ),
        ),
    )


def register_objects(
    source: ArrayLike,
    target: ArrayLike,
    config: RegistrationConfig = _DEFAULT_CONFIG,
) -> Registration:
    """Registers the points of one object onto the points of another.

    Both clouds are downsampled and the smaller one is registered onto the
    larger one, so a partial view of an object still fits the whole. The
    pose comes from a feature match refined by trimmed ICP, or from a
    centroid shift refined the same way, whichever has the higher fitness;
    fits within ``tie_tolerance`` resolve to the smaller rotation, so a
    cuboid that only moved does not come out turned by a half turn.

    Args:
        source: Source points of shape ``(N, 3)``.
        target: Target points of shape ``(M, 3)``.
        config: Downsampling, feature, RANSAC, and ICP settings.

    Returns:
        The estimated ``target_from_source`` and its fit. A registration
        that cannot be estimated, for lack of points or of ICP pairs, has
        ``succeeded`` false and no exception.

    Raises:
        ValueError: If a cloud is not a finite ``(N, 3)`` array.
    """
    raw = [
        as_finite_points(points, name, 0)
        for points, name in ((source, "source"), (target, "target"))
    ]
    config = _fitted(config, raw)
    clouds = [_downsample(cloud, config.voxel_size) for cloud in raw]
    failed = Registration(SE3.identity(), 0.0, math.inf, False)
    if min(len(c.points) for c in clouds) < config.min_points:
        return failed
    swapped = len(clouds[0].points) > len(clouds[1].points)
    small, large = clouds[::-1] if swapped else clouds
    small_points = np.asarray(small.points)
    index = TargetIndex(np.asarray(large.points))
    starts = _yaw_starts(small_points, index.points, config.yaw_starts)
    pose = _feature_match(small, large, config)
    if pose is not None:
        starts.append(pose)
    fits = [
        fit
        for start in starts
        if (fit := _refine(start, small_points, index, config)) is not None
    ]
    if not fits:
        return failed
    best = _choose(fits, config)
    if swapped:
        return dataclasses.replace(best, transform=best.transform.inverse())
    return best
