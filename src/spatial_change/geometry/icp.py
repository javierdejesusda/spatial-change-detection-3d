"""Trimmed robust point-to-point ICP between two point clouds."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3
from spatial_change.domain.transforms import as_finite_points
from spatial_change.geometry.alignment import umeyama

_MIN_PAIRS = 3
_IDENTITY = SE3.identity()


@dataclass(frozen=True)
class IcpConfig:
    """Parameters of ``trimmed_icp``.

    Attributes:
        max_iterations: Maximum number of ICP iterations.
        trim_fraction: Fraction of gated correspondences dropped each
            iteration, worst (largest distance) first.
        max_correspondence_distance: Correspondences farther than this, in
            scene units, are discarded before trimming.
        tolerance: Convergence threshold on the incremental update, in
            radians for rotation and scene units for translation.
    """

    max_iterations: int = 50
    trim_fraction: float = 0.2
    max_correspondence_distance: float = 0.25
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a parameter is outside its valid range.
        """
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0.0 <= self.trim_fraction < 1.0:
            raise ValueError("trim_fraction must be in [0, 1)")
        if not self.max_correspondence_distance > 0.0:
            raise ValueError("max_correspondence_distance must be positive")
        if not self.tolerance > 0.0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True, eq=False)
class IcpResult:
    """Outcome of ``trimmed_icp``.

    Attributes:
        transform: Estimated ``target_from_source``.
        rmse: Root mean squared distance of the kept pairs at the final
            transform.
        inlier_fraction: Fraction of the source points with a
            neighbour inside the correspondence gate at the final transform.
        iterations: Number of iterations run.
        converged: Whether the incremental update fell below the tolerance
            before the iteration cap.
    """

    transform: SE3
    rmse: float
    inlier_fraction: float
    iterations: int
    converged: bool


class TargetIndex:
    """Nearest-neighbour index over a fixed target cloud.

    Build it once to register several source clouds against one target
    without validating or indexing the target again.

    Attributes:
        points: The validated target points the index was built on.
    """

    def __init__(self, target: ArrayLike) -> None:
        """Validates the target and builds the index once.

        Args:
            target: Target points of shape ``(M, 3)``.

        Raises:
            ValueError: If the shape is wrong, there are fewer than 3
                points, or a value is not finite.
        """
        import open3d as o3d

        self.points = as_finite_points(target, "target", _MIN_PAIRS)
        self._search = o3d.core.nns.NearestNeighborSearch(
            o3d.core.Tensor.from_numpy(np.ascontiguousarray(self.points))
        )
        self._search.knn_index()

    def nearest(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Finds the nearest target point of every query point.

        Args:
            points: Query points of shape ``(N, 3)``.

        Returns:
            A tuple ``(index, distance)`` of shape ``(N,)`` each.
        """
        import open3d as o3d

        query = o3d.core.Tensor.from_numpy(np.ascontiguousarray(points))
        index, sq_distance = self._search.knn_search(query, 1)
        return index.numpy()[:, 0], np.sqrt(sq_distance.numpy()[:, 0])


def _kept_pairs(
    moved: NDArray[np.float64],
    index: TargetIndex,
    config: IcpConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], int]:
    """Gates and trims nearest-neighbour pairs.

    Returns:
        Kept source rows, their target rows, their distances, and the
        number of pairs inside the gate before trimming.
    """
    target_rows, distance = index.nearest(moved)
    gated = np.flatnonzero(distance <= config.max_correspondence_distance)
    keep = int(np.ceil((1.0 - config.trim_fraction) * len(gated)))
    if keep < _MIN_PAIRS:
        raise ValueError(
            f"fewer than 3 usable pairs ({len(gated)} inside the gate, "
            f"{keep} after trimming)"
        )
    # Select the closest pairs without sorting all of them; a mask over the
    # gated rows keeps the source order, so the result is independent of the
    # partition order.
    closest = np.argpartition(distance[gated], keep - 1)[:keep]
    selected = np.zeros(len(gated), dtype=np.bool_)
    selected[closest] = True
    best = gated[selected]
    return best, target_rows[best], distance[best], len(gated)


def _step_size(update: SE3) -> float:
    """Larger of the rotation angle (rad) and translation length."""
    rotation = update.rotation
    sin = np.linalg.norm(rotation - rotation.T) / (2.0 * np.sqrt(2.0))
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(max(np.arctan2(sin, cos), np.linalg.norm(update.translation)))


_DEFAULT_CONFIG = IcpConfig()


def trimmed_icp(
    source: ArrayLike,
    target: ArrayLike | TargetIndex,
    init: SE3 = _IDENTITY,
    config: IcpConfig = _DEFAULT_CONFIG,
) -> IcpResult:
    """Estimates the rigid transform that registers ``source`` onto ``target``.

    Point-to-point ICP: each iteration moves the source by the current
    estimate, finds nearest target neighbours in one vectorized search,
    discards pairs beyond the correspondence gate, keeps the closest
    ``1 - trim_fraction`` of the rest, and solves the incremental rigid
    update in closed form (``umeyama`` without scale). Gating and trimming
    make the fit robust to points that have no counterpart, such as moved,
    added, or removed objects.

    Args:
        source: Source points of shape ``(N, 3)``.
        target: Target points of shape ``(M, 3)``, or a ``TargetIndex`` built
            on them, which is reused instead of indexing the target again.
        init: Initial ``target_from_source`` estimate.
        config: ICP parameters.

    Returns:
        The estimated ``target_from_source`` and fit statistics.

    Raises:
        ValueError: If a shape is wrong, a value is not finite, fewer than 3
            source or target points are given, or fewer than 3
            correspondences survive the gate and trimming.
    """
    source_points = as_finite_points(source, "source", _MIN_PAIRS)
    index = target if isinstance(target, TargetIndex) else TargetIndex(target)
    estimate = init
    converged = False
    iterations = 0
    for iterations in range(1, config.max_iterations + 1):  # noqa: B007
        moved = estimate.apply(source_points)
        rows, target_rows, _, _ = _kept_pairs(moved, index, config)
        step = umeyama(moved[rows], index.points[target_rows], with_scale=False)
        update = SE3(step.rotation, step.translation)
        estimate = update.compose(estimate)
        if _step_size(update) < config.tolerance:
            converged = True
            break

    _, _, kept_distance, gated = _kept_pairs(
        estimate.apply(source_points), index, config
    )
    return IcpResult(
        transform=estimate,
        rmse=float(np.sqrt(np.mean(kept_distance**2))),
        inlier_fraction=gated / len(source_points),
        iterations=iterations,
        converged=converged,
    )
