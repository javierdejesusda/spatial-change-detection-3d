"""Change-aware registration of two epochs on their static structure."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import SE3
from spatial_change.domain.transforms import as_finite_points, check_array
from spatial_change.geometry.icp import (
    IcpConfig,
    IcpResult,
    TargetIndex,
    trimmed_icp,
)

_MIN_POINTS = 3
# The coarse pass is skipped when either strided cloud would hold fewer
# points: on small clouds it saves nothing and only adds a registration.
_COARSE_MIN_POINTS = 1000


@dataclass(frozen=True)
class RefineConfig:
    """Parameters of ``refine_alignment``.

    Attributes:
        icp: Parameters of each ICP round.
        change_threshold: A source point farther than this from the target
            after alignment is a change candidate, in scene units.
        max_rounds: Maximum number of ICP rounds.
        coarse_stride: Before the first round, one ICP on every
            ``coarse_stride``-th point of both clouds gives the first round a
            close start. 1 disables the coarse pass; it is also skipped
            when either strided cloud would hold fewer than 1000 points.
    """

    icp: IcpConfig = IcpConfig()
    change_threshold: float = 0.05
    max_rounds: int = 5
    coarse_stride: int = 8

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a parameter is outside its valid range.
        """
        if not self.change_threshold > 0.0:
            raise ValueError("change_threshold must be positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if self.coarse_stride < 1:
            raise ValueError("coarse_stride must be at least 1")


@dataclass(frozen=True, eq=False)
class Refinement:
    """Outcome of ``refine_alignment``.

    Attributes:
        static_mask: Per source point, True where the point is static under
            ``transform`` (not excluded and within the change threshold of
            the target).
        rounds: Number of ICP rounds run.
        icp: Result of the last ICP round.
    """

    static_mask: NDArray[np.bool_]
    rounds: int
    icp: IcpResult

    @property
    def transform(self) -> SE3:
        """Estimated ``target_from_source``."""
        return self.icp.transform


_IDENTITY = SE3.identity()
_DEFAULT_CONFIG = RefineConfig()


def refine_alignment(
    source: ArrayLike,
    target: ArrayLike,
    init: SE3 = _IDENTITY,
    config: RefineConfig = _DEFAULT_CONFIG,
    exclude: NDArray[np.bool_] | None = None,
) -> Refinement:
    """Registers ``source`` onto ``target`` while ignoring changed points.

    Moved, added, or removed objects have no counterpart in the other epoch
    and pull a plain registration toward them. Each round fits a trimmed ICP
    on the current static points, then marks every source point farther than
    ``config.change_threshold`` from the target as a change candidate. The
    next static set is everything that is neither excluded nor a candidate.
    The loop stops when the static set stops changing or after
    ``config.max_rounds`` rounds. On large clouds a coarse ICP on strided
    subsamples runs first and seeds the first round (see
    ``RefineConfig.coarse_stride``).

    Args:
        source: Source points of shape ``(N, 3)``.
        target: Target points of shape ``(M, 3)``.
        init: Initial ``target_from_source`` estimate.
        config: Refinement parameters.
        exclude: Optional ``(N,)`` bool mask of source points to ignore, for
            example change candidates from a later stage. None excludes
            nothing.

    Returns:
        The refined ``target_from_source`` and the final static mask.

    Raises:
        ValueError: If a cloud is malformed or not finite, ``exclude`` has
            the wrong shape or dtype, or the static set becomes too small
            for ICP.
    """
    source_points = as_finite_points(source, "source", _MIN_POINTS)
    if exclude is None:
        allowed = np.ones(len(source_points), dtype=np.bool_)
    else:
        check_array(exclude, (len(source_points),), np.bool_, "exclude")
        allowed = ~exclude

    index = TargetIndex(target)
    estimate = init
    stride = config.coarse_stride
    coarse_rows = np.flatnonzero(allowed)[::stride]
    coarse_target = index.points[::stride]
    coarse_size = min(len(coarse_rows), len(coarse_target))
    if stride > 1 and coarse_size >= _COARSE_MIN_POINTS:
        coarse = trimmed_icp(
            source_points[coarse_rows],
            TargetIndex(coarse_target),
            estimate,
            config.icp,
        )
        estimate = coarse.transform

    static = allowed
    for rounds in range(1, config.max_rounds + 1):  # noqa: B007
        icp = trimmed_icp(source_points[static], index, estimate, config.icp)
        estimate = icp.transform
        _, distance = index.nearest(estimate.apply(source_points))
        updated = allowed & (distance <= config.change_threshold)
        stable = np.array_equal(updated, static)
        static = updated
        if stable:
            break
    return Refinement(static, rounds, icp)
