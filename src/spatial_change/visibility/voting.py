"""Projective visibility voting of one epoch against the other epoch frames.

Each point is projected into every frame of the other epoch and compared with
that frame's depth map. The frame casts one vote per point: consistent with
the surface, free space in front of the stored surface, occluded by nearer
geometry, or unobserved. The votes decide a tri-state per point.

Torch is imported inside the functions that need it, so importing this module
does not load it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import (
    Epoch,
    EpochReconstruction,
    EvidenceCounts,
    Frame,
    Intrinsics,
)
from spatial_change.domain.transforms import as_finite_points
from spatial_change.geometry.projection import (
    DEFAULT_MIN_CONFIDENCE,
    confident_pixels,
)

if TYPE_CHECKING:
    import torch


class Vote(IntEnum):
    """Outcome of one frame for one point; the column order of the counts.

    Attributes:
        CONSISTENT: The frame's depth matches the point's depth.
        FREE: The frame sees through the point's location to farther
            surfaces.
        OCCLUDED: Nearer geometry blocks the frame's view of the point.
        UNOBSERVED: The point is outside the frustum or the depth is unusable.
    """

    CONSISTENT = 0
    FREE = 1
    OCCLUDED = 2
    UNOBSERVED = 3


class PointState(IntEnum):
    """Tri-state of a point, as stored in the states array.

    Attributes:
        UNKNOWN: The votes do not decide.
        PRESENT: The other epoch confirms the surface.
        CHANGED: The other epoch sees through the surface's location.
    """

    UNKNOWN = 0
    PRESENT = 1
    CHANGED = 2


@dataclass(frozen=True)
class VisibilityConfig:
    """Parameters of the visibility vote.

    The depth tolerance of a vote is
    ``(abs_tolerance + rel_tolerance * z) * (1 + confidence_slack * (1 - c))``
    with ``z`` the point's depth in the voting camera and ``c`` the frame's
    confidence at the pixel (1 without a confidence map). A free vote uses
    the lowest confidence of the window instead. Lengths are in the units
    of the world frame, meters after the alignment stage.

    The visibility veto is what keeps what the other epoch could not see out
    of the changes: a point is changed only on free votes, so a point that
    the other epoch saw only from behind nearer geometry (occluded), or not
    at all (unobserved), stays unknown. ``veto=False`` removes it and marks
    every point that the other epoch does not confirm as changed, as plain
    cloud differencing would. That floods the changes with objects that were
    only hidden or never seen, so it is meant for ablations that measure
    what the vote buys, not for runs.

    Attributes:
        abs_tolerance: Depth tolerance at zero depth, positive.
        rel_tolerance: Extra tolerance per unit of depth, at least 0.
        confidence_slack: How much a low-confidence pixel widens the
            tolerance, at least 0.
        min_confidence: Pixels with a lower confidence count as invalid,
            in ``[0, 1]``.
        neighbourhood: Radius of the square window whose in-image pixels
            must all be valid and show farther surfaces for a free vote, at
            least 0.
        min_free_votes: Free votes needed for a changed point, at least 1.
        min_consistent_votes: Consistent votes needed for a present point,
            at least 1.
        batch_size: Points per batch, at least 1.
        device: Torch device the computation runs on.
        veto: Whether occluded and unobserved votes keep a point from being
            changed. If ``False``, every point with fewer than
            ``min_consistent_votes`` consistent votes is changed, whatever
            else it collected.
    """

    abs_tolerance: float = 0.05
    rel_tolerance: float = 0.02
    confidence_slack: float = 1.0
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    neighbourhood: int = 1
    min_free_votes: int = 1
    min_consistent_votes: int = 1
    batch_size: int = 262_144
    device: str = "cpu"
    veto: bool = True

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a tolerance is not finite or out of range, the
                confidence threshold is outside ``[0, 1]``, a count is out
                of range, or the device is empty.
        """
        if not 0.0 < self.abs_tolerance < math.inf:
            raise ValueError("abs_tolerance must be positive and finite")
        if not 0.0 <= self.rel_tolerance < math.inf:
            raise ValueError("rel_tolerance must be >= 0 and finite")
        if not 0.0 <= self.confidence_slack < math.inf:
            raise ValueError("confidence_slack must be >= 0 and finite")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.neighbourhood < 0:
            raise ValueError("neighbourhood must be >= 0")
        if self.min_free_votes < 1 or self.min_consistent_votes < 1:
            raise ValueError(
                "min_free_votes and min_consistent_votes must be >= 1"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not self.device:
            raise ValueError("device must be a non-empty torch device string")


_DEFAULT_CONFIG = VisibilityConfig()


@dataclass(frozen=True, eq=False)
class PointEvidence:
    """Votes and tri-state of a set of points.

    Attributes:
        counts: Vote counts of shape ``(N, 4)``, int32, in ``Vote`` order.
        states: ``PointState`` codes of shape ``(N,)``, int8.
    """

    counts: NDArray[np.int32]
    states: NDArray[np.int8]

    @property
    def consistent(self) -> NDArray[np.int32]:
        """Consistent votes per point."""
        return self.counts[:, Vote.CONSISTENT]

    @property
    def free(self) -> NDArray[np.int32]:
        """Free votes per point."""
        return self.counts[:, Vote.FREE]

    @property
    def occluded(self) -> NDArray[np.int32]:
        """Occluded votes per point."""
        return self.counts[:, Vote.OCCLUDED]

    @property
    def unobserved(self) -> NDArray[np.int32]:
        """Unobserved votes per point."""
        return self.counts[:, Vote.UNOBSERVED]

    def total(self, indices: NDArray[np.int64]) -> EvidenceCounts:
        """Sums the votes of some points.

        Args:
            indices: Rows of the points to sum, shape ``(K,)``; may be empty.

        Returns:
            The vote counts summed over those rows, all zeros for none.
        """
        sums = self.counts[indices].sum(axis=0, dtype=np.int64)
        return EvidenceCounts(
            consistent=int(sums[Vote.CONSISTENT]),
            free=int(sums[Vote.FREE]),
            occluded=int(sums[Vote.OCCLUDED]),
            unobserved=int(sums[Vote.UNOBSERVED]),
        )


@dataclass(frozen=True)
class _FrameTensors:
    """A frame's depth data and pose on the voting device.

    Attributes:
        intrinsics: The frame's camera intrinsics.
        depth: Flat ``(H * W,)`` depth, 0 where invalid.
        pooled: Flat ``(H * W,)`` minimum of ``depth`` over the window.
        factor: Flat ``(H * W,)`` tolerance factor
            ``1 + confidence_slack * (1 - c)`` with ``c`` the confidence
            clipped to ``[0, 1]``.
        free_factor: Flat ``(H * W,)`` tolerance factor of a free vote, from
            the lowest ``c`` over the window. A window with an invalid pixel
            has ``pooled`` 0, casts no free vote, and never reads it.
        rotation: Camera-from-world rotation transposed, ``(3, 3)``.
        center: Camera center relative to the point origin, ``(3,)``.
    """

    intrinsics: Intrinsics
    depth: torch.Tensor
    pooled: torch.Tensor
    factor: torch.Tensor
    free_factor: torch.Tensor
    rotation: torch.Tensor
    center: torch.Tensor


def _window_min(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Takes the minimum of a map over a square window around each pixel.

    The minimum is separable: it runs over the horizontal shifts, then over
    the vertical ones. Pixels outside the image are left out.

    Args:
        image: ``(H, W)`` float map; a NaN pixel makes the minimum of every
            window that contains it NaN.
        radius: Window radius in pixels, at least 0.

    Returns:
        The ``(H, W)`` window minima; ``image`` itself for a radius of 0.
    """
    import torch
    import torch.nn.functional as functional

    if radius == 0:
        return image
    height, width = image.shape
    padded = functional.pad(
        image, (radius, radius, radius, radius), value=math.inf
    )
    rows = padded[:, :width]
    for shift in range(1, 2 * radius + 1):
        rows = torch.minimum(rows, padded[:, shift : shift + width])
    result = rows[:height]
    for shift in range(1, 2 * radius + 1):
        result = torch.minimum(result, rows[shift : shift + height])
    return result


def _frame_tensors(
    frame: Frame,
    origin: NDArray[np.float64],
    config: VisibilityConfig,
    device: torch.device,
) -> _FrameTensors:
    """Moves a frame's maps, their window minima, and its pose to a device.

    Args:
        frame: Frame with a depth map; no confidence map counts as 1.
        origin: Point mean that the camera center is taken relative to.
        config: Confidence threshold, tolerance slack, and window radius.
        device: Torch device of the vote.

    Returns:
        The flattened maps and the float32 pose.
    """
    import torch

    assert frame.depth is not None
    raw = (
        np.ones_like(frame.depth)
        if frame.confidence is None
        else frame.confidence
    )
    valid = torch.from_numpy(
        confident_pixels(frame.depth, raw, config.min_confidence)
    ).to(device)
    depth = torch.from_numpy(frame.depth).to(device).masked_fill(~valid, 0.0)
    confidence = torch.from_numpy(raw).to(device).clamp(0.0, 1.0)
    slack = config.confidence_slack
    radius = config.neighbourhood
    pose = frame.world_from_camera
    return _FrameTensors(
        intrinsics=frame.intrinsics,
        depth=depth.reshape(-1),
        pooled=_window_min(depth, radius).reshape(-1),
        factor=(1.0 + slack * (1.0 - confidence)).reshape(-1),
        free_factor=(
            1.0 + slack * (1.0 - _window_min(confidence, radius))
        ).reshape(-1),
        rotation=torch.from_numpy(pose.rotation.astype(np.float32)).to(device),
        center=torch.from_numpy(
            (pose.translation - origin).astype(np.float32)
        ).to(device),
    )


def _vote_batch(
    points: torch.Tensor,
    tensors: _FrameTensors,
    config: VisibilityConfig,
) -> torch.Tensor:
    """Casts one frame's vote for every point of a batch.

    Args:
        points: ``(B, 3)`` float32 points relative to the point origin.
        tensors: The frame's maps, intrinsics, and pose from
            ``_frame_tensors``.
        config: Tolerances.

    Returns:
        ``(B,)`` int64 ``Vote`` codes.
    """
    import torch

    intrinsics = tensors.intrinsics
    camera = (points - tensors.center) @ tensors.rotation
    z = camera[:, 2]
    u = intrinsics.fx * camera[:, 0] / z + intrinsics.cx
    v = intrinsics.fy * camera[:, 1] / z + intrinsics.cy
    inside = (
        (z > 0.0)
        & (u >= -0.5)
        & (u < intrinsics.width - 0.5)
        & (v >= -0.5)
        & (v < intrinsics.height - 0.5)
    )
    col = torch.round(torch.where(inside, u, 0.0)).long()
    row = torch.round(torch.where(inside, v, 0.0)).long()
    flat = row * intrinsics.width + col
    d = tensors.depth[flat]
    observed = inside & (d > 0.0)
    scale = config.abs_tolerance + config.rel_tolerance * z
    tau = scale * tensors.factor[flat]
    free_tau = scale * tensors.free_factor[flat]
    consistent = observed & ((z - d).abs() <= tau)
    occluded = observed & (d < z - tau)
    free = observed & (tensors.pooled[flat] > z + free_tau)
    votes = torch.full_like(flat, Vote.UNOBSERVED)
    votes[occluded] = Vote.OCCLUDED
    votes[free] = Vote.FREE
    votes[consistent] = Vote.CONSISTENT
    return votes


def _states(
    counts: NDArray[np.int32], config: VisibilityConfig
) -> NDArray[np.int8]:
    """Decides the tri-state of every point from its vote counts.

    Args:
        counts: ``(N, 4)`` vote counts in ``Vote`` order.
        config: Vote thresholds and whether the visibility veto applies.

    Returns:
        ``(N,)`` int8 ``PointState`` codes.
    """
    consistent = counts[:, Vote.CONSISTENT]
    free = counts[:, Vote.FREE]
    states = np.full(len(counts), PointState.UNKNOWN, dtype=np.int8)
    states[consistent >= config.min_consistent_votes] = PointState.PRESENT
    changed = (free >= config.min_free_votes) & (free > consistent)
    states[changed] = PointState.CHANGED
    if not config.veto:
        states[states == PointState.UNKNOWN] = PointState.CHANGED
    return states


def vote_points(
    points: ArrayLike,
    frames: Sequence[Frame],
    config: VisibilityConfig = _DEFAULT_CONFIG,
) -> PointEvidence:
    """Votes on points by projecting them into frames with depth.

    Every frame casts one vote per point. The point is moved into the
    camera frame and projected; outside the image or behind the camera it
    is unobserved, as it is when the center pixel is invalid. Otherwise the
    stored depth ``d`` is compared with the point's depth ``z`` within the
    tolerance of ``VisibilityConfig``: close is consistent, nearer is
    occluded, and farther is free if every in-image pixel of the window
    around the pixel is valid and farther too, within the tolerance of the
    window's lowest confidence (so a point on a depth edge does not see
    through its own silhouette), else unobserved.

    Points and camera centers are shifted by the float64 mean of the points
    before the float32 computation, so precision does not depend on where
    the room sits in the world.

    Args:
        points: World points of shape ``(N, 3)``; may be empty.
        frames: Posed frames, each with a depth map, at least one.
        config: Tolerances, thresholds, batching, and device.

    Returns:
        The vote counts and tri-state of every point.

    Raises:
        ValueError: If ``points`` is not a finite ``(N, 3)`` array, there is
            no frame, or a frame has no depth map.
    """
    import torch

    cloud = as_finite_points(points, "points", 0)
    if not frames:
        raise ValueError("vote_points needs at least one frame")
    if any(frame.depth is None for frame in frames):
        raise ValueError("every frame needs a depth map")
    count = len(cloud)
    if count == 0:
        return PointEvidence(
            np.zeros((0, len(Vote)), dtype=np.int32),
            np.zeros(0, dtype=np.int8),
        )
    device = torch.device(config.device)
    origin = cloud.mean(axis=0)
    local = torch.from_numpy((cloud - origin).astype(np.float32)).to(device)
    counts = torch.zeros((count, len(Vote)), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for frame in frames:
            tensors = _frame_tensors(frame, origin, config, device)
            for start in range(0, count, config.batch_size):
                stop = min(start + config.batch_size, count)
                votes = _vote_batch(local[start:stop], tensors, config)[:, None]
                counts[start:stop].scatter_add_(
                    1, votes, torch.ones_like(votes, dtype=torch.int32)
                )
    result = counts.cpu().numpy()
    return PointEvidence(result, _states(result, config))


@dataclass(frozen=True, eq=False)
class CrossEvidence:
    """Visibility evidence of both epochs against each other.

    Attributes:
        before: The BEFORE points voted by the AFTER frames; a free vote
            means the surface disappeared.
        after: The AFTER points voted by the BEFORE frames; a free vote
            means the surface appeared.
    """

    before: PointEvidence
    after: PointEvidence


def vote_epochs(
    before: EpochReconstruction,
    after: EpochReconstruction,
    config: VisibilityConfig = _DEFAULT_CONFIG,
) -> CrossEvidence:
    """Votes each epoch's points against the other epoch's frames.

    Both reconstructions must share one world frame.

    Args:
        before: The BEFORE reconstruction, with depth in its frames.
        after: The AFTER reconstruction, with depth in its frames.
        config: Tolerances, thresholds, batching, and device.

    Returns:
        The evidence for the BEFORE points and for the AFTER points.

    Raises:
        ValueError: If ``before`` is not the BEFORE epoch, ``after`` is not
            the AFTER epoch, or ``vote_points`` rejects the inputs.
    """
    if before.epoch != Epoch.BEFORE:
        raise ValueError("before must be the BEFORE reconstruction")
    if after.epoch != Epoch.AFTER:
        raise ValueError("after must be the AFTER reconstruction")
    return CrossEvidence(
        before=vote_points(before.points, after.frames, config),
        after=vote_points(after.points, before.frames, config),
    )
