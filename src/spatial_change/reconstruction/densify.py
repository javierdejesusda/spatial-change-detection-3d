"""Chunked densification of an epoch around the joint anchor pass.

The joint pass reconstructs a few keyframes of both epochs in one world
frame. The remaining frames of each epoch are reconstructed in chunks that
also contain some anchor frames. A chunk lives in its own model-defined
world, so a Sim3 estimated from the anchors' point maps (seen once in the
joint pass and once in the chunk) moves the chunk into the anchor world.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.capture import VideoFrame
from spatial_change.domain import Frame, Sim3
from spatial_change.geometry.alignment import umeyama
from spatial_change.geometry.projection import (
    DEFAULT_MIN_CONFIDENCE,
    confident_pixels,
    unproject_depth,
)
from spatial_change.ports import ReconstructionResult, Reconstructor
from spatial_change.reconstruction.joint import (
    frame_from_result,
    reconstruct_views,
)


@dataclass(frozen=True)
class DensifyConfig:
    """Parameters of the chunked densification.

    Attributes:
        chunk_size: New (non-anchor) frames per reconstructor call.
        anchors_per_chunk: Anchor frames included in every chunk, at least 1.
        min_confidence: Per-pixel confidence threshold for alignment only,
            in ``[0, 1]``; ``fuse_frames`` takes its own threshold. The Pi3X
            adapter returns a sigmoid.
        trim_fraction: Fraction of the largest-residual correspondences
            dropped in each robust refit, in ``[0, 1)``.
        trim_iterations: Number of trimmed refits after the initial fit.
        max_alignment_points: Correspondences per chunk after a
            deterministic subsample, at least 3.
        min_alignment_points: Fewer confident correspondences than this
            raise ``ValueError``, at least 3.
        ratio_gate: Correspondences whose anchor-to-chunk depth ratio is
            more than this factor away from the median ratio are dropped
            before the fit, above 1. The same pixel has
            ``anchor_depth = scale * chunk_depth`` when the two
            reconstructions differ by a similarity, so the median ratio is
            a robust scale estimate. A fitted scale outside the gate around
            the median raises ``ValueError``.
    """

    chunk_size: int = 32
    anchors_per_chunk: int = 4
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    trim_fraction: float = 0.2
    trim_iterations: int = 3
    max_alignment_points: int = 20000
    min_alignment_points: int = 100
    ratio_gate: float = 1.5

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a count is out of range, a fraction or the
                confidence threshold is outside its interval, or the
                minimum number of alignment points exceeds the maximum.
        """
        if self.chunk_size < 1 or self.anchors_per_chunk < 1:
            raise ValueError("chunk_size and anchors_per_chunk must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if not 0.0 <= self.trim_fraction < 1.0:
            raise ValueError("trim_fraction must be within [0, 1)")
        if self.trim_iterations < 0:
            raise ValueError("trim_iterations must be >= 0")
        if self.min_alignment_points < 3:
            raise ValueError("min_alignment_points must be >= 3")
        if self.max_alignment_points < self.min_alignment_points:
            raise ValueError(
                "max_alignment_points must be >= min_alignment_points"
            )
        if not self.ratio_gate > 1.0:
            raise ValueError("ratio_gate must be > 1")


@dataclass(frozen=True, eq=False)
class ChunkAlignment:
    """How one chunk was moved into the anchor world.

    Attributes:
        frame_indices: Source indices of the new frames in the chunk.
        anchor_indices: Source indices of the anchors included in the chunk.
        transform: Similarity ``anchor_world_from_chunk_world``.
        rmse: Root mean square distance between the transformed chunk
            points and the anchor points over the kept correspondences.
        point_count: Number of kept correspondences.
    """

    frame_indices: list[int]
    anchor_indices: list[int]
    transform: Sim3
    rmse: float
    point_count: int


@dataclass(frozen=True, eq=False)
class DensifiedEpoch:
    """All frames of one epoch in the anchor world.

    Attributes:
        frames: Anchor frames and new frames, sorted by source index.
        chunks: One alignment per reconstructor call, in call order.
    """

    frames: list[Frame]
    chunks: list[ChunkAlignment]


_DEFAULT_CONFIG = DensifyConfig()


def densify_epoch(
    reconstructor: Reconstructor,
    anchor_videos: Sequence[VideoFrame],
    anchor_frames: Sequence[Frame],
    others: Sequence[VideoFrame],
    config: DensifyConfig = _DEFAULT_CONFIG,
) -> DensifiedEpoch:
    """Reconstructs the remaining frames of an epoch around its anchors.

    The frames in ``others`` are sorted by source index and split into
    consecutive groups of at most ``config.chunk_size``. Each group is
    reconstructed together with the anchors nearest to it, and the result
    is moved into the anchor world with a Sim3 fitted on the anchors. The
    reconstructor is not unloaded; that stays with the caller.

    Args:
        reconstructor: Model that predicts poses, intrinsics, and depth.
        anchor_videos: Input keyframes of the anchors.
        anchor_frames: Joint pass output of the same epoch, in the same
            order as ``anchor_videos``.
        others: Frames of the epoch to add.
        config: Chunking and alignment parameters.

    Returns:
        The anchor frames, unchanged, plus the new frames in the anchor
        world, sorted by source index, and the alignment of each chunk.

    Raises:
        ValueError: If there are no anchors, the anchors disagree in count,
            source index, or epoch, a source index appears twice, an anchor
            lacks depth or confidence, the reconstructor returns the wrong
            number of views or a depth map of another shape than the joint
            pass, or a chunk cannot be aligned: it has fewer confident
            correspondences than ``config.min_alignment_points`` left after
            the ratio gate, its correspondences are degenerate (collinear
            or coincident), or the fitted scale contradicts the depth
            ratios. The message names the chunk's new frames and anchors.
    """
    _check_inputs(anchor_videos, anchor_frames, others)
    anchors = list(zip(anchor_videos, anchor_frames, strict=True))
    epoch = anchor_frames[0].epoch
    ordered = sorted(others, key=lambda video_frame: video_frame.index)
    frames = [frame for _, frame in anchors]
    chunks = []
    for start in range(0, len(ordered), config.chunk_size):
        group = ordered[start : start + config.chunk_size]
        chosen = _nearest_anchors(
            anchors, group[0].index, group[-1].index, config.anchors_per_chunk
        )
        frame_indices = [video.index for video in group]
        anchor_indices = [video.index for video, _ in chosen]
        images = [video.image for video, _ in chosen]
        images += [video.image for video in group]
        result = reconstruct_views(reconstructor, images)
        try:
            transform, rmse, count = _align_chunk(
                [frame for _, frame in chosen], result, config
            )
        except ValueError as error:
            raise ValueError(
                f"chunk with new frames {frame_indices} and anchors "
                f"{anchor_indices}: {error}"
            ) from error
        for j, video in enumerate(group):
            frame = frame_from_result(epoch, video, result, len(chosen) + j)
            frames.append(frame.transformed(transform))
        chunks.append(
            ChunkAlignment(
                frame_indices=frame_indices,
                anchor_indices=anchor_indices,
                transform=transform,
                rmse=rmse,
                point_count=count,
            )
        )
    frames.sort(key=lambda frame: frame.index)
    return DensifiedEpoch(frames, chunks)


def _check_inputs(
    anchor_videos: Sequence[VideoFrame],
    anchor_frames: Sequence[Frame],
    others: Sequence[VideoFrame],
) -> None:
    if not anchor_frames:
        raise ValueError("need at least one anchor frame")
    if [v.index for v in anchor_videos] != [f.index for f in anchor_frames]:
        raise ValueError(
            "anchor_videos and anchor_frames differ in length or order"
        )
    if any(frame.epoch != anchor_frames[0].epoch for frame in anchor_frames):
        raise ValueError("all anchor frames must belong to one epoch")
    for frame in anchor_frames:
        _depth_and_confidence(frame)
    indices = [video.index for video in (*anchor_videos, *others)]
    if len(set(indices)) != len(indices):
        raise ValueError("source indices must be unique")


def _depth_and_confidence(
    frame: Frame,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Returns the depth and confidence maps of an anchor frame.

    Raises:
        ValueError: If the frame lacks depth or confidence.
    """
    if frame.depth is None or frame.confidence is None:
        raise ValueError(
            f"anchor frame {frame.index} needs depth and confidence"
        )
    return frame.depth, frame.confidence


def _nearest_anchors(
    anchors: Sequence[tuple[VideoFrame, Frame]],
    first: int,
    last: int,
    count: int,
) -> list[tuple[VideoFrame, Frame]]:
    """Picks the anchors closest to the index range ``[first, last]``.

    Anchors inside the range have distance 0, and ties go to the lower
    index. The result is ordered by source index.
    """

    def distance(anchor: tuple[VideoFrame, Frame]) -> tuple[int, int]:
        index = anchor[0].index
        return max(first - index, index - last, 0), index

    return sorted(
        sorted(anchors, key=distance)[:count], key=lambda a: a[0].index
    )


def _align_chunk(
    anchors: Sequence[Frame],
    result: ReconstructionResult,
    config: DensifyConfig,
) -> tuple[Sim3, float, int]:
    """Fits ``anchor_world_from_chunk_world`` on the chunk's anchor views."""
    chunk_points, anchor_points, log_ratios = [], [], []
    for i, anchor in enumerate(anchors):
        anchor_depth, anchor_confidence = _depth_and_confidence(anchor)
        depth, confidence = result.depth[i], result.confidence[i]
        if depth.shape != anchor_depth.shape:
            raise ValueError(
                f"anchor {anchor.index} depth shape {depth.shape} differs "
                f"from the joint pass shape {anchor_depth.shape}"
            )
        valid = confident_pixels(
            depth, confidence, config.min_confidence
        ) & confident_pixels(
            anchor_depth, anchor_confidence, config.min_confidence
        )
        chunk_points.append(
            unproject_depth(
                depth, result.intrinsics[i], result.world_from_camera[i], valid
            )
        )
        anchor_points.append(
            unproject_depth(
                anchor_depth, anchor.intrinsics, anchor.world_from_camera, valid
            )
        )
        log_ratios.append(
            np.log(
                anchor_depth[valid].astype(np.float64)
                / depth[valid].astype(np.float64)
            )
        )
    log_ratio = np.concatenate(log_ratios)
    if len(log_ratio) == 0:
        raise ValueError(
            f"0 confident correspondences, need at least "
            f"{config.min_alignment_points}"
        )
    log_median = float(np.median(log_ratio))
    gated = np.flatnonzero(
        np.abs(log_ratio - log_median) <= math.log(config.ratio_gate)
    )
    if len(gated) < config.min_alignment_points:
        raise ValueError(
            f"{len(gated)} confident correspondences, need at least "
            f"{config.min_alignment_points}"
        )
    if len(gated) > config.max_alignment_points:
        picked = np.linspace(0, len(gated) - 1, config.max_alignment_points)
        gated = gated[picked.round().astype(np.intp)]
    transform, rmse, count = _fit_trimmed(
        np.concatenate(chunk_points)[gated],
        np.concatenate(anchor_points)[gated],
        config,
    )
    median_ratio = math.exp(log_median)
    low = median_ratio / config.ratio_gate
    high = median_ratio * config.ratio_gate
    if not low <= transform.scale <= high:
        raise ValueError(
            f"fitted scale {transform.scale:.4g} is outside the ratio gate "
            f"[{low:.4g}, {high:.4g}] around the median depth ratio "
            f"{median_ratio:.4g}"
        )
    return transform, rmse, count


def _fit_trimmed(
    src: NDArray[np.float64], dst: NDArray[np.float64], config: DensifyConfig
) -> tuple[Sim3, float, int]:
    keep = np.arange(len(src))
    count = max(3, round((1.0 - config.trim_fraction) * len(src)))
    transform = umeyama(src, dst, with_scale=True)
    for _ in range(config.trim_iterations):
        residuals = np.linalg.norm(transform.apply(src) - dst, axis=1)
        keep = np.argsort(residuals, kind="stable")[:count]
        transform = umeyama(src[keep], dst[keep], with_scale=True)
    residuals = np.linalg.norm(transform.apply(src[keep]) - dst[keep], axis=1)
    return transform, float(np.sqrt(np.mean(residuals**2))), len(keep)
