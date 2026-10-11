"""Appearance cue: do two views of the same surface still look alike.

Geometry alone cannot see a surface that kept its shape but changed its look,
such as a repainted wall or a swapped cover. Each epoch's points are binned
into voxels. Every frame of both epochs is encoded once into patch features,
and a voxel collects the feature at its projection from each frame that sees
the surface there. The cosine between the voxel's features in its own epoch
and in the other epoch is its appearance similarity. PRESENT points whose
similarity is low are then marked CHANGED.

Importing this module does not load torch.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import Epoch, EpochReconstruction, Frame
from spatial_change.domain.transforms import as_finite_points
from spatial_change.geometry.projection import (
    DEFAULT_MIN_CONFIDENCE,
    confident_pixels,
    in_image,
    project,
)
from spatial_change.perception.sampling import sample_features
from spatial_change.ports import FeatureEncoder
from spatial_change.visibility import CrossEvidence, PointEvidence, PointState

_MIN_NORM = 1e-12


@dataclass(frozen=True)
class AppearanceConfig:
    """Parameters of the appearance comparison and its fusion.

    The depth test of a view is ``|z - d| <= abs_tolerance + rel_tolerance *
    z``, with ``z`` the voxel's depth in the frame and ``d`` the frame's depth
    at its pixel. Lengths are in the units of the world frame, meters after
    the alignment stage.

    Attributes:
        voxel_size: Edge of the voxels whose appearance is compared,
            positive and finite.
        abs_tolerance: Depth tolerance at zero depth, positive and finite.
        rel_tolerance: Extra tolerance per unit of depth, at least 0 and
            finite.
        min_confidence: Pixels with a lower confidence give no view, in
            ``[0, 1]``.
        min_views: Views each side needs before a voxel gets a similarity,
            at least 1.
        changed_below: PRESENT points with a cosine below this become
            CHANGED, in ``[-1, 1]``. Provisional: it is calibrated on real
            captures at T22, so treat the default as a placeholder.
    """

    voxel_size: float = 0.05
    abs_tolerance: float = 0.05
    rel_tolerance: float = 0.02
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_views: int = 2
    changed_below: float = 0.4

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a length is not positive and finite, a tolerance
                is negative or not finite, the confidence or the cosine
                threshold is out of range, or ``min_views`` is below 1.
        """
        if not 0.0 < self.voxel_size < math.inf:
            raise ValueError("voxel_size must be positive and finite")
        if not 0.0 < self.abs_tolerance < math.inf:
            raise ValueError("abs_tolerance must be positive and finite")
        if not 0.0 <= self.rel_tolerance < math.inf:
            raise ValueError("rel_tolerance must be >= 0 and finite")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.min_views < 1:
            raise ValueError("min_views must be >= 1")
        if not -1.0 <= self.changed_below <= 1.0:
            raise ValueError("changed_below must be within [-1, 1]")


_DEFAULT_CONFIG = AppearanceConfig()


@dataclass(frozen=True, eq=False)
class PointAppearance:
    """Appearance similarity of a set of points.

    Attributes:
        similarity: Cosine between the point's features in its own epoch
            and in the other epoch, shape ``(N,)``, float32. NaN where
            either side has fewer than ``min_views`` views.
        own_views: Frames of the point's own epoch that saw it, shape
            ``(N,)``, int32.
        other_views: Frames of the other epoch that saw it, shape ``(N,)``,
            int32.
    """

    similarity: NDArray[np.float32]
    own_views: NDArray[np.int32]
    other_views: NDArray[np.int32]


@dataclass(frozen=True, eq=False)
class CrossAppearance:
    """Appearance similarity of both epochs' points.

    Attributes:
        before: The BEFORE points, compared with the AFTER frames.
        after: The AFTER points, compared with the BEFORE frames.
    """

    before: PointAppearance
    after: PointAppearance


@dataclass
class _Views:
    """Summed unit features and view count of the voxels from one epoch.

    Attributes:
        total: ``(V, C)`` sum of the unit features, allocated by the first
            view because the channel count is only known after encoding.
        count: ``(V,)`` number of views.
    """

    total: NDArray[np.float32] | None
    count: NDArray[np.int32]


@dataclass
class _Voxels:
    """The occupied voxels of one epoch with their collected views.

    Attributes:
        point_voxel: Voxel row of every point, shape ``(N,)``.
        centers: Mean of the points in each voxel, shape ``(V, 3)``.
        views: Views collected from the frames of each epoch.
    """

    point_voxel: NDArray[np.intp]
    centers: NDArray[np.float64]
    views: dict[Epoch, _Views]


def _voxelize(points: NDArray[np.float64], voxel_size: float) -> _Voxels:
    """Bins points into voxels and represents each by its mean point.

    Args:
        points: World points of shape ``(N, 3)``.
        voxel_size: Voxel edge.

    Returns:
        The voxels, with no views collected yet.
    """
    cells = np.floor(points / voxel_size).astype(np.int64)
    occupied, inverse = np.unique(cells, axis=0, return_inverse=True)
    point_voxel = inverse.reshape(-1).astype(np.intp)
    count = len(occupied)
    sizes = np.bincount(point_voxel, minlength=count)
    centers = np.stack(
        [
            np.bincount(point_voxel, weights=points[:, k], minlength=count)
            / np.maximum(sizes, 1)
            for k in range(3)
        ],
        axis=1,
    )
    return _Voxels(
        point_voxel,
        centers,
        {epoch: _Views(None, np.zeros(count, np.int32)) for epoch in Epoch},
    )


def _usable_pixels(frame: Frame, config: AppearanceConfig) -> NDArray[np.bool_]:
    """Marks the pixels that can give a view.

    A frame without a confidence map counts every pixel as fully confident.

    Args:
        frame: A frame with depth.
        config: Supplies ``min_confidence``.

    Returns:
        ``(H, W)`` mask of the pixels with a valid depth and enough
        confidence.
    """
    assert frame.depth is not None
    confidence = (
        np.ones_like(frame.depth)
        if frame.confidence is None
        else frame.confidence
    )
    return confident_pixels(frame.depth, confidence, config.min_confidence)


def _add_views(
    views: _Views,
    centers: NDArray[np.float64],
    frame: Frame,
    usable: NDArray[np.bool_],
    features: NDArray[np.float32],
    patch_size: int,
    config: AppearanceConfig,
) -> None:
    """Adds the features of the voxels a frame sees to one epoch's views.

    A voxel is seen when its center projects into the image, the pixel is
    usable, and the frame's depth there matches the center's depth, so a
    voxel hidden behind a nearer surface or floating in front of a farther
    one gets no view.

    Args:
        views: The views to add to.
        centers: Voxel centers of shape ``(V, 3)``.
        frame: The frame, with depth.
        usable: The frame's usable pixels, from ``_usable_pixels``.
        features: The frame's patch features.
        patch_size: Side of a feature patch in pixels.
        config: Supplies the depth tolerances.
    """
    assert frame.depth is not None
    uv, z = project(centers, frame.intrinsics, frame.world_from_camera)
    visible = np.flatnonzero(in_image(uv, z, frame.intrinsics))
    uv, z = uv[visible], z[visible]
    pixels = np.rint(uv).astype(np.intp)
    rows, cols = pixels[:, 1], pixels[:, 0]
    depth_error = np.abs(z - frame.depth[rows, cols])
    tolerance = config.abs_tolerance + config.rel_tolerance * z
    seen = usable[rows, cols] & (depth_error <= tolerance)
    voxels = visible[seen]
    if voxels.size == 0:
        return
    sampled = sample_features(features, uv[seen], patch_size)
    norms = np.sqrt(np.einsum("vc,vc->v", sampled, sampled))
    sampled /= np.maximum(norms, _MIN_NORM)[:, None]
    if views.total is None:
        views.total = np.zeros((len(centers), sampled.shape[1]), np.float32)
    # A frame sees each voxel at most once, so the rows are unique.
    views.total[voxels] += sampled
    views.count[voxels] += 1


def _similarity(
    voxels: _Voxels, config: AppearanceConfig
) -> NDArray[np.float32]:
    """Computes the cosine between the BEFORE and the AFTER summed features.

    Args:
        voxels: Voxels with their collected views.
        config: Supplies ``min_views``.

    Returns:
        ``(V,)`` float32 cosines, NaN under ``min_views`` on either side.
    """
    before, after = voxels.views[Epoch.BEFORE], voxels.views[Epoch.AFTER]
    count = len(voxels.centers)
    if before.total is None or after.total is None:
        return np.full(count, np.nan, np.float32)
    dot = np.einsum("vc,vc->v", before.total, after.total)
    norms = np.linalg.norm(before.total, axis=1) * np.linalg.norm(
        after.total, axis=1
    )
    similarity = dot / np.maximum(norms, _MIN_NORM)
    enough = (before.count >= config.min_views) & (
        after.count >= config.min_views
    )
    return np.where(enough, similarity, np.nan).astype(np.float32)


def _point_appearance(
    voxels: _Voxels, epoch: Epoch, config: AppearanceConfig
) -> PointAppearance:
    """Gives every point the values of its voxel.

    Args:
        voxels: Voxels of one epoch with their collected views.
        epoch: The epoch of those voxels, whose views count as own.
        config: Supplies ``min_views``.

    Returns:
        The similarity and the own and other view counts per point.
    """
    rows = voxels.point_voxel
    other = Epoch.AFTER if epoch == Epoch.BEFORE else Epoch.BEFORE
    return PointAppearance(
        similarity=_similarity(voxels, config)[rows],
        own_views=voxels.views[epoch].count[rows],
        other_views=voxels.views[other].count[rows],
    )


def compare_appearance(
    before: EpochReconstruction,
    after: EpochReconstruction,
    encoder: FeatureEncoder,
    config: AppearanceConfig = _DEFAULT_CONFIG,
) -> CrossAppearance:
    """Compares the look of each epoch's surfaces with the other epoch's.

    Each epoch's points are binned into voxels of ``config.voxel_size``,
    represented by their mean. Every frame with an image and a depth map is
    encoded exactly once and then dropped, so only one frame's features are
    in memory. A frame adds the feature at the projection of a voxel of
    either epoch to the views of the frame's epoch, provided it sees the
    surface there (see ``AppearanceConfig``). The per-voxel sums take memory
    proportional to the voxel count times the feature size.

    Args:
        before: The BEFORE reconstruction.
        after: The AFTER reconstruction, in the same world frame.
        encoder: Feature encoder; it is not unloaded.
        config: Voxel size, depth test, and view threshold.

    Returns:
        The similarity and view counts for the points of both epochs.

    Raises:
        ValueError: If ``before`` is not the BEFORE epoch, ``after`` is not
            the AFTER epoch, or a point is not finite.
    """
    if before.epoch != Epoch.BEFORE:
        raise ValueError("before must be the BEFORE reconstruction")
    if after.epoch != Epoch.AFTER:
        raise ValueError("after must be the AFTER reconstruction")
    voxels = {
        Epoch.BEFORE: _voxelize(
            as_finite_points(before.points, "before points", 0),
            config.voxel_size,
        ),
        Epoch.AFTER: _voxelize(
            as_finite_points(after.points, "after points", 0),
            config.voxel_size,
        ),
    }
    if any(len(v.centers) for v in voxels.values()):
        patch_size = encoder.patch_size
        for frame in (*before.frames, *after.frames):
            if frame.image is None or frame.depth is None:
                continue
            features = encoder.encode(frame.image)
            usable = _usable_pixels(frame, config)
            for target in voxels.values():
                _add_views(
                    target.views[frame.epoch],
                    target.centers,
                    frame,
                    usable,
                    features,
                    patch_size,
                    config,
                )
    return CrossAppearance(
        before=_point_appearance(voxels[Epoch.BEFORE], Epoch.BEFORE, config),
        after=_point_appearance(voxels[Epoch.AFTER], Epoch.AFTER, config),
    )


def _fuse_side(
    evidence: PointEvidence,
    appearance: PointAppearance,
    config: AppearanceConfig,
    side: str,
) -> PointEvidence:
    """Marks the PRESENT points of one epoch whose similarity is low.

    Args:
        evidence: The geometric votes on the epoch's points.
        appearance: The appearance of the same points.
        config: Supplies ``changed_below``.
        side: Name of the epoch, for the error message.

    Returns:
        New evidence with the same counts and the fused states.

    Raises:
        ValueError: If the appearance has another number of rows than the
            evidence.
    """
    if len(appearance.similarity) != len(evidence.states):
        raise ValueError(
            f"{side} appearance has {len(appearance.similarity)} rows for "
            f"{len(evidence.states)} {side} points"
        )
    flip = (evidence.states == PointState.PRESENT) & (
        appearance.similarity < config.changed_below
    )
    states = np.where(flip, np.int8(PointState.CHANGED), evidence.states)
    return PointEvidence(counts=evidence.counts, states=states.astype(np.int8))


def fuse_appearance(
    evidence: CrossEvidence,
    appearance: CrossAppearance,
    config: AppearanceConfig = _DEFAULT_CONFIG,
) -> CrossEvidence:
    """Marks PRESENT points that look different in the other epoch CHANGED.

    A PRESENT point whose similarity is below ``config.changed_below``
    becomes CHANGED. Every other state is kept, and a NaN similarity never
    changes anything. The vote counts are returned unchanged, so a CHANGED
    point with mostly consistent votes is an appearance change: the geometry
    agreed and the look did not.

    Args:
        evidence: The geometric votes, as returned by ``vote_epochs``.
        appearance: The appearance of the same points.
        config: Supplies ``changed_below``.

    Returns:
        New evidence with the same counts and the fused states.

    Raises:
        ValueError: If the appearance of an epoch has another number of
            rows than its evidence.
    """
    return CrossEvidence(
        before=_fuse_side(evidence.before, appearance.before, config, "before"),
        after=_fuse_side(evidence.after, appearance.after, config, "after"),
    )
