"""From the joint reconstruction to two epochs in one z-up room frame.

The joint reconstruction puts both epochs in one world frame already, but
that frame has the orientation of the first camera and an arbitrary floor.
The chain here fuses each epoch, levels both with one gravity alignment
fitted on the union of their points and cameras, and then registers the
AFTER epoch onto the BEFORE one on the structure that did not change.

Importing this module does not load torch, open3d, or cv2.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.alignment import (
    Refinement,
    ScalePolicy,
    gravity_align,
    refine_alignment,
)
from spatial_change.domain import EpochReconstruction, Frame, Sim3
from spatial_change.pipeline.config import StageConfigs

ALIGNMENT_POINTS = 400_000
"""Points of the union cloud the floor search looks at, at most."""


@dataclass(frozen=True, eq=False)
class AlignedEpochs:
    """Both epochs in the room frame.

    Attributes:
        before: The BEFORE reconstruction, floor at ``z = 0``.
        after: The AFTER reconstruction, registered onto the BEFORE one.
        room_from_world: The gravity alignment applied to both epochs, from
            the world frame of the joint reconstruction.
        refinement: The registration applied to the AFTER epoch after it.
        floor_observed: Whether the floor was fitted to a visible plane
            (True) or assumed at the low end of the points (False).
    """

    before: EpochReconstruction
    after: EpochReconstruction
    room_from_world: Sim3
    refinement: Refinement
    floor_observed: bool


def _thin_points(
    points: NDArray[np.float64], limit: int, seed: int
) -> NDArray[np.float64]:
    """Draws a random subset of a cloud.

    A fixed stride would follow the voxel-key order of the fused points and
    could skip a whole plane, so the subset is a seeded draw instead.

    Args:
        points: Points of shape ``(N, 3)``.
        limit: Most points to keep, at least 1.
        seed: Seed of the draw.

    Returns:
        ``points`` itself if it has at most ``limit`` points, otherwise
        ``limit`` distinct points in their original order.
    """
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(points), limit, replace=False))
    return points[keep]


def align_epochs(
    before: Sequence[Frame], after: Sequence[Frame], stages: StageConfigs
) -> AlignedEpochs:
    """Fuses the two epochs and brings them into one room frame.

    Args:
        before: The posed BEFORE frames of the joint reconstruction.
        after: The posed AFTER frames, in the same world frame.
        stages: Fusion, floor search, and registration settings.

    Returns:
        The epochs in the room frame with the transforms that did it.

    Raises:
        ValueError: If a stage rejects its input, for example when no point
            is confident enough, no floor is found, or the static structure is
            too small or degenerate to register.
    """
    from spatial_change.reconstruction.fusion import fuse_frames

    fused = [
        fuse_frames(frames, stages.min_confidence, stages.fusion_voxel_size)
        for frames in (before, after)
    ]
    union = np.concatenate([item.points for item in fused])
    if len(union) == 0:
        raise ValueError("no confident points to align")
    thinned = _thin_points(union, ALIGNMENT_POINTS, stages.floor.seed)
    poses = [frame.world_from_camera for frame in (*before, *after)]
    alignment = gravity_align(thinned, poses, ScalePolicy.MODEL, stages.floor)
    leveled = [item.transformed(alignment.room_from_world) for item in fused]
    refinement = refine_alignment(
        leveled[1].points, leveled[0].points, config=stages.refine
    )
    registered = leveled[1].transformed(Sim3.from_se3(refinement.transform))
    return AlignedEpochs(
        leveled[0],
        registered,
        alignment.room_from_world,
        refinement,
        alignment.floor_observed,
    )
