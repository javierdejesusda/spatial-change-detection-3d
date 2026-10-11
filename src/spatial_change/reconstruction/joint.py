"""The joint anchor pass: both epochs reconstructed in one forward pass.

Feeding the keyframes of both videos to one reconstruction places every
camera in a single world frame, so the two epochs are comparable without a
separate alignment step.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from spatial_change.capture import VideoFrame
from spatial_change.capture.resize import resize_image
from spatial_change.domain import Epoch, Frame
from spatial_change.ports import ReconstructionResult, Reconstructor


@dataclass(frozen=True, eq=False)
class JointReconstruction:
    """Posed frames of both epochs in one world frame.

    Attributes:
        before: Frames of the first capture, in input order.
        after: Frames of the second capture, in input order.
    """

    before: list[Frame]
    after: list[Frame]


def reconstruct_joint(
    reconstructor: Reconstructor,
    before: Sequence[VideoFrame],
    after: Sequence[VideoFrame],
) -> JointReconstruction:
    """Reconstructs the keyframes of both epochs in a single call.

    The reconstructor is not unloaded; that stays with the caller. Each
    frame's image is resized to the depth map resolution so later fusion
    can sample colors at the depth pixels.

    Args:
        reconstructor: Model that predicts poses, intrinsics, and depth.
        before: Keyframes of the first capture.
        after: Keyframes of the second capture.

    Returns:
        The frames of each epoch with pose, intrinsics, depth, confidence,
        and image.

    Raises:
        ValueError: If an epoch has no keyframes, the reconstructor
            returns a different number of views than it was given images,
            or the reconstructor rejects the input. The Pi3X adapter
            rejects frames of mixed sizes, so the frames of both epochs
            must share one size, and it rejects more frames than its
            ``max_frames``, which bounds ``len(before) + len(after)``
            together, not each epoch.
    """
    if not before or not after:
        raise ValueError("each epoch needs at least one keyframe")
    video_frames = [*before, *after]
    result = reconstructor.reconstruct([vf.image for vf in video_frames])
    if len(result.depth) != len(video_frames):
        raise ValueError(
            f"reconstructor returned {len(result.depth)} views for "
            f"{len(video_frames)} images"
        )
    frames = [
        _to_frame(
            Epoch.BEFORE if i < len(before) else Epoch.AFTER,
            video_frame,
            result,
            i,
        )
        for i, video_frame in enumerate(video_frames)
    ]
    return JointReconstruction(
        before=frames[: len(before)], after=frames[len(before) :]
    )


def _to_frame(
    epoch: Epoch, video_frame: VideoFrame, result: ReconstructionResult, i: int
) -> Frame:
    intrinsics = result.intrinsics[i]
    return Frame(
        epoch=epoch,
        index=video_frame.index,
        timestamp_s=video_frame.timestamp_s,
        intrinsics=intrinsics,
        world_from_camera=result.world_from_camera[i],
        image=resize_image(
            video_frame.image, intrinsics.width, intrinsics.height
        ),
        depth=result.depth[i],
        confidence=result.confidence[i],
    )
