"""The joint anchor pass: both epochs reconstructed in one forward pass.

Feeding the keyframes of both videos to one reconstruction places every
camera in a single world frame, so the two epochs are comparable without a
separate alignment step.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
    result = reconstruct_views(reconstructor, [vf.image for vf in video_frames])
    frames = [
        frame_from_result(
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


def reconstruct_views(
    reconstructor: Reconstructor, images: Sequence[NDArray[np.uint8]]
) -> ReconstructionResult:
    """Runs the reconstructor and checks that it returned one view per image.

    Args:
        reconstructor: Model that predicts poses, intrinsics, and depth.
        images: Input images, one per requested view.

    Returns:
        The reconstruction result.

    Raises:
        ValueError: If the result holds a different number of views than
            ``images``.
    """
    result = reconstructor.reconstruct(images)
    if len(result.depth) != len(images):
        raise ValueError(
            f"reconstructor returned {len(result.depth)} views for "
            f"{len(images)} images"
        )
    return result


def frame_from_result(
    epoch: Epoch, video_frame: VideoFrame, result: ReconstructionResult, i: int
) -> Frame:
    """Builds the posed frame for view ``i`` of a reconstruction result.

    The image is resized to the depth map resolution so fusion can sample
    colors at the depth pixels.

    Args:
        epoch: Capture the frame belongs to.
        video_frame: Input frame that was fed to the reconstructor as view
            ``i``.
        result: Reconstruction output.
        i: View index in ``result``.

    Returns:
        The frame with pose, intrinsics, depth, confidence, and image.
    """
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
