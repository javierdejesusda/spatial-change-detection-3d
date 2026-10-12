"""Keyframes of two videos, cut to one size, with bounded memory.

Each video is decoded twice. The first pass keeps only a thumbnail and a
sharpness score of every sampled frame while the keyframes are chosen. The
second pass keeps only the keyframes, cropped and resized at once, so the
full-resolution frames never pile up.

Importing this module does not load cv2; the capture package loads it when
a video is read.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from spatial_change.pipeline.config import (
    BACKBONE_MULTIPLE,
    MIN_KEYFRAMES,
    StageConfigs,
)

if TYPE_CHECKING:
    from spatial_change.capture import VideoFrame, VideoInfo

_ASPECT_TOLERANCE = 0.01
"""Relative aspect difference below which the AFTER frames are not cropped."""


@dataclass(frozen=True, eq=False)
class LoadedVideos:
    """The keyframes of both videos at one size.

    Attributes:
        before: BEFORE keyframes in temporal order.
        after: AFTER keyframes in temporal order, cut to the aspect of the
            BEFORE frames.
        before_info: Properties of the BEFORE video.
        after_info: Properties of the AFTER video.
        after_cropped: Whether the AFTER frames were center-cropped to the
            BEFORE aspect.
    """

    before: list[VideoFrame]
    after: list[VideoFrame]
    before_info: VideoInfo
    after_info: VideoInfo
    after_cropped: bool


def center_crop_to_aspect(
    image: NDArray[np.uint8], aspect: float
) -> NDArray[np.uint8]:
    """Cuts the middle of an image to a width over height ratio.

    Args:
        image: ``(H, W, 3)`` image.
        aspect: Target width over height, positive.

    Returns:
        The largest centered crop of that ratio; the image itself if it
        already has it.
    """
    height, width = image.shape[:2]
    if width / height > aspect:
        new_width = max(1, round(height * aspect))
        left = (width - new_width) // 2
        return image[:, left : left + new_width]
    new_height = max(1, round(width / aspect))
    top = (height - new_height) // 2
    return image[top : top + new_height]


def _keyframes(
    path: Path,
    stride: int,
    cap: int,
    prepare: Callable[[NDArray[np.uint8]], NDArray[np.uint8]],
) -> list[VideoFrame]:
    """Picks the keyframes of one video and decodes only those.

    Args:
        path: The video.
        stride: Every how many decoded frames one is sampled.
        cap: Most keyframes to keep.
        prepare: Turns a decoded image into the image that is kept.

    Returns:
        The keyframes in temporal order.

    Raises:
        ValueError: If the video has no sharp frame.
    """
    import spatial_change.capture as capture
    from spatial_change.capture import KeyframeConfig, VideoFrame

    config = KeyframeConfig(
        max_keyframes=cap, min_keyframes=min(MIN_KEYFRAMES, cap)
    )
    positions = capture.select_keyframes(
        capture.read_frames(path, stride), config
    )
    if not positions:
        raise ValueError(f"no sharp frame in video: {path}")
    # The positions count the sampled frames, not the frames of the video.
    wanted = set(positions)
    last = max(positions)
    kept = []
    for position, frame in enumerate(capture.read_frames(path, stride)):
        if position in wanted:
            kept.append(
                VideoFrame(frame.index, frame.timestamp_s, prepare(frame.image))
            )
        if position >= last:
            break
    return kept


def _frame_rate(info: VideoInfo) -> float:
    """Gives the frame rate of a video, or the nominal one if unreported."""
    from spatial_change.capture.video import NOMINAL_FPS

    return info.fps if math.isfinite(info.fps) and info.fps > 0 else NOMINAL_FPS


def load_video_frames(
    before_path: Path,
    after_path: Path,
    stages: StageConfigs,
    max_frames: int,
) -> LoadedVideos:
    """Reads the keyframes of both videos at one size.

    The size is that of the BEFORE video under the backbone's rules. When
    the AFTER video has another aspect ratio, its frames are first
    center-cropped to the BEFORE ratio, so both epochs see the same field
    of view shape and no frame is stretched.

    Args:
        before_path: The BEFORE video.
        after_path: The AFTER video.
        stages: Sample rate, keyframe budget, and backbone size.
        max_frames: Views the reconstruction accepts; the keyframes of both
            videos together stay within it.

    Returns:
        The keyframes and what was done to them.

    Raises:
        FileNotFoundError: If a video does not exist.
        ValueError: If a video cannot be decoded or has no sharp frame.
    """
    from spatial_change.capture import ResizePolicy, probe_video
    from spatial_change.capture.resize import resize_image

    infos = [probe_video(before_path), probe_video(after_path)]
    policy = ResizePolicy(stages.backbone_long_side, BACKBONE_MULTIPLE)
    height, width = policy.target_size(infos[0].height, infos[0].width)
    aspect = infos[0].width / infos[0].height
    after_aspect = infos[1].width / infos[1].height
    cropped = abs(after_aspect - aspect) > _ASPECT_TOLERANCE * aspect
    cap = min(stages.max_keyframes, max_frames // 2)

    def prepare_before(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return resize_image(image, width, height)

    def prepare_after(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if cropped:
            image = center_crop_to_aspect(image, aspect)
        return resize_image(np.ascontiguousarray(image), width, height)

    strides = [
        max(1, round(_frame_rate(info) / stages.sample_fps)) for info in infos
    ]
    before = _keyframes(before_path, strides[0], cap, prepare_before)
    after = _keyframes(after_path, strides[1], cap, prepare_after)
    return LoadedVideos(before, after, infos[0], infos[1], cropped)
