"""Keyframes of two videos, cut to one size, with bounded memory.

Each video is decoded twice. The first pass keeps only a thumbnail and a
sharpness score of every sampled frame while the keyframes are chosen. The
second pass keeps only the keyframes, cropped and resized at once, so the
full-resolution frames never pile up. A fixed frame stride asks for every
Nth frame instead, which is decoded once.

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


@dataclass(frozen=True)
class ImageWindow:
    """The part of an image a frame shows, as fractions of the image.

    Attributes:
        top: Top edge, from 0 at the top of the image to 1 at the bottom.
        left: Left edge, from 0 at the left of the image to 1 at the right.
        bottom: Bottom edge, below ``top``.
        right: Right edge, right of ``left``.
    """

    top: float = 0.0
    left: float = 0.0
    bottom: float = 1.0
    right: float = 1.0

    def __post_init__(self) -> None:
        """Validates the edges.

        Raises:
            ValueError: If the window is empty or reaches outside the image.
        """
        if not (
            0.0 <= self.top < self.bottom <= 1.0
            and 0.0 <= self.left < self.right <= 1.0
        ):
            raise ValueError(
                "an image window needs 0 <= top < bottom <= 1 and "
                f"0 <= left < right <= 1, got {self}"
            )

    def slices(self, height: int, width: int) -> tuple[slice, slice]:
        """Names the rows and columns the window covers in a pixel grid.

        Args:
            height: Rows of the grid.
            width: Columns of the grid.

        Returns:
            The row slice and the column slice, each holding at least one
            pixel however small the window is.
        """
        top = min(round(self.top * height), height - 1)
        bottom = max(top + 1, min(round(self.bottom * height), height))
        left = min(round(self.left * width), width - 1)
        right = max(left + 1, min(round(self.right * width), width))
        return slice(top, bottom), slice(left, right)


WHOLE_IMAGE = ImageWindow()
"""The window of a frame that shows all of its image."""


@dataclass(frozen=True, eq=False)
class LoadedVideos:
    """The keyframes of both videos at one size.

    Attributes:
        before: BEFORE keyframes in temporal order.
        after: AFTER keyframes in temporal order, cut to the aspect of the
            BEFORE frames.
        before_info: Properties of the BEFORE video.
        after_info: Properties of the AFTER video.
        after_window: The part of the AFTER image the frames show; the
            whole image unless they were center-cropped to the BEFORE
            aspect.
        strided_before: On a fixed frame stride, the BEFORE frames whose
            index is a multiple of the stride, of which ``before`` holds
            all or, above the keyframe cap, an even subset; ``None`` for
            keyframes.
        strided_after: The same for the AFTER video.
    """

    before: list[VideoFrame]
    after: list[VideoFrame]
    before_info: VideoInfo
    after_info: VideoInfo
    after_window: ImageWindow
    strided_before: int | None = None
    strided_after: int | None = None

    @property
    def after_cropped(self) -> bool:
        """Whether the AFTER frames were center-cropped."""
        return self.after_window != WHOLE_IMAGE


def crop_window(height: int, width: int, aspect: float) -> ImageWindow:
    """Names the largest centered window of an image with a given ratio.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        aspect: Target width over height, positive.

    Returns:
        The window; the whole image if it already has that ratio.
    """
    if width / height > aspect:
        new_width = max(1, round(height * aspect))
        left = (width - new_width) // 2
        return ImageWindow(0.0, left / width, 1.0, (left + new_width) / width)
    new_height = max(1, round(width / aspect))
    top = (height - new_height) // 2
    return ImageWindow(top / height, 0.0, (top + new_height) / height, 1.0)


def center_crop_to_aspect(
    image: NDArray[np.uint8], aspect: float
) -> NDArray[np.uint8]:
    """Cuts the largest centered window of an image with a given ratio.

    Args:
        image: Image of shape ``(H, W, 3)``.
        aspect: Target width over height, positive.

    Returns:
        The window ``crop_window`` names; the image itself if it already has
        the ratio.
    """
    height, width = image.shape[:2]
    rows, cols = crop_window(height, width, aspect).slices(height, width)
    return image[rows, cols]


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


def _strided_frames(
    path: Path,
    stride: int,
    cap: int,
    prepare: Callable[[NDArray[np.uint8]], NDArray[np.uint8]],
) -> tuple[list[VideoFrame], int]:
    """Decodes the frames whose index is a multiple of the stride.

    A fixed stride is a promise that the frames can be found by their index,
    so none is dropped for being blurry or redundant. Each is cropped and
    resized as soon as it is decoded, which keeps a long video small. When
    more frames than ``cap`` qualify, an evenly spaced subset is kept that
    still starts at the first frame on the stride and ends at the last.

    Args:
        path: The video.
        stride: Frames whose index is a multiple of this are kept.
        cap: Most frames to keep.
        prepare: Turns a decoded image into the image that is kept.

    Returns:
        The frames in temporal order, and how many frames were on the stride
        before any were thinned out.

    Raises:
        ValueError: If no frame is left.
    """
    import spatial_change.capture as capture
    from spatial_change.capture import VideoFrame
    from spatial_change.capture.keyframes import uniform

    frames = [
        VideoFrame(frame.index, frame.timestamp_s, prepare(frame.image))
        for frame in capture.read_frames(path, stride)
    ]
    on_stride = len(frames)
    if on_stride > cap:
        frames = uniform(frames, cap)
    if not frames:
        raise ValueError(f"no frame on the stride in video: {path}")
    return frames, on_stride


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

    With ``stages.frame_stride`` set, the frames are those whose index is a
    multiple of the stride instead of the sharpest and least redundant ones,
    and the result says how many there were before any were thinned out.

    Args:
        before_path: The BEFORE video.
        after_path: The AFTER video.
        stages: Sample rate, keyframe budget, backbone size, and frame
            stride.
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
    window = (
        crop_window(infos[1].height, infos[1].width, aspect)
        if cropped
        else WHOLE_IMAGE
    )
    cap = min(stages.max_keyframes, max_frames // 2)

    def prepare_before(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return resize_image(image, width, height)

    def prepare_after(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if cropped:
            image = center_crop_to_aspect(image, aspect)
        return resize_image(np.ascontiguousarray(image), width, height)

    stride = stages.frame_stride
    if stride is not None:
        before, strided_before = _strided_frames(
            before_path, stride, cap, prepare_before
        )
        after, strided_after = _strided_frames(
            after_path, stride, cap, prepare_after
        )
        return LoadedVideos(
            before,
            after,
            infos[0],
            infos[1],
            window,
            strided_before,
            strided_after,
        )
    strides = [
        max(1, round(_frame_rate(info) / stages.sample_fps)) for info in infos
    ]
    before = _keyframes(before_path, strides[0], cap, prepare_before)
    after = _keyframes(after_path, strides[1], cap, prepare_after)
    return LoadedVideos(before, after, infos[0], infos[1], window)
