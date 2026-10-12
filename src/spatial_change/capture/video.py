"""Video decoding for phone recordings.

Frames are decoded with OpenCV and converted to RGB at the decode boundary,
so every image inside the package is an ``(H, W, 3)`` uint8 RGB array.
"""

import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

NOMINAL_FPS = 30.0
"""Frame rate assumed when a container reports none."""


@dataclass(frozen=True)
class VideoInfo:
    """Basic properties of a video stream.

    Attributes:
        fps: Frames per second reported by the container.
        frame_count: Number of frames reported by the container.
        width: Frame width in pixels, after orientation is applied.
        height: Frame height in pixels, after orientation is applied.
    """

    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame.

    Attributes:
        index: 0-based index of the frame in the decoded stream.
        timestamp_s: Presentation time in seconds.
        image: RGB image of shape (H, W, 3), dtype uint8.
    """

    index: int
    timestamp_s: float
    image: NDArray[np.uint8]


@contextmanager
def _capture(path: Path) -> Iterator[cv2.VideoCapture]:
    """Opens a capture with auto-orientation enabled and releases it on exit.

    Args:
        path: Video file path.

    Yields:
        An opened capture.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot open the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open video: {path}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        yield cap
    finally:
        cap.release()


def probe_video(path: Path) -> VideoInfo:
    """Reads the stream properties of a video.

    The size is taken from the first decoded frame, because the container
    properties may not reflect the rotation metadata.

    Args:
        path: Video file path.

    Returns:
        The fps, frame count and oriented frame size.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot open the file or decode a frame.
    """
    with _capture(path) as cap:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
    if not ok:
        raise ValueError(f"no decodable frames in video: {path}")
    height, width = frame.shape[:2]
    return VideoInfo(
        fps=fps, frame_count=frame_count, width=width, height=height
    )


def read_frames(
    path: Path, stride: int = 1, max_frames: int | None = None
) -> Iterator[VideoFrame]:
    """Decodes every ``stride``-th frame as RGB.

    Errors for a missing or unreadable file are raised when iteration starts.

    Args:
        path: Video file path.
        stride: Keep one frame out of every ``stride`` decoded frames.
        max_frames: Stop after yielding this many frames. None means no limit.

    Yields:
        Frames with their stream index and timestamp.

    Raises:
        ValueError: If ``stride`` is below 1 or OpenCV cannot open the file.
        FileNotFoundError: If the path does not exist.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    with _capture(path) as cap:
        # Damaged or variable-rate files can report 0 fps; phone video is
        # nominally 30 fps.
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or NOMINAL_FPS
        yielded = 0
        for index in itertools.count():
            if max_frames is not None and yielded >= max_frames:
                break
            if not cap.grab():
                break
            if index % stride:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                break
            msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_s = msec / 1000.0 if msec > 0 else index / fps
            yield VideoFrame(
                index=index,
                timestamp_s=timestamp_s,
                image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(
                    np.uint8, copy=False
                ),
            )
            yielded += 1
