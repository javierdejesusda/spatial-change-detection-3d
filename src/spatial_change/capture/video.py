"""Video decoding for phone recordings.

Frames are decoded with OpenCV and converted to RGB at the decode boundary,
so every image inside the package is an ``(H, W, 3)`` uint8 RGB array.
"""

import itertools
from collections.abc import Collection, Iterator
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
        index: 0-based index of the frame in the decoded stream, unless the
            frame was placed on another timeline, such as the 30 fps slots
            of a fixed frame stride in the pipeline.
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


def _frame_rate(cap: cv2.VideoCapture) -> float:
    """Reads the frame rate of a capture, or the nominal one if unreported.

    Damaged or variable-rate files can report 0 fps; phone video is
    nominally 30 fps.
    """
    return float(cap.get(cv2.CAP_PROP_FPS)) or NOMINAL_FPS


def _timestamp_s(cap: cv2.VideoCapture, index: int, fps: float) -> float:
    """Reads the presentation time of the frame just grabbed.

    Args:
        cap: A capture that has just grabbed frame ``index``.
        index: Stream index of that frame.
        fps: Frame rate to fall back on when the file gives no time.

    Returns:
        The time in seconds from the start of the video stream.
    """
    msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    return msec / 1000.0 if msec > 0 else index / fps


def _to_rgb(bgr: cv2.typing.MatLike) -> NDArray[np.uint8]:
    """Converts a decoded BGR frame to RGB."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


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
        fps = _frame_rate(cap)
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
            yield VideoFrame(
                index=index,
                timestamp_s=_timestamp_s(cap, index, fps),
                image=_to_rgb(bgr),
            )
            yielded += 1


def read_timestamps(path: Path) -> list[float]:
    """Lists the presentation time of every frame without converting any.

    Each frame is grabbed, which decodes it, but none is retrieved, so no
    image is converted or kept.

    Args:
        path: Video file path.

    Returns:
        The time in seconds of every frame from the start of the video
        stream, in stream order, as ``read_frames`` stamps it.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot open the file.
    """
    with _capture(path) as cap:
        fps = _frame_rate(cap)
        stamps = []
        for index in itertools.count():
            if not cap.grab():
                break
            stamps.append(_timestamp_s(cap, index, fps))
    return stamps


def read_frames_at(
    path: Path, indices: Collection[int]
) -> Iterator[VideoFrame]:
    """Decodes only the frames with the given stream indices as RGB.

    Every frame up to the last one named is grabbed, which decodes it, but
    only the named ones are converted. Errors for a missing or unreadable
    file are raised when iteration starts.

    Args:
        path: Video file path.
        indices: Stream indices of the frames to keep. Indices the video
            does not reach are ignored.

    Yields:
        The named frames in stream order with their stream index and
        timestamp.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot open the file.
    """
    wanted = set(indices)
    last = max(wanted, default=-1)
    with _capture(path) as cap:
        fps = _frame_rate(cap)
        for index in range(last + 1):
            if not cap.grab():
                break
            if index not in wanted:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                break
            yield VideoFrame(
                index=index,
                timestamp_s=_timestamp_s(cap, index, fps),
                image=_to_rgb(bgr),
            )
