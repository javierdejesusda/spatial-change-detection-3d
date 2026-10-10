"""Writing rendered frames to an mp4 file."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import Frame


def write_video(frames: Sequence[Frame], path: Path, fps: float) -> None:
    """Writes the RGB images of frames to an mp4 file.

    The frames are validated before any output is opened and the video is
    written to a temporary sibling that replaces ``path`` only on success,
    so a failure never leaves a stub or partial file.

    The file decodes through ``spatial_change.capture.video`` back to RGB
    frames of the same size and count.

    Args:
        frames: Frames with images, all of the same size.
        path: Output file path.
        fps: Frames per second, positive.

    Raises:
        ValueError: If ``frames`` is empty, a frame has no image, the
            sizes differ, or ``fps`` is not positive.
        RuntimeError: If OpenCV cannot open the output for writing.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    first = frames[0].intrinsics
    size = (first.width, first.height)
    images: list[NDArray[np.uint8]] = []
    for frame in frames:
        if frame.image is None:
            raise ValueError(f"frame {frame.index} has no image")
        if (frame.intrinsics.width, frame.intrinsics.height) != size:
            raise ValueError("all frames must have the same size")
        images.append(frame.image)
    temporary = path.with_name(f".{path.name}.tmp{path.suffix}")
    writer = cv2.VideoWriter()
    try:
        writer.open(str(temporary), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot write mp4 video: {path}")
        for image in images:
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # Release first: Windows cannot replace a file that is still open.
        writer.release()
        os.replace(temporary, path)
    except BaseException:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
