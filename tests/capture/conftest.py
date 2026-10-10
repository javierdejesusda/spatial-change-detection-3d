"""Synthetic video and frame fixtures for capture tests."""

from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.capture.video import VideoFrame

WIDTH = 160
HEIGHT = 120
FPS = 30.0

VideoFactory = Callable[..., Path]
FramesFactory = Callable[..., list[VideoFrame]]
TextureFactory = Callable[..., NDArray[np.uint8]]
VideoWriterFn = Callable[[Path, Sequence[NDArray[np.uint8]]], None]


def texture(
    width: int, height: int, *, seed: int, sigma: float
) -> NDArray[np.uint8]:
    """Builds a seeded RGB texture of blurred noise.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Seed of the noise generator.
        sigma: Standard deviation of the Gaussian blur, in pixels.

    Returns:
        An image of shape (height, width, 3), dtype uint8.
    """
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), sigma)


def write_video(path: Path, bgr_frames: Sequence[NDArray[np.uint8]]) -> None:
    """Writes BGR frames to an mp4 file, skipping if it cannot be encoded.

    Args:
        path: Output file path.
        bgr_frames: Frames of shape (H, W, 3), dtype uint8, all the same size.
    """
    height, width = bgr_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot write mp4v on this platform")
    try:
        for bgr in bgr_frames:
            writer.write(bgr)
    finally:
        writer.release()


def _render(
    n_frames: int, step: int, blurred: Collection[int]
) -> list[NDArray[np.uint8]]:
    canvas = texture(WIDTH + step * n_frames, HEIGHT, seed=1234, sigma=1.2)
    images = []
    for i in range(n_frames):
        crop = np.ascontiguousarray(canvas[:, i * step : i * step + WIDTH])
        if i in blurred:
            crop = cv2.GaussianBlur(crop, (0, 0), 4.0)
        images.append(crop)
    return images


@pytest.fixture(name="texture")
def texture_fixture() -> TextureFactory:
    """Returns ``texture`` for tests that need seeded blurred noise."""
    return texture


@pytest.fixture(name="write_video")
def write_video_fixture() -> VideoWriterFn:
    """Returns ``write_video`` for tests that need a custom clip."""
    return write_video


@pytest.fixture
def make_frames() -> FramesFactory:
    """Returns a factory of RGB frames from a steadily translating texture."""

    def factory(
        n_frames: int = 60,
        step: int = 8,
        blurred: Collection[int] = (),
    ) -> list[VideoFrame]:
        images = _render(n_frames, step, blurred)
        return [
            VideoFrame(index=i, timestamp_s=i / FPS, image=image)
            for i, image in enumerate(images)
        ]

    return factory


@pytest.fixture
def make_video(tmp_path: Path) -> VideoFactory:
    """Returns a factory that writes a small synthetic mp4 file."""

    def factory(
        n_frames: int = 60,
        step: int = 4,
        blurred: Collection[int] = (),
        name: str = "clip.mp4",
    ) -> Path:
        path = tmp_path / name
        write_video(
            path,
            [
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for rgb in _render(n_frames, step, blurred)
            ],
        )
        return path

    return factory
