import dataclasses
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

import spatial_change.capture as capture
from spatial_change.capture import read_frames
from spatial_change.pipeline import RunConfig
from spatial_change.pipeline.config import MIN_KEYFRAMES, build_stage_configs
from spatial_change.pipeline.videos import (
    center_crop_to_aspect,
    load_video_frames,
)

WriteVideo = Callable[..., None]
FPS = 30.0
STAGES = build_stage_configs(RunConfig(sample_fps=10.0, max_keyframes=6), "cpu")


def noise_video(
    path: Path, width: int, height: int, count: int, seed: int
) -> Path:
    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 256, (height, width + 4 * count, 3), np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot write mp4v on this platform")
    try:
        for i in range(count):
            writer.write(np.ascontiguousarray(canvas[:, 4 * i : 4 * i + width]))
    finally:
        writer.release()
    return path


def test_the_frames_of_each_video_are_capped_and_share_one_size(
    tmp_path: Path,
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 90, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 90, seed=2)

    loaded = load_video_frames(before, after, STAGES, max_frames=128)

    assert 2 <= len(loaded.before) <= 6
    assert 2 <= len(loaded.after) <= 6
    sizes = {f.image.shape for f in (*loaded.before, *loaded.after)}
    assert len(sizes) == 1
    height, width, _ = sizes.pop()
    assert height % 14 == 0 and width % 14 == 0
    assert max(height, width) <= STAGES.backbone_long_side
    assert not loaded.after_cropped
    assert (loaded.before_info.width, loaded.before_info.height) == (160, 120)


def test_the_reconstruction_ceiling_caps_each_video(tmp_path: Path) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 90, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 90, seed=2)

    loaded = load_video_frames(before, after, STAGES, max_frames=8)

    assert len(loaded.before) + len(loaded.after) <= 8


def test_a_wider_after_video_is_cropped_to_the_before_aspect(
    tmp_path: Path,
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 60, seed=1)
    after = noise_video(tmp_path / "b.mp4", 240, 120, 60, seed=2)

    loaded = load_video_frames(before, after, STAGES, max_frames=128)

    assert loaded.after_cropped
    assert loaded.before[0].image.shape == loaded.after[0].image.shape
    assert (loaded.after_info.width, loaded.after_info.height) == (240, 120)


def test_keyframe_positions_map_to_the_frames_they_were_chosen_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 60, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 60, seed=2)
    monkeypatch.setattr(
        capture, "select_keyframes", lambda frames, config: [1, 4, 5]
    )

    loaded = load_video_frames(before, after, STAGES, max_frames=128)

    stride = 3
    assert [f.index for f in loaded.before] == [stride, 4 * stride, 5 * stride]
    wanted = {f.index: f for f in loaded.before}
    for frame in read_frames(before, stride=stride):
        if frame.index in wanted:
            resized = cv2.resize(
                frame.image,
                wanted[frame.index].image.shape[1::-1],
                interpolation=cv2.INTER_AREA,
            )
            assert (
                np.abs(
                    resized.astype(int) - wanted[frame.index].image.astype(int)
                ).mean()
                < 2.0
            )


def test_a_video_without_a_sharp_frame_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 30, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 30, seed=2)
    monkeypatch.setattr(capture, "select_keyframes", lambda frames, config: [])

    with pytest.raises(ValueError, match="a.mp4"):
        load_video_frames(before, after, STAGES, max_frames=128)


def test_a_missing_video_is_reported() -> None:
    with pytest.raises(FileNotFoundError, match="absent.mp4"):
        load_video_frames(
            Path("absent.mp4"), Path("absent.mp4"), STAGES, max_frames=128
        )


@pytest.mark.parametrize(
    ("shape", "aspect", "expected"),
    [
        ((120, 240, 3), 4 / 3, (120, 160)),
        ((120, 160, 3), 2.0, (80, 160)),
        ((120, 160, 3), 4 / 3, (120, 160)),
    ],
)
def test_center_crop_matches_the_aspect(
    shape: tuple[int, int, int], aspect: float, expected: tuple[int, int]
) -> None:
    image = np.arange(np.prod(shape), dtype=np.uint32).astype(np.uint8)
    image = image.reshape(shape)

    cropped = center_crop_to_aspect(image, aspect)

    assert cropped.shape[:2] == expected
    top = (shape[0] - expected[0]) // 2
    left = (shape[1] - expected[1]) // 2
    np.testing.assert_array_equal(
        cropped, image[top : top + expected[0], left : left + expected[1]]
    )


@pytest.mark.parametrize(("cap", "minimum"), [(48, MIN_KEYFRAMES), (3, 3)])
def test_the_keyframe_minimum_follows_the_shared_constant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cap: int,
    minimum: int,
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 30, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 30, seed=2)
    seen = []

    def select(frames: object, config: object) -> list[int]:
        seen.append(config)
        return [0]

    monkeypatch.setattr(capture, "select_keyframes", select)
    stages = build_stage_configs(RunConfig(max_keyframes=min(cap, 48)), "cpu")

    load_video_frames(before, after, stages, max_frames=2 * cap)

    assert [c.min_keyframes for c in seen] == [minimum, minimum]


@pytest.mark.parametrize("fps", [0.0, float("nan"), float("inf")])
def test_an_unreported_frame_rate_falls_back_to_the_nominal_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fps: float
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 30, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 30, seed=2)
    real = capture.probe_video
    strides = []
    real_read = capture.read_frames

    def probe(path: Path) -> object:
        return dataclasses.replace(real(path), fps=fps)

    def read(path: Path, stride: int = 1, max_frames: int | None = None):  # type: ignore[no-untyped-def]
        strides.append(stride)
        return real_read(path, stride, max_frames)

    monkeypatch.setattr(capture, "probe_video", probe)
    monkeypatch.setattr(capture, "read_frames", read)

    load_video_frames(before, after, STAGES, max_frames=128)

    assert set(strides) == {3}
