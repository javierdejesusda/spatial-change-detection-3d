import dataclasses
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

import spatial_change.capture as capture
from spatial_change.capture import read_frames
from spatial_change.pipeline import RunConfig, videos
from spatial_change.pipeline.config import MIN_KEYFRAMES, build_stage_configs
from spatial_change.pipeline.videos import (
    ImageWindow,
    center_crop_to_aspect,
    crop_window,
    load_video_frames,
)

WriteVideo = Callable[..., None]
FPS = 30.0
STAGES = build_stage_configs(RunConfig(sample_fps=10.0, max_keyframes=6), "cpu")
SCORED = build_stage_configs(RunConfig(max_keyframes=6, frame_stride=30), "cpu")


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
    assert loaded.after_window == ImageWindow()
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
    assert dataclasses.astuple(loaded.after_window) == pytest.approx(
        (0.0, 1 / 6, 1.0, 5 / 6)
    )
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


CROP_CASES = [
    ((120, 240, 3), 4 / 3, (120, 160)),
    ((120, 160, 3), 2.0, (80, 160)),
    ((120, 160, 3), 4 / 3, (120, 160)),
    ((121, 240, 3), 4 / 3, (121, 161)),
    ((121, 160, 3), 2.0, (80, 160)),
]


@pytest.mark.parametrize(("shape", "aspect", "expected"), CROP_CASES)
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


@pytest.mark.parametrize(("shape", "aspect", "expected"), CROP_CASES)
def test_crop_window_names_the_window_the_center_crop_cuts(
    shape: tuple[int, int, int], aspect: float, expected: tuple[int, int]
) -> None:
    height, width = shape[:2]
    image = np.arange(np.prod(shape), dtype=np.uint32).astype(np.uint8)
    image = image.reshape(shape)
    top = (height - expected[0]) // 2
    left = (width - expected[1]) // 2

    window = crop_window(height, width, aspect)

    assert dataclasses.astuple(window) == pytest.approx(
        (
            top / height,
            left / width,
            (top + expected[0]) / height,
            (left + expected[1]) / width,
        )
    )
    rows, cols = window.slices(height, width)
    assert (rows, cols) == (
        slice(top, top + expected[0]),
        slice(left, left + expected[1]),
    )
    np.testing.assert_array_equal(
        center_crop_to_aspect(image, aspect), image[rows, cols]
    )


def test_the_center_crop_takes_its_window_from_crop_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    monkeypatch.setattr(
        videos, "crop_window", lambda h, w, a: ImageWindow(0.1, 0.2, 0.4, 0.6)
    )

    cropped = center_crop_to_aspect(image, 1.0)

    np.testing.assert_array_equal(cropped, image[1:4, 2:6])


@pytest.mark.parametrize(
    "bounds",
    [
        {"top": -0.1},
        {"left": -0.1},
        {"bottom": 1.1},
        {"right": 1.1},
        {"top": 0.5, "bottom": 0.5},
        {"top": 0.6, "bottom": 0.4},
        {"left": 0.5, "right": 0.5},
        {"left": 0.9, "right": 0.1},
        {"top": float("nan")},
        {"right": float("nan")},
    ],
)
def test_an_image_window_must_be_a_proper_part_of_the_image(
    bounds: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="window"):
        ImageWindow(**bounds)


def test_the_default_image_window_is_the_whole_image() -> None:
    assert ImageWindow() == ImageWindow(0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("window", "shape", "expected"),
    [
        (ImageWindow(), (8, 10), (slice(0, 8), slice(0, 10))),
        (
            ImageWindow(0.25, 0.1, 0.75, 0.5),
            (8, 10),
            (slice(2, 6), slice(1, 5)),
        ),
        (
            ImageWindow(0.0, 0.0, 0.01, 0.01),
            (10, 10),
            (slice(0, 1), slice(0, 1)),
        ),
        (
            ImageWindow(0.99, 0.99, 1.0, 1.0),
            (10, 10),
            (slice(9, 10), slice(9, 10)),
        ),
    ],
)
def test_a_window_covers_the_pixels_it_rounds_to_and_at_least_one(
    window: ImageWindow,
    shape: tuple[int, int],
    expected: tuple[slice, slice],
) -> None:
    assert window.slices(*shape) == expected


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


def forbid_selection(frames: object, config: object) -> list[int]:
    raise AssertionError("a fixed stride must not select keyframes")


def test_a_fixed_stride_returns_exactly_the_frames_on_the_stride(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 100, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 100, seed=2)
    monkeypatch.setattr(capture, "select_keyframes", forbid_selection)

    loaded = load_video_frames(before, after, SCORED, max_frames=128)

    assert [f.index for f in loaded.before] == [0, 30, 60, 90]
    assert [f.index for f in loaded.after] == [0, 30, 60, 90]
    sizes = {f.image.shape for f in (*loaded.before, *loaded.after)}
    assert len(sizes) == 1
    assert not loaded.after_cropped
    frame = next(f for f in read_frames(before, stride=30) if f.index == 30)
    kept = next(f for f in loaded.before if f.index == 30)
    resized = cv2.resize(
        frame.image, kept.image.shape[1::-1], interpolation=cv2.INTER_AREA
    )
    assert np.abs(resized.astype(int) - kept.image.astype(int)).mean() < 2.0


@pytest.mark.parametrize(
    ("count", "max_frames", "expected"),
    [
        (100, 4, [0, 90]),
        (200, 6, [0, 90, 180]),
        (300, 8, [0, 90, 180, 270]),
    ],
)
def test_strided_frames_beyond_the_cap_are_thinned_evenly_on_the_stride(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    max_frames: int,
    expected: list[int],
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, count, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, count, seed=2)
    monkeypatch.setattr(capture, "select_keyframes", forbid_selection)

    loaded = load_video_frames(before, after, SCORED, max_frames=max_frames)

    assert [f.index for f in loaded.before] == expected
    assert [f.index for f in loaded.after] == expected


@pytest.mark.parametrize(
    ("counts", "max_frames", "strided", "kept"),
    [
        ((100, 130), 4, (4, 5), (2, 2)),
        ((100, 130), 6, (4, 5), (3, 3)),
        ((100, 130), 128, (4, 5), (4, 5)),
        ((91, 90), 128, (4, 3), (4, 3)),
    ],
)
def test_a_fixed_stride_counts_the_frames_that_qualified_before_thinning(
    tmp_path: Path,
    counts: tuple[int, int],
    max_frames: int,
    strided: tuple[int, int],
    kept: tuple[int, int],
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, counts[0], seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, counts[1], seed=2)

    loaded = load_video_frames(before, after, SCORED, max_frames=max_frames)

    assert (loaded.strided_before, loaded.strided_after) == strided
    assert (len(loaded.before), len(loaded.after)) == kept


def test_keyframes_have_no_strided_count(tmp_path: Path) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 90, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 90, seed=2)

    loaded = load_video_frames(before, after, STAGES, max_frames=128)

    assert loaded.strided_before is None
    assert loaded.strided_after is None


def test_a_wider_after_video_is_still_cropped_on_a_fixed_stride(
    tmp_path: Path,
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 70, seed=1)
    after = noise_video(tmp_path / "b.mp4", 240, 120, 70, seed=2)

    loaded = load_video_frames(before, after, SCORED, max_frames=128)

    assert loaded.after_cropped
    assert loaded.after_window != ImageWindow()
    assert [f.index for f in loaded.after] == [0, 30, 60]
    assert loaded.before[0].image.shape == loaded.after[0].image.shape


def test_no_room_for_a_strided_frame_is_an_error(tmp_path: Path) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 40, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 40, seed=2)

    with pytest.raises(ValueError, match="a.mp4"):
        load_video_frames(before, after, SCORED, max_frames=1)
