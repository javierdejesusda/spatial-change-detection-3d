import dataclasses
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from rle_helpers import encode_mask

import spatial_change.capture as capture
from spatial_change.capture import read_frames
from spatial_change.capture.durations import ffprobe_path
from spatial_change.domain import SE3, ChangeType, Epoch, Frame, Intrinsics
from spatial_change.evaluation.projection import (
    ChangeObject3D,
    ProjectionConfig,
    project_to_pred_items,
)
from spatial_change.evaluation.scenediff import (
    gt_items,
    parse_pair_annotations,
)
from spatial_change.pipeline import RunConfig, videos
from spatial_change.pipeline.config import MIN_KEYFRAMES, build_stage_configs
from spatial_change.pipeline.videos import (
    FrameDurations,
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
    path: Path,
    width: int,
    height: int,
    count: int,
    seed: int,
    fps: float = FPS,
) -> Path:
    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 256, (height, width + 4 * count, 3), np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
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


TIMELINE = build_stage_configs(
    RunConfig(max_keyframes=64, frame_stride=3), "cpu"
)


def native_images(
    path: Path, shape: tuple[int, ...]
) -> list[NDArray[np.float32]]:
    """Decodes a video, each frame resized to ``shape`` as a loaded one is."""
    height, width = shape[:2]
    return [
        cv2.resize(
            frame.image, (width, height), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        for frame in read_frames(path)
    ]


def source_of(
    image: NDArray[np.uint8], natives: list[NDArray[np.float32]]
) -> int:
    """Names the decoded frame an image shows, by the least squared error."""
    errors = [
        float(((image.astype(np.float32) - native) ** 2).mean())
        for native in natives
    ]
    return int(np.argmin(errors))


def ffmpeg_slot_images(
    video: Path, folder: Path, shape: tuple[int, ...]
) -> list[NDArray[np.uint8]]:
    """Reads the frames ``ffmpeg -r 30`` writes, resized to ``shape``."""
    folder.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video)]
        + ["-r", "30", "-start_number", "0", str(folder / "%d.png")],
        check=True,
    )
    height, width = shape[:2]
    return [
        cv2.resize(
            cv2.cvtColor(
                cv2.imread(str(folder / f"{n}.png")), cv2.COLOR_BGR2RGB
            ),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        for n in range(len(list(folder.glob("*.png"))))
    ]


def test_a_60_fps_clip_is_indexed_on_the_30_fps_timeline(
    tmp_path: Path,
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 60, seed=1, fps=60.0)

    loaded = load_video_frames(before, before, TIMELINE, max_frames=128)

    slots = list(range(0, 31, 3))
    assert [f.index for f in loaded.before] == slots
    assert [f.index for f in loaded.after] == slots
    natives = native_images(before, loaded.before[0].image.shape)
    sources = [n if n < 4 else 2 * n - 3 for n in slots]
    assert [source_of(f.image, natives) for f in loaded.before] == sources
    assert [f.timestamp_s for f in loaded.before] == pytest.approx(
        [k / 60 for k in sources], abs=1e-3
    )
    assert (loaded.strided_before, loaded.strided_after) == (11, 11)


def test_a_slow_clip_shows_each_frame_in_the_slots_it_fills(
    tmp_path: Path,
) -> None:
    stages = build_stage_configs(
        RunConfig(max_keyframes=64, frame_stride=1), "cpu"
    )
    before = noise_video(tmp_path / "a.mp4", 160, 120, 10, seed=1, fps=15.0)

    loaded = load_video_frames(before, before, stages, max_frames=128)

    assert [f.index for f in loaded.before] == list(range(20))
    natives = native_images(before, loaded.before[0].image.shape)
    assert [source_of(f.image, natives) for f in loaded.before] == [
        k for k in range(10) for _ in range(2)
    ]
    assert loaded.strided_before == 20


@pytest.mark.parametrize("fps", [15.0, 24.0, 60.0, 120.0])
def test_the_slots_hold_the_frames_ffmpeg_writes_at_30_fps(
    tmp_path: Path, fps: float
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    before = noise_video(tmp_path / "a.mp4", 160, 120, 80, seed=1, fps=fps)
    stages = build_stage_configs(
        RunConfig(max_keyframes=64, frame_stride=2), "cpu"
    )

    loaded = load_video_frames(before, before, stages, max_frames=128)

    shape = loaded.before[0].image.shape
    natives = native_images(before, shape)
    written = ffmpeg_slot_images(before, tmp_path / "ffmpeg", shape)
    assert loaded.strided_before == (len(written) + 1) // 2
    for frame in loaded.before:
        expected = written[frame.index]
        assert source_of(frame.image, natives) == source_of(expected, natives)


def test_a_prediction_and_its_annotation_share_the_key_of_a_60_fps_slot(
    tmp_path: Path,
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 60, seed=1, fps=60.0)
    loaded = load_video_frames(before, before, TIMELINE, max_frames=128)
    natives = native_images(before, loaded.before[0].image.shape)
    # On a 60 fps clip, ffmpeg writes the native frame 21 as output frame 12.
    annotated = 12
    shown = next(f for f in loaded.before if source_of(f.image, natives) == 21)
    intrinsics = Intrinsics(
        fx=10.0, fy=10.0, cx=10.0, cy=10.0, width=20, height=20
    )
    posed = Frame(
        epoch=Epoch.BEFORE,
        index=shown.index,
        timestamp_s=shown.timestamp_s,
        intrinsics=intrinsics,
        world_from_camera=SE3.identity(),
        depth=np.full((20, 20), 5.0, dtype=np.float32),
    )
    xs = np.linspace(-0.5, 0.5, 5)
    gx, gy = np.meshgrid(xs, xs)
    points = np.stack([gx.ravel(), gy.ravel(), np.full(25, 5.0)], axis=1)
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.5, {Epoch.BEFORE: points})
    mask = np.zeros((20, 20), dtype=bool)
    mask[9:12, 9:12] = True
    raw = {
        "scene_type": "living_room",
        "objects": [
            {
                "label": "thing",
                "in_video1": True,
                "in_video2": False,
                "original_obj_idx": 0,
            }
        ],
        "video1_objects": {0: {str(annotated): encode_mask(mask)}},
        "video2_objects": {},
        "comments": "",
        "light_change": False,
    }

    predicted = project_to_pred_items(
        [obj],
        [posed],
        [],
        20,
        20,
        config=ProjectionConfig(splat_radius=0),
        stride=3,
    )
    truth = gt_items(parse_pair_annotations("p", raw), 20, 20, stride=3)

    assert shown.index == annotated
    assert sorted(predicted[0].masks) == sorted(truth[0].masks) == [(1, 4)]


def test_only_the_frames_of_the_kept_slots_are_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = build_stage_configs(
        RunConfig(max_keyframes=4, frame_stride=3), "cpu"
    )
    before = noise_video(tmp_path / "a.mp4", 160, 120, 60, seed=1, fps=60.0)
    asked: list[set[int]] = []
    real = capture.read_frames_at

    def spy(path: Path, indices: set[int]) -> object:
        asked.append(set(indices))
        return real(path, indices)

    monkeypatch.setattr(capture, "read_frames_at", spy)

    loaded = load_video_frames(before, before, stages, max_frames=128)

    # 11 slots are on the stride; the cap of 4 keeps slots 0, 9, 21 and 30,
    # which show the frames 0, 15, 39 and 57.
    assert [f.index for f in loaded.before] == [0, 9, 21, 30]
    assert asked == [{0, 15, 39, 57}] * 2
    assert loaded.strided_before == 11


def test_a_slot_whose_frame_cannot_be_decoded_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 60, seed=1, fps=60.0)
    real = capture.read_frames_at

    def lossy(path: Path, indices: set[int]) -> object:
        return (f for f in real(path, indices) if f.index != 15)

    monkeypatch.setattr(capture, "read_frames_at", lossy)

    loaded = load_video_frames(before, before, TIMELINE, max_frames=128)

    assert [f.index for f in loaded.before] == [
        0,
        3,
        6,
        12,
        15,
        18,
        21,
        24,
        27,
        30,
    ]


def test_a_video_without_timestamps_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 30, seed=1)
    monkeypatch.setattr(capture, "read_timestamps", lambda path: [])

    with pytest.raises(ValueError, match="a.mp4"):
        load_video_frames(before, before, TIMELINE, max_frames=128)


ONE_EACH = build_stage_configs(
    RunConfig(max_keyframes=64, frame_stride=1), "cpu"
)


def with_durations(
    monkeypatch: pytest.MonkeyPatch, *answers: list[float] | None
) -> list[Path]:
    """Makes the container give each video of a load the next answer."""
    asked: list[Path] = []
    pending = list(answers)

    def read(path: Path) -> list[float] | None:
        asked.append(path)
        return pending.pop(0)

    monkeypatch.setattr(capture, "read_container_durations", read)
    return asked


def longer_tenth(count: int) -> list[float]:
    """Gives every frame one 30 fps tick but the tenth, which lasts two."""
    return [2 / 30 if i == 10 else 1 / 30 for i in range(count)]


def test_the_container_durations_decide_which_frame_fills_each_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 40, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 40, seed=2)
    asked = with_durations(monkeypatch, longer_tenth(40), longer_tenth(40))

    loaded = load_video_frames(before, after, ONE_EACH, max_frames=128)

    assert asked == [before, after]
    assert [f.index for f in loaded.before] == list(range(41))
    natives = native_images(before, loaded.before[0].image.shape)
    assert [source_of(f.image, natives) for f in loaded.before] == [
        *range(11),
        *range(10, 40),
    ]
    assert loaded.strided_before == 41
    assert loaded.frame_durations == "container"


def dropped_frame_clip(tmp_path: Path) -> Path:
    """Re-encodes a noise clip with B-frames and without its 21st frame."""
    source = noise_video(tmp_path / "source.mp4", 160, 120, 40, seed=1)
    clip = tmp_path / "dropped.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
    command += ["-vf", r"select=not(eq(n\,20))", "-fps_mode", "vfr"]
    command += ["-c:v", "libx264", "-bf", "2", "-g", "100"]
    command += ["-pix_fmt", "yuv420p", str(clip)]
    if subprocess.run(command, capture_output=True).returncode != 0:
        pytest.skip("ffmpeg cannot encode with libx264")
    return clip


def test_the_container_durations_match_the_frames_ffmpeg_writes_after_a_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("ffmpeg") is None or ffprobe_path() is None:
        pytest.skip("ffmpeg and ffprobe are not installed")
    clip = dropped_frame_clip(tmp_path)
    contained = load_video_frames(clip, clip, ONE_EACH, max_frames=128)
    with_durations(monkeypatch, None, None)
    by_gaps = load_video_frames(clip, clip, ONE_EACH, max_frames=128)

    shape = contained.before[0].image.shape
    natives = native_images(clip, shape)
    written = ffmpeg_slot_images(clip, tmp_path / "ffmpeg", shape)
    expected = [source_of(image, natives) for image in written]
    assert (contained.frame_durations, by_gaps.frame_durations) == (
        "container",
        "gaps",
    )
    assert [source_of(f.image, natives) for f in contained.before] == expected
    assert [source_of(f.image, natives) for f in by_gaps.before] != expected


@pytest.mark.parametrize(
    "answer",
    [None, [1 / 30] * 39, [1 / 30] * 41],
    ids=["none", "one fewer", "one more"],
)
def test_without_usable_container_durations_the_gaps_between_frames_are_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: list[float] | None
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 40, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 40, seed=2)
    with_durations(monkeypatch, answer, answer)

    loaded = load_video_frames(before, after, ONE_EACH, max_frames=128)

    assert [f.index for f in loaded.before] == list(range(40))
    natives = native_images(before, loaded.before[0].image.shape)
    assert [source_of(f.image, natives) for f in loaded.before] == list(
        range(40)
    )
    assert loaded.frame_durations == "gaps"


UNIFORM = [1 / 30] * 40


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ((None, UNIFORM), "gaps"),
        ((UNIFORM, None), "gaps"),
        ((UNIFORM, UNIFORM), "container"),
    ],
)
def test_the_durations_are_the_containers_only_when_both_videos_have_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answers: tuple[list[float] | None, list[float] | None],
    expected: FrameDurations,
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 40, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 40, seed=2)
    with_durations(monkeypatch, *answers)

    loaded = load_video_frames(before, after, ONE_EACH, max_frames=128)

    assert loaded.frame_durations == expected


def test_keyframes_neither_ask_for_durations_nor_report_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = noise_video(tmp_path / "a.mp4", 160, 120, 60, seed=1)
    after = noise_video(tmp_path / "b.mp4", 160, 120, 60, seed=2)
    asked = with_durations(monkeypatch)

    loaded = load_video_frames(before, after, STAGES, max_frames=128)

    assert asked == []
    assert loaded.frame_durations is None
