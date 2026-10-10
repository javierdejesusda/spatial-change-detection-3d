import inspect
import weakref
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from spatial_change.capture.keyframes import (
    KeyframeConfig,
    overlap,
    select_keyframes,
)
from spatial_change.capture.sharpness import sharp_mask, sharpness
from spatial_change.capture.video import VideoFrame

MakeFrames = Callable[..., list[VideoFrame]]


def _gray(frame: VideoFrame) -> np.ndarray:
    return frame.image.mean(axis=2).astype(np.uint8)


def test_overlap_large_shift_is_low(make_frames: MakeFrames) -> None:
    frames = make_frames(n_frames=2, step=100)

    assert overlap(_gray(frames[0]), _gray(frames[1])) < 0.5


@pytest.mark.parametrize("step", [0, 4, 8, 24])
def test_overlap_of_translation_is_shared_area(
    make_frames: MakeFrames, step: int
) -> None:
    frames = make_frames(n_frames=2, step=step)
    a, b = _gray(frames[0]), _gray(frames[1])

    assert overlap(a, b) == pytest.approx(1 - step / a.shape[1], abs=0.01)


def test_overlap_matches_known_2d_shift(make_frames: MakeFrames) -> None:
    a = _gray(make_frames(n_frames=1)[0])
    b = np.roll(a, (3, 5), axis=(0, 1))
    height, width = a.shape
    expected = (1 - 5 / width) * (1 - 3 / height)

    assert overlap(a, b) == pytest.approx(expected, abs=0.01)
    assert overlap(b, a) == pytest.approx(expected, abs=0.01)


def test_overlap_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        overlap(np.zeros((10, 10), np.uint8), np.zeros((10, 12), np.uint8))


def test_static_video_yields_about_min_keyframes(
    make_frames: MakeFrames,
) -> None:
    frames = make_frames(n_frames=60, step=0)
    config = KeyframeConfig()

    picked = select_keyframes(frames, config)

    assert abs(len(picked) - config.min_keyframes) <= 1


def test_translating_video_yields_more_than_static(
    make_frames: MakeFrames,
) -> None:
    config = KeyframeConfig(min_keyframes=2)

    moving = select_keyframes(make_frames(step=8), config)
    static = select_keyframes(make_frames(step=0), config)

    assert len(moving) > len(static)


def test_budget_is_respected(make_frames: MakeFrames) -> None:
    config = KeyframeConfig(max_keyframes=5, min_keyframes=2)

    picked = select_keyframes(make_frames(step=8), config)

    assert len(picked) <= 5


def test_blurred_frames_are_never_selected(make_frames: MakeFrames) -> None:
    blurred = {0, 1, 14, 15, 16, 30, 58, 59}
    frames = make_frames(blurred=blurred)

    picked = select_keyframes(frames, KeyframeConfig())

    assert not blurred.intersection(picked)


def test_output_is_deterministic_sorted_and_spans_sharp_range(
    make_frames: MakeFrames,
) -> None:
    frames = make_frames(blurred={0, 1, 58, 59, 20, 21})
    config = KeyframeConfig()

    first = select_keyframes(frames, config)
    second = select_keyframes(frames, config)

    mask = sharp_mask([sharpness(f.image) for f in frames])
    sharp_idx = np.flatnonzero(mask)
    assert first == second
    assert first == sorted(set(first))
    assert first[0] == sharp_idx[0]
    assert first[-1] == sharp_idx[-1]
    assert set(first) <= set(sharp_idx.tolist())


def test_empty_input_returns_empty() -> None:
    assert select_keyframes([], KeyframeConfig()) == []


def test_few_sharp_frames_are_all_returned(make_frames: MakeFrames) -> None:
    frames = make_frames(n_frames=6, step=2)
    mask = sharp_mask([sharpness(f.image) for f in frames])

    picked = select_keyframes(frames, KeyframeConfig())

    assert picked == np.flatnonzero(mask).tolist()


def test_config_rejects_inconsistent_budget() -> None:
    with pytest.raises(ValueError):
        KeyframeConfig(max_keyframes=4, min_keyframes=8)


def test_top_up_fills_the_largest_gaps(make_frames: MakeFrames) -> None:
    moving = make_frames(n_frames=30, step=8)
    still = [
        VideoFrame(index=30 + i, timestamp_s=0.0, image=moving[-1].image)
        for i in range(30)
    ]

    # A 48 px shift overlaps by exactly 0.7, so a threshold of 0.68 keeps the
    # greedy pass off that tie and every 7th moving frame is a keyframe.
    config = KeyframeConfig(min_keyframes=6, overlap_threshold=0.68)

    # The greedy pass keeps only the moving part plus the last frame, so the
    # top-up must land in the still part instead of crowding the start.
    selected = select_keyframes([*moving, *still], config)

    assert len(selected) == config.min_keyframes
    assert max(np.diff(selected)) <= 16


def test_generator_input_gives_the_same_result_as_a_list(
    make_frames: MakeFrames,
) -> None:
    frames = make_frames(blurred={0, 1, 58, 59, 20, 21})
    config = KeyframeConfig()

    streamed = select_keyframes((frame for frame in frames), config)

    assert streamed == select_keyframes(frames, config)


def test_frames_are_not_retained_while_streaming(
    make_frames: MakeFrames,
) -> None:
    frames = make_frames()
    images: list[weakref.ref[np.ndarray]] = []
    alive_when_next_is_decoded: list[int] = []

    def stream() -> Iterator[VideoFrame]:
        for frame in frames:
            alive_when_next_is_decoded.append(
                sum(ref() is not None for ref in images)
            )
            fresh = VideoFrame(
                index=frame.index,
                timestamp_s=frame.timestamp_s,
                image=frame.image.copy(),
            )
            images.append(weakref.ref(fresh.image))
            yield fresh

    select_keyframes(stream(), KeyframeConfig())

    assert max(alive_when_next_is_decoded) <= 2


def test_blur_parameters_default_to_the_sharp_mask_defaults() -> None:
    defaults = inspect.signature(sharp_mask).parameters
    config = KeyframeConfig()

    assert config.blur_keep_quantile == defaults["keep_quantile"].default
    assert config.blur_floor_ratio == defaults["floor_ratio"].default


def test_blur_parameters_reach_the_sharp_mask(
    make_frames: MakeFrames,
) -> None:
    blurred = {0, 1, 14, 15, 16, 30, 58, 59}
    config = KeyframeConfig(blur_keep_quantile=0.0, blur_floor_ratio=0.0)

    picked = select_keyframes(make_frames(blurred=blurred), config)

    assert blurred.intersection(picked)
