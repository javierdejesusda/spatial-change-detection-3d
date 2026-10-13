import pytest

from spatial_change.capture.timeline import cfr_source_indices

TICK = 1 / 30


def evenly(count: int, fps: float) -> list[float]:
    return [i / fps for i in range(count)]


def test_a_30_fps_video_maps_every_slot_to_its_own_frame() -> None:
    assert cfr_source_indices(evenly(10, 30.0)) == list(range(10))


def test_a_60_fps_video_keeps_the_first_four_then_every_other_frame() -> None:
    sources = cfr_source_indices(evenly(120, 60.0))

    assert sources == [0, 1, 2, 3, *range(5, 120, 2)]
    assert all(sources[n] == 2 * n - 3 for n in range(4, len(sources)))


def test_a_59_94_fps_video_starts_as_60_fps_and_shifts_phase_late() -> None:
    times = [i * 1001 / 60000 for i in range(900)]

    sources = cfr_source_indices(times)

    assert sources[:6] == [0, 1, 2, 3, 5, 7]
    assert sources[401] == 799
    assert sources[402] == 800
    assert sources[403] == 802


def test_a_120_fps_video_keeps_every_fourth_frame_after_the_first_two() -> None:
    sources = cfr_source_indices(evenly(40, 120.0))

    assert sources == [0, 1, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def test_a_29_97_fps_video_repeats_one_frame_after_half_a_tick_of_drift() -> (
    None
):
    times = [i * 1001 / 30000 for i in range(1200)]

    sources = cfr_source_indices(times)

    assert sources[:499] == list(range(499))
    # The tie at frame 499 falls on either side of the floating point error,
    # so the repeated frame is 499 or 500; the offset after it is not.
    assert sources[499:502] in ([499, 499, 500], [499, 500, 500])
    assert sources[502:1000] == list(range(501, 999))


def test_a_24_fps_video_repeats_every_fourth_frame_at_an_exact_tie() -> None:
    # Each frame lasts 1.25 ticks, so frames 1, 5, 9, ... start a quarter tick
    # late and end on the 1.5 tick tie that rounds to two slots.
    sources = cfr_source_indices(evenly(40, 24.0))

    assert sources == [
        i for i in range(40) for _ in range(2 if i % 4 == 1 else 1)
    ]


def test_a_rate_just_under_30_fps_repeats_a_frame_at_half_a_tick() -> None:
    times = [i / 29.7 for i in range(100)]

    sources = cfr_source_indices(times)

    assert sources[:50] == list(range(50))
    assert sources[49] == 49
    assert sources[50] == 49
    assert sources[51:100] == list(range(50, 99))


def test_a_gap_repeats_the_previous_frame_then_fills_with_the_next() -> None:
    times = [0 * TICK, 1 * TICK, 2 * TICK, 6 * TICK, 7 * TICK]

    sources = cfr_source_indices(times, [TICK] * 5)

    assert sources == [0, 1, 2, 2, 2, 3, 3, 4]


def test_a_frame_lasts_until_the_next_one_by_default() -> None:
    times = [0 * TICK, 1 * TICK, 2 * TICK, 6 * TICK, 7 * TICK]

    assert cfr_source_indices(times) == [0, 1, 2, 2, 2, 2, 3, 4]


def test_a_video_that_starts_late_repeats_its_first_frame() -> None:
    times = [2 * TICK, 3 * TICK, 4 * TICK]

    assert cfr_source_indices(times) == [0, 0, 0, 1, 2]


def test_a_single_frame_fills_one_slot() -> None:
    assert cfr_source_indices([0.0]) == [0]


def test_no_frame_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        cfr_source_indices([])


def test_durations_must_pair_with_the_times() -> None:
    with pytest.raises(ValueError, match="one duration per frame"):
        cfr_source_indices([0.0, TICK], [TICK])


def test_the_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        cfr_source_indices([0.0], fps=0.0)
