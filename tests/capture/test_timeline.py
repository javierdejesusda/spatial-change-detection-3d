import math

import pytest

from spatial_change.capture.timeline import (
    MAX_TIMELINE_S,
    cfr_source_indices,
)

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
    # Frame 499 starts at tick 499.499 and lasts 1.001 ticks, so it ends on
    # the 1.5 tick tie. ffmpeg rounds the start down to 499.49899292 (a
    # multiple of 1/65536) and then nudges it up by half that step, which
    # leaves the end at 1.5000005 ticks: two slots.
    assert sources[499:502] == [499, 499, 500]
    assert sources[502:1000] == list(range(501, 999))


def test_a_24_fps_video_repeats_every_fourth_frame_at_an_exact_tie() -> None:
    # Each frame lasts 1.25 ticks, so frames 1, 5, 9, ... start a quarter tick
    # late and end on the 1.5 tick tie, which the nudge of their start sends
    # to two slots.
    sources = cfr_source_indices(evenly(40, 24.0))

    assert sources == [
        i for i in range(40) for _ in range(2 if i % 4 == 1 else 1)
    ]


def test_a_start_a_hair_short_of_a_half_tick_counts_as_the_half() -> None:
    # The second frame ends 4e-6 ticks short of 1.5: it rounds to the half
    # and the nudge lifts it over, so it fills two slots.
    times = [0.0, (1.5 - 4e-6) * TICK]

    assert cfr_source_indices(times, [TICK, TICK]) == [0, 1, 1]


def test_a_start_a_step_short_of_a_half_tick_stays_short() -> None:
    # 1.5e-5 ticks is one step of 1/65536 below 1.5: the start keeps that
    # step and the nudge of half a step leaves the end under the half.
    times = [0.0, (1.5 - 1.5e-5) * TICK]

    assert cfr_source_indices(times, [TICK, TICK]) == [0, 1]


def test_a_negative_start_is_nudged_down() -> None:
    # The end is 1.5 minus the nudge, which rounds to one slot; a nudge up
    # would make it two.
    times = [-1.5 * TICK]

    assert cfr_source_indices(times, [3 * TICK]) == [0]


def test_a_start_on_a_whole_tick_is_not_nudged() -> None:
    # The first frame ends on 1.5 ticks exactly and starts on tick 0, which
    # ffmpeg leaves alone: the tie goes to the even count, two slots. A nudge
    # toward minus would leave the end under the half and fill one.
    assert cfr_source_indices([0.0], [1.5 * TICK]) == [0, 0]


def test_a_start_on_a_later_whole_tick_is_not_nudged() -> None:
    # The second frame starts on tick 2, one past the free slot, and lasts
    # 1.5 ticks, so its end is 2.5 past the slot. The tie goes to the even
    # count, two slots; a nudge toward plus would make it three.
    times = [0.0, 2 * TICK]

    assert cfr_source_indices(times, [TICK, 1.5 * TICK]) == [0, 1, 1]


def test_a_count_a_hair_under_a_half_rounds_up_in_single_precision() -> None:
    # 1e-9 ticks is far below the resolution of a float32 near 1.5, so
    # llrintf sees the half itself and sends it to two. Plain double rounding
    # would see the shortfall and give one.
    assert cfr_source_indices([0.0], [(1.5 - 1e-9) * TICK]) == [0, 0]


def test_a_start_is_rounded_to_a_step_before_it_is_nudged() -> None:
    # The second frame lasts one tick after a first frame that fills three
    # slots, so it is dropped when its start is under 0.9 ticks (its end is
    # then under -1.1). It starts 0.6 steps of 1/65536 tick under 0.9. Only
    # nudged by half a step it would still be 0.1 steps under and dropped.
    # ffmpeg first rounds it to the nearest step, 0.4 under 0.9, and the nudge
    # then lifts it to 0.1 steps over: the frame stays.
    times = [0.0, (58981.8 / 65536) * TICK]

    assert cfr_source_indices(times, [3 * TICK, TICK]) == [0, 0, 0, 1]


def test_a_tie_between_slots_goes_to_the_larger_count() -> None:
    # The second frame ends on 2.5 exactly, which would round to 2, and
    # starts 1.5 ticks late, so its first slot repeats the first frame.
    times = [0.0, 2.5 * TICK]

    assert cfr_source_indices(times, [TICK, TICK]) == [0, 0, 1, 1]


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


def test_a_timeline_that_ends_at_the_cap_is_built() -> None:
    times = [0.0, MAX_TIMELINE_S - 1]

    sources = cfr_source_indices(times, [1.0, 1.0], fps=1.0)

    assert len(sources) == MAX_TIMELINE_S


def test_a_timeline_past_the_cap_is_refused_naming_its_length() -> None:
    times = [0.0, MAX_TIMELINE_S]

    with pytest.raises(
        ValueError, match=r"3601 s.*longer than.*3600 s.*corrupt"
    ):
        cfr_source_indices(times, [1.0, 1.0], fps=1.0)


@pytest.mark.parametrize(
    ("times", "durations"),
    [
        ([0.0, 1e5], None),
        ([0.0, 1e12], None),
        ([0.0, 1.0], [TICK, 1e5]),
        ([0.0, math.inf], None),
    ],
    ids=[
        "jump of a day",
        "jump past any video",
        "huge duration",
        "infinite time",
    ],
)
def test_a_corrupt_timestamp_is_refused_before_any_slot_is_made(
    times: list[float], durations: list[float] | None
) -> None:
    with pytest.raises(ValueError, match="longer than"):
        cfr_source_indices(times, durations)
