"""The constant-rate timeline that annotation frame indices count on.

SceneDiff annotations index a frame as the n-th output frame of
``ffmpeg -r 30``, which resamples a video to a constant 30 fps by dropping
and repeating frames. That index is a slot on the 30 fps timeline, not the
position of a frame in the decoded stream, and the two only agree for a
video that is already 30 fps without drift.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np

ANNOTATION_FPS = 30.0
"""Rate of the timeline annotation frame indices are counted on."""

MAX_TIMELINE_S = 3600.0
"""Longest timeline, in seconds, that is built.

A corrupt timestamp far in the future would otherwise make the list of slots
grow without bound.
"""

_TOLERANCE_TICKS = 1.1
"""Gap, in output ticks, past which ffmpeg drops or repeats frames."""

_PREVIOUS_FRAME_BIAS_TICKS = 0.6
"""Part of a gap that is not filled with the previous frame."""

_START_STEPS_PER_TICK = 1 << 16
"""Resolution of a frame start in ffmpeg's encoder, in steps per tick."""


def _round_ticks(ticks: float) -> int:
    """Rounds a tick count like C ``llrintf``.

    The value is first narrowed to single precision, which turns the last
    bits of floating point noise on an exact half into the half itself, and
    halves then go to the even integer.
    """
    return round(float(np.float32(ticks)))


def _encoder_start(ticks: float) -> float:
    """Rounds a frame start the way ffmpeg's encoder does.

    The start goes to the nearest step of 1/65536 tick, halves away from
    zero. Unless that lands on a whole tick it is then nudged by half a step
    away from zero. The nudge keeps a start that lands on a half tick from
    being decided by rounding noise further on.
    """
    steps = math.copysign(
        math.floor(abs(ticks) * _START_STEPS_PER_TICK + 0.5), ticks
    )
    start = steps / _START_STEPS_PER_TICK
    if start == round(start):
        return start
    return start + math.copysign(0.5 / _START_STEPS_PER_TICK, start)


def cfr_source_indices(
    times_s: Sequence[float],
    durations_s: Sequence[float] | None = None,
    fps: float = ANNOTATION_FPS,
) -> list[int]:
    """Names the source frame each slot of a constant-rate timeline shows.

    This follows ffmpeg's constant frame rate sync (``video_sync_process``)
    as run by ``-r``, checked against the output of ffmpeg 8.0.1 on phone
    video at 30, 60 and 120 fps. The frame durations must not be rounded to
    whole ticks. The decoded images are not needed, only when each frame is
    shown.

    The sync keeps ``next_pts``, the next free slot. A frame shown at ``t``
    for ``d`` seconds starts at tick ``t * fps`` and ends ``d * fps`` ticks
    later; ``delta0`` is how far its start lies past ``next_pts`` and
    ``delta`` how far its end does. It fills no slot when ``delta`` is below
    -1.1 and one slot when ``delta`` is within 1.1 of zero. Above 1.1 it
    fills ``round(delta)`` slots, and when ``delta0`` is also above 1.1 the
    first ``round(delta0 - 0.6)`` of them repeat the previous frame instead.

    Two roundings are done as ffmpeg does them, so that a frame ending near
    a tie is decided as ffmpeg decides it. The start is first rounded the
    way its encoder does (``adjust_frame_pts_to_encoder_tb``, see
    ``_encoder_start``); the durations are not rounded. Then the slot counts
    are rounded in single precision with halves going to the even integer,
    as ``llrintf`` does. ffmpeg also clips a frame that starts before
    ``next_pts``; that only rewrites a negative ``delta0``, which no branch
    above reads, so it is not repeated here.

    ffmpeg runs a file with one stream, as every phone video here is, in its
    ``vscfr`` mode, which does not repeat the first frame when it starts at
    least half a tick late. Times measured from the video stream start, as
    OpenCV reports them, put the first frame at 0, where ``vscfr`` and the
    plain rule agree, so that mode is not needed. Absolute presentation
    times are not that: a stream that starts at 0.7 ticks would shift every
    slot.

    Args:
        times_s: Presentation time in seconds of every frame in the order it
            is shown, measured from the start of the video stream.
        durations_s: How long each frame is shown, in seconds. By default a
            frame lasts until the next one, the last one as long as the one
            before it, and a lone frame one tick. ffmpeg uses the duration
            its container gives each packet, which is the same except next
            to dropped frames in variable frame rate video: there it counts
            the gap twice, so pass the container's durations to match it
            (``read_container_durations`` lists them).
        fps: Rate of the timeline.

    Returns:
        The index into ``times_s`` of the frame that fills each slot, from
        slot 0. A slot before the first frame shows the first frame.

    Raises:
        ValueError: If there is no frame, the durations do not pair with the
            times, ``fps`` is not positive, or the last frame ends after
            ``MAX_TIMELINE_S`` seconds. That is checked before any slot is
            made, so a corrupt timestamp cannot exhaust memory.
    """
    if not times_s:
        raise ValueError("a video needs at least one frame")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if durations_s is None:
        gaps = [after - before for before, after in itertools.pairwise(times_s)]
        durations_s = [*gaps, gaps[-1] if gaps else 1 / fps]
    elif len(durations_s) != len(times_s):
        raise ValueError("a frame needs one duration per frame")
    length_s = max(
        time_s + duration_s
        for time_s, duration_s in zip(times_s, durations_s, strict=True)
    )
    if length_s > MAX_TIMELINE_S:
        raise ValueError(
            f"timeline of {length_s:.0f} s is longer than the limit of "
            f"{MAX_TIMELINE_S:.0f} s: the timestamps are corrupt or the "
            "video is too long"
        )
    sources: list[int] = []
    for index, (time_s, duration_s) in enumerate(
        zip(times_s, durations_s, strict=True)
    ):
        delta0 = _encoder_start(time_s * fps) - len(sources)
        delta = delta0 + duration_s * fps
        slots = 1
        repeats = 0
        if delta < -_TOLERANCE_TICKS:
            slots = 0
        elif delta > _TOLERANCE_TICKS:
            slots = _round_ticks(delta)
            if delta0 > _TOLERANCE_TICKS:
                repeats = _round_ticks(delta0 - _PREVIOUS_FRAME_BIAS_TICKS)
        sources.extend(
            max(index - 1, 0) if slot < repeats else index
            for slot in range(slots)
        )
    return sources
