import pytest

from spatial_change.pipeline import STAGES, StageEvent, format_event
from spatial_change.pipeline.progress import StageClock


def test_a_stage_reports_its_start_then_its_end() -> None:
    events: list[StageEvent] = []
    ticks = iter([10.0, 12.5])
    clock = StageClock(events.append, now=ticks.__next__)

    with clock.stage("voting"):
        pass

    index = STAGES.index("voting") + 1
    assert events == [
        StageEvent("voting", index, len(STAGES), None),
        StageEvent("voting", index, len(STAGES), 2.5),
    ]
    assert clock.timings == {"voting": 2.5}


def test_a_failed_stage_reports_no_end_and_records_no_time() -> None:
    events: list[StageEvent] = []
    clock = StageClock(events.append, now=iter([0.0, 1.0]).__next__)

    with pytest.raises(RuntimeError, match="boom"), clock.stage("voting"):
        raise RuntimeError("boom")

    assert [e.seconds for e in events] == [None]
    assert clock.timings == {}


def test_an_unknown_stage_is_rejected() -> None:
    clock = StageClock(None)

    with pytest.raises(ValueError, match="stage"), clock.stage("nonsense"):
        pass


def test_without_a_callback_the_timings_are_still_recorded() -> None:
    clock = StageClock(None, now=iter([1.0, 4.0]).__next__)

    with clock.stage("lifting"):
        pass

    assert clock.timings == {"lifting": 3.0}


def test_the_end_event_is_one_plain_line() -> None:
    end = StageEvent("voting", 3, 12, 4.23)

    assert format_event(end) == "[3/12] voting: 4.2 s"
    assert format_event(StageEvent("voting", 3, 12, None)) is None


def test_the_stage_names_are_unique() -> None:
    assert len(set(STAGES)) == len(STAGES)
