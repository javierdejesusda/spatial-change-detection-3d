"""Tests for the profiling helpers."""

from spatial_change.diagnostics.profiling import (
    Measurement,
    measure,
    to_markdown_table,
)


class _FakeProbe:
    def __init__(self) -> None:
        self.events: list[str] = []

    def reset(self) -> None:
        self.events.append("reset")

    def peak_mb(self) -> float:
        self.events.append("peak")
        return 123.45


def test_measure_returns_result_and_elapsed_time() -> None:
    result, m = measure(
        "square",
        lambda: 7 * 7,
        params={"n": 7},
        clock=iter([10.0, 12.5]).__next__,
    )

    assert result == 49
    assert m == Measurement("square", 2.5, None, {"n": 7})


def test_probe_is_reset_before_and_read_after_call() -> None:
    probe = _FakeProbe()

    def fn() -> None:
        probe.events.append("call")

    _, m = measure("x", fn, clock=iter([0.0, 1.0]).__next__, memory_probe=probe)

    assert probe.events == ["reset", "call", "peak"]
    assert m.peak_memory_mb == 123.45


def test_sync_brackets_the_timed_call() -> None:
    events: list[str] = []

    def clock() -> float:
        events.append("clock")
        return float(len(events))

    def fn() -> None:
        events.append("call")

    measure("x", fn, clock=clock, sync=lambda: events.append("sync"))

    assert events == ["sync", "clock", "call", "sync", "clock"]


def test_peak_memory_is_none_without_probe() -> None:
    _, m = measure("x", lambda: None, clock=iter([0.0, 1.0]).__next__)

    assert m.peak_memory_mb is None
    assert m.params == {}


def test_markdown_table_has_header_rows_and_na() -> None:
    table = to_markdown_table(
        [
            Measurement("a", 1.234, 56.78, {"size": 10}),
            Measurement("b", 0.5, None, {"mode": "fast"}),
        ]
    )

    lines = table.splitlines()
    assert lines[0] == "| name | seconds | peak_memory_mb | mode | size |"
    assert lines[1] == "| --- | --- | --- | --- | --- |"
    assert lines[2] == "| a | 1.23 | 56.8 |  | 10 |"
    assert lines[3] == "| b | 0.50 | n/a | fast |  |"
