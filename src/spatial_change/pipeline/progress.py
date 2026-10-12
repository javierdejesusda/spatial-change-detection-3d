"""Stage names, timing, and progress events of a pipeline run."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

STAGES: tuple[str, ...] = (
    "keyframes",
    "reconstruction",
    "alignment",
    "voting",
    "appearance",
    "candidates",
    "lifting",
    "descriptors",
    "labeling",
    "matching",
    "describing",
    "outputs",
)
"""The stages of a run in order; a run of frames starts at the second."""


@dataclass(frozen=True)
class StageEvent:
    """The start or the end of one stage.

    Attributes:
        name: The stage, one of ``STAGES``.
        index: Position of the stage in ``STAGES``, counting from 1.
        total: Number of stages.
        seconds: Wall time of the stage on its end event, ``None`` on its
            start event.
    """

    name: str
    index: int
    total: int
    seconds: float | None


ProgressCallback = Callable[[StageEvent], None]


def format_event(event: StageEvent) -> str | None:
    """Words a stage event as one plain line.

    Args:
        event: The event.

    Returns:
        A line such as ``[3/12] voting: 4.2 s`` for an end event, ``None``
        for a start event.
    """
    if event.seconds is None:
        return None
    return f"[{event.index}/{event.total}] {event.name}: {event.seconds:.1f} s"


class StageClock:
    """Times the stages of a run and reports them to a callback.

    Attributes:
        timings: Wall time of every stage that finished, by name.
    """

    def __init__(
        self,
        callback: ProgressCallback | None,
        now: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Creates a clock.

        Args:
            callback: Receives a start and an end event per stage; ``None``
                only records the timings.
            now: Monotonic clock in seconds.
        """
        self._callback = callback
        self._now = now
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Times one stage.

        A stage that raises sends no end event and records no time.

        Args:
            name: One of ``STAGES``.

        Yields:
            Nothing; the body of the ``with`` is the stage.

        Raises:
            ValueError: If ``name`` is not a stage.
        """
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}")
        index = STAGES.index(name) + 1
        self._send(StageEvent(name, index, len(STAGES), None))
        started = self._now()
        yield
        seconds = self._now() - started
        self.timings[name] = seconds
        self._send(StageEvent(name, index, len(STAGES), seconds))

    def _send(self, event: StageEvent) -> None:
        if self._callback is not None:
            self._callback(event)
