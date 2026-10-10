"""Small helpers to time a call and record its peak accelerator memory.

torch is imported lazily so the module works on machines without it.
"""

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

Param = str | int | float


@dataclass(frozen=True)
class Measurement:
    """One timed call.

    Attributes:
        name: Label of the measured step.
        seconds: Wall-clock duration.
        peak_memory_mb: Peak memory in MB, or None when not probed.
        params: Parameters describing the run.
    """

    name: str
    seconds: float
    peak_memory_mb: float | None
    params: dict[str, Param]


class MemoryProbe(Protocol):
    """Source of peak memory readings."""

    def reset(self) -> None:
        """Clears the recorded peak."""

    def peak_mb(self) -> float:
        """Returns the peak memory since the last reset, in MB."""


class CudaMemoryProbe:
    """Peak CUDA memory allocated, read through torch."""

    def reset(self) -> None:
        """Clears the CUDA peak memory statistics."""
        importlib.import_module("torch").cuda.reset_peak_memory_stats()

    def peak_mb(self) -> float:
        """Returns the peak allocated CUDA memory in MB."""
        torch = importlib.import_module("torch")
        return float(torch.cuda.max_memory_allocated()) / (1 << 20)


def measure[T](
    name: str,
    fn: Callable[[], T],
    *,
    params: Mapping[str, Param] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    memory_probe: MemoryProbe | None = None,
    sync: Callable[[], None] | None = None,
) -> tuple[T, Measurement]:
    """Runs ``fn`` once and records its duration and peak memory.

    Args:
        name: Label of the measured step.
        fn: Zero-argument callable to run.
        params: Parameters describing the run.
        clock: Monotonic clock returning seconds.
        memory_probe: Optional probe, reset before and read after the call.
        sync: Optional barrier called right before the start clock read and
            right after ``fn`` returns, so asynchronous accelerator work is
            included in the duration. Pass ``torch.cuda.synchronize`` when
            timing CUDA code.

    Returns:
        The result of ``fn`` and its measurement.
    """
    if memory_probe is not None:
        memory_probe.reset()
    if sync is not None:
        sync()
    start = clock()
    result = fn()
    if sync is not None:
        sync()
    seconds = clock() - start
    peak = memory_probe.peak_mb() if memory_probe is not None else None
    return result, Measurement(name, seconds, peak, dict(params or {}))


def to_markdown_table(measurements: Sequence[Measurement]) -> str:
    """Renders measurements as a Markdown table.

    Args:
        measurements: Rows to render.

    Returns:
        A table with name, seconds, peak memory, and one column per param
        key (sorted union across rows). Missing values are left blank.
    """
    keys = sorted({k for m in measurements for k in m.params})
    header = ["name", "seconds", "peak_memory_mb", *keys]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for m in measurements:
        peak = "n/a" if m.peak_memory_mb is None else f"{m.peak_memory_mb:.1f}"
        cells = [
            m.name,
            f"{m.seconds:.2f}",
            peak,
            *(str(m.params[k]) if k in m.params else "" for k in keys),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
