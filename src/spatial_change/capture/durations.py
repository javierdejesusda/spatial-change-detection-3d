"""Frame durations as the container stores them.

ffmpeg shows each frame for the duration of the packet it was decoded from,
which can differ from the gap to the next frame in video that dropped frames,
and OpenCV does not report it. ``ffprobe`` lists it without decoding.

The durations are read as whole ticks of the stream time base and scaled
here, not as the decimal seconds ffprobe can also print: those carry six
decimals, enough to decide a frame ending on a half tick the wrong way.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from spatial_change.capture.container import ISO_BMFF_DEMUXERS

TIMEOUT_S = 60.0
"""Seconds ffprobe may take to list the packets of one video."""

_DISCARD_FLAG = "D"
"""Flag of a packet the decoder does not turn into a frame."""


def ffprobe_path() -> str | None:
    """Finds ffprobe in the absolute folders of the ``PATH`` variable.

    The current directory is never searched, and neither is an empty or
    relative ``PATH`` entry, which is read from it. ``shutil.which`` searches
    the current directory first on Windows and for such an entry elsewhere,
    so a run started in a folder holding a file called ffprobe would run
    that file.

    On Windows only ``ffprobe.exe`` counts, not the other extensions of
    ``PATHEXT``. Python starts a batch file (``.bat``, ``.cmd``) through
    ``cmd.exe``, which reads its arguments under rules the quoting of the
    argument list does not cover, so a video name with an ``&`` or a quote
    in it could run a command.

    Returns:
        The absolute path of the first executable file called ffprobe, or
        None when there is none.
    """
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not Path(folder).is_absolute():
            continue
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _ticks(value: Any, what: str) -> int:
    """Checks that a packet field is a whole number of ticks."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"packet {what} is not a whole number: {value!r}")
    return value


def _time_base(listing: dict[str, Any]) -> tuple[int, int]:
    """Reads the time base of the first stream as numerator and denominator."""
    text = listing["streams"][0]["time_base"]
    numerator, denominator = (int(part) for part in text.split("/"))
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"unusable time base: {text!r}")
    return numerator, denominator


def parse_packet_durations(output: str) -> list[float]:
    """Reads the packet list of ``ffprobe`` into frame durations.

    Args:
        output: The JSON of ``ffprobe -show_entries
            packet=pts,duration,flags:stream=time_base -of json``.

    Returns:
        The duration in seconds of each packet that becomes a frame, in
        presentation order. Packets flagged as discarded, such as those an
        edit list cuts off, are left out.

    Raises:
        ValueError: If the listing is not JSON of that shape, the time base
            is not a positive ratio, or a packet that becomes a frame has no
            whole-number time or no positive duration.
    """
    try:
        listing = json.loads(output)
        numerator, denominator = _time_base(listing)
        packets = []
        for entry in listing["packets"]:
            if _DISCARD_FLAG in entry.get("flags", ""):
                continue
            time = _ticks(entry.get("pts"), "time")
            duration = _ticks(entry.get("duration"), "duration")
            if duration <= 0:
                raise ValueError(f"packet duration is not positive: {entry!r}")
            packets.append((time, duration))
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"unexpected packet listing: {error!r}") from error
    return [
        duration * numerator / denominator for _, duration in sorted(packets)
    ]


def read_container_durations(
    path: Path, timeout_s: float = TIMEOUT_S
) -> list[float] | None:
    """Lists how long the container shows each frame of a video.

    Only the packets of the first video stream are read, not decoded, so
    this takes a fraction of a second. It never raises: a missing ffprobe, a
    failed or slow run, or a packet list it cannot use all give None.

    The video is named to ffprobe as ``file:`` and its absolute path, so a
    name such as ``concat:...``, ``subfile,...`` or ``http://...`` is read as
    the file it is and not as a protocol, and a name starting with a dash is
    not read as an option. ffprobe is also limited to the ``file`` protocol
    and the MP4 and MOV demuxer: the ``file`` protocol alone would still let a
    playlist or script saved under an ``.mp4`` name make it read other files.

    Args:
        path: Video file path.
        timeout_s: Seconds ffprobe may take.

    Returns:
        The durations in seconds in presentation order, or None when they
        could not be read.
    """
    executable = ffprobe_path()
    if executable is None:
        return None
    argv = [executable, "-v", "error", "-select_streams", "v:0"]
    argv += ["-show_entries", "packet=pts,duration,flags:stream=time_base"]
    argv += ["-protocol_whitelist", "file"]
    argv += ["-format_whitelist", ISO_BMFF_DEMUXERS]
    try:
        argv += ["-of", "json", "-i", f"file:{path.absolute()}"]
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
        return parse_packet_durations(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
