import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from spatial_change.capture import durations
from spatial_change.capture.durations import (
    ffprobe_path,
    parse_packet_durations,
    read_container_durations,
)

MakeVideo = Callable[..., Path]


# What `ffprobe -v error -select_streams v:0 -show_entries
# packet=pts,duration,flags:stream=time_base -of json` printed for a phone
# video whose edit list cuts off its first frames (ffprobe 8.0.1). Nine of
# its 213 packets are listed here: four carry the D flag and start at a
# negative time, and the other five are kept, one of which lasts 21 ticks
# instead of 20.
EDIT_LIST_LISTING = """{
    "packets": [
        {
            "pts": -280,
            "duration": 80,
            "flags": "KD_"
        },
        {
            "pts": -200,
            "duration": 20,
            "flags": "_D_"
        },
        {
            "pts": -240,
            "duration": 60,
            "flags": "_D_"
        },
        {
            "pts": 40,
            "duration": 20,
            "flags": "___"
        },
        {
            "pts": 0,
            "duration": 20,
            "flags": "___"
        },
        {
            "pts": -20,
            "duration": 20,
            "flags": "_D_"
        },
        {
            "pts": 20,
            "duration": 20,
            "flags": "___"
        },
        {
            "pts": 280,
            "duration": 20,
            "flags": "K__"
        },
        {
            "pts": 620,
            "duration": 21,
            "flags": "___"
        }
    ],
    "programs": [

    ],
    "stream_groups": [

    ],
    "streams": [
        {
            "time_base": "1/600",
            "side_data_list": [
                {

                },
                {

                },
                {

                }
            ]
        }
    ]
}
"""

# The same command on a phone video at about 30 fps that dropped frames
# (ffprobe 8.0.1), cut to eight of its 296 packets. They come in decoding
# order, not presentation order. The frame at 158080 lasts 640 ticks but the
# next one comes 1280 later, and the frame after that lasts 1280 for a gap of
# 640: the container charges the dropped frame to the packet after it.
DROPPED_FRAME_LISTING = """{
    "packets": [
        {
            "pts": 154240,
            "duration": 640,
            "flags": "___"
        },
        {
            "pts": 156800,
            "duration": 1280,
            "flags": "___"
        },
        {
            "pts": 155520,
            "duration": 1280,
            "flags": "___"
        },
        {
            "pts": 159360,
            "duration": 1280,
            "flags": "___"
        },
        {
            "pts": 158080,
            "duration": 640,
            "flags": "___"
        },
        {
            "pts": 160640,
            "duration": 640,
            "flags": "___"
        },
        {
            "pts": 160000,
            "duration": 640,
            "flags": "___"
        },
        {
            "pts": 161920,
            "duration": 640,
            "flags": "___"
        }
    ],
    "programs": [

    ],
    "stream_groups": [

    ],
    "streams": [
        {
            "time_base": "1/19200"
        }
    ]
}
"""


def listing(
    packets: list[dict[str, Any]], time_base: str | None = "1/600"
) -> str:
    """Writes a packet list the way ``ffprobe -of json`` does."""
    streams = [] if time_base is None else [{"time_base": time_base}]
    return json.dumps({"packets": packets, "streams": streams})


def packet(pts: int, duration: int, flags: str = "___") -> dict[str, Any]:
    return {"pts": pts, "duration": duration, "flags": flags}


def test_packets_become_durations_in_presentation_order() -> None:
    output = listing([packet(60, 20), packet(0, 40, "K__"), packet(20, 30)])

    assert parse_packet_durations(output) == [40 / 600, 30 / 600, 20 / 600]


def test_durations_are_whole_ticks_of_the_stream_time_base() -> None:
    output = listing([packet(0, 640, "K__"), packet(640, 641)], "1/19200")

    assert parse_packet_durations(output) == [640 / 19200, 641 / 19200]


def test_the_time_base_may_be_a_ratio_of_larger_numbers() -> None:
    output = listing([packet(0, 1001)], "1001/30000")

    assert parse_packet_durations(output) == [1001 * 1001 / 30000]


def test_discarded_packets_are_dropped() -> None:
    output = listing(
        [
            packet(-280, 80, "KD_"),
            packet(-200, 20, "_D_"),
            packet(40, 20),
            packet(0, 30),
            packet(-20, 20, "_D_"),
        ]
    )

    assert parse_packet_durations(output) == [30 / 600, 20 / 600]


def test_packets_without_flags_are_all_kept() -> None:
    output = listing([{"pts": 0, "duration": 5}, {"pts": 5, "duration": 7}])

    assert parse_packet_durations(output) == [5 / 600, 7 / 600]


def test_no_packet_gives_no_duration() -> None:
    assert parse_packet_durations(listing([])) == []


def test_a_discarded_packet_without_a_duration_is_not_an_error() -> None:
    output = listing([{"pts": -5, "flags": "_D_"}, packet(0, 20, "K__")])

    assert parse_packet_durations(output) == [20 / 600]


def test_a_real_listing_with_an_edit_list_drops_the_cut_packets() -> None:
    durations = parse_packet_durations(EDIT_LIST_LISTING)

    assert durations == [20 / 600] * 4 + [21 / 600]


def test_a_real_listing_with_dropped_frames_gives_packet_durations() -> None:
    durations = parse_packet_durations(DROPPED_FRAME_LISTING)

    assert durations == [
        640 / 19200,
        1280 / 19200,
        1280 / 19200,
        640 / 19200,
        1280 / 19200,
        640 / 19200,
        640 / 19200,
        640 / 19200,
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {"pts": 0, "flags": "___"},
        {"duration": 20, "flags": "___"},
        packet(0, 0),
        packet(0, -20),
        {"pts": "N/A", "duration": 20},
        {"pts": 0, "duration": None},
        {"pts": 0.5, "duration": 20},
        {"pts": True, "duration": 20},
        {"pts": 0, "duration": "twenty"},
    ],
    ids=[
        "no duration",
        "no time",
        "zero duration",
        "negative duration",
        "pts not a number",
        "duration null",
        "pts fraction",
        "pts boolean",
        "duration word",
    ],
)
def test_a_packet_without_a_usable_time_or_duration_is_an_error(
    bad: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        parse_packet_durations(listing([packet(0, 20), bad]))


@pytest.mark.parametrize(
    "output",
    [
        "",
        "not json",
        "[]",
        "{}",
        json.dumps({"packets": 3, "streams": [{"time_base": "1/600"}]}),
        json.dumps({"packets": [3], "streams": [{"time_base": "1/600"}]}),
        listing([packet(0, 20)], None),
        listing([packet(0, 20)], "0/1"),
        listing([packet(0, 20)], "1/0"),
        listing([packet(0, 20)], "-1/600"),
        listing([packet(0, 20)], "fast"),
        listing([packet(0, 20)], "1/2/3"),
    ],
    ids=[
        "empty",
        "not json",
        "list",
        "object without lists",
        "packets not a list",
        "packet not an object",
        "no stream",
        "zero numerator",
        "zero denominator",
        "negative numerator",
        "time base word",
        "time base three parts",
    ],
)
def test_a_listing_that_cannot_be_read_is_an_error(output: str) -> None:
    with pytest.raises(ValueError):
        parse_packet_durations(output)


PROGRAM = "ffprobe.exe" if os.name == "nt" else "ffprobe"


def fake_ffprobe(folder: Path) -> Path:
    """Puts an executable file called ffprobe in a folder."""
    folder.mkdir(parents=True, exist_ok=True)
    program = folder / PROGRAM
    program.write_text("", encoding="utf-8")
    program.chmod(0o755)
    return program


def same_file(found: str | None, expected: Path) -> bool:
    """Tells whether a lookup gave a path, whatever the case on Windows."""
    return found is not None and Path(found) == expected


@pytest.fixture(name="search")
def search_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., Path]:
    """Runs from an empty folder, with the lookup shortcuts of Windows off.

    Returns:
        A function that sets ``PATH`` to the given entries and gives the
        folder the process runs from.
    """
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)

    def use(*entries: str) -> Path:
        monkeypatch.setenv("PATH", os.pathsep.join(entries))
        return here

    return use


def test_ffprobe_is_the_first_one_in_the_folders_of_the_path(
    tmp_path: Path, search: Callable[..., Path]
) -> None:
    first = fake_ffprobe(tmp_path / "first")
    fake_ffprobe(tmp_path / "second")
    search(
        str(tmp_path / "missing"), str(first.parent), str(tmp_path / "second")
    )

    assert same_file(ffprobe_path(), first)


def test_ffprobe_is_absent_when_no_folder_of_the_path_has_it(
    tmp_path: Path, search: Callable[..., Path]
) -> None:
    (tmp_path / "empty").mkdir()
    search(str(tmp_path / "empty"))

    assert ffprobe_path() is None


def test_without_a_path_there_is_no_ffprobe(
    monkeypatch: pytest.MonkeyPatch, search: Callable[..., Path]
) -> None:
    fake_ffprobe(search())
    monkeypatch.delenv("PATH")

    assert ffprobe_path() is None


def test_ffprobe_in_the_current_directory_is_not_found(
    tmp_path: Path, search: Callable[..., Path]
) -> None:
    (tmp_path / "bin").mkdir()
    fake_ffprobe(search(str(tmp_path / "bin")))

    assert ffprobe_path() is None


def test_ffprobe_on_the_path_wins_over_the_current_directory(
    tmp_path: Path, search: Callable[..., Path]
) -> None:
    installed = fake_ffprobe(tmp_path / "bin")
    fake_ffprobe(search(str(installed.parent)))

    found = ffprobe_path()

    assert same_file(found, installed)
    assert found is not None and Path(found).is_absolute()


@pytest.mark.parametrize("entry", [".", "", "sub", "../here/sub"])
def test_a_relative_folder_of_the_path_is_not_searched(
    tmp_path: Path, search: Callable[..., Path], entry: str
) -> None:
    here = search(entry, str(tmp_path / "missing"))
    fake_ffprobe(here)
    fake_ffprobe(here / "sub")

    assert ffprobe_path() is None


def test_a_folder_called_ffprobe_is_not_the_program(
    tmp_path: Path, search: Callable[..., Path]
) -> None:
    (tmp_path / "bin" / PROGRAM).mkdir(parents=True)
    search(str(tmp_path / "bin"))

    assert ffprobe_path() is None


@pytest.mark.skipif(os.name == "nt", reason="Windows has no execute bit")
def test_a_file_that_cannot_be_run_is_not_the_program(
    tmp_path: Path, search: Callable[..., Path]
) -> None:
    fake_ffprobe(tmp_path / "bin").chmod(0o644)
    search(str(tmp_path / "bin"))

    assert ffprobe_path() is None


@pytest.mark.skipif(os.name != "nt", reason="batch files run on Windows")
def test_a_batch_file_is_never_the_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search: Callable[..., Path],
) -> None:
    (tmp_path / "bin").mkdir()
    for name in ("ffprobe.bat", "ffprobe.cmd"):
        (tmp_path / "bin" / name).write_text("", encoding="utf-8")
    search(str(tmp_path / "bin"))
    monkeypatch.setenv("PATHEXT", ".BAT;.CMD;.EXE")

    assert ffprobe_path() is None


@pytest.mark.skipif(os.name != "nt", reason="batch files run on Windows")
def test_an_exe_is_found_behind_a_batch_file_in_an_earlier_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search: Callable[..., Path],
) -> None:
    (tmp_path / "first").mkdir()
    (tmp_path / "first" / "ffprobe.bat").write_text("", encoding="utf-8")
    installed = fake_ffprobe(tmp_path / "second")
    search(str(tmp_path / "first"), str(installed.parent))
    monkeypatch.setenv("PATHEXT", ".BAT;.CMD;.EXE")

    assert same_file(ffprobe_path(), installed)


class FakeRun:
    """Stands in for ``subprocess.run`` and records how it was called."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> object:
        self.calls.append((argv, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@pytest.fixture(name="present")
def present_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(durations, "ffprobe_path", lambda: "ffprobe")


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["ffprobe"], 0, stdout=stdout)


@pytest.mark.usefixtures("present")
def test_ffprobe_lists_the_video_stream_packets_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeRun(completed(listing([packet(0, 20, "K__"), packet(20, 20)])))
    monkeypatch.setattr(subprocess, "run", run)

    result = read_container_durations(Path("-odd name.mp4"), timeout_s=7.5)

    assert result == [20 / 600, 20 / 600]
    ((argv, kwargs),) = run.calls
    assert argv[0] == "ffprobe"
    assert argv[argv.index("-select_streams") + 1] == "v:0"
    entries = argv[argv.index("-show_entries") + 1]
    assert entries == "packet=pts,duration,flags:stream=time_base"
    assert argv[argv.index("-of") + 1] == "json"
    given = argv[argv.index("-i") + 1]
    assert given.startswith("file:")
    assert Path(given.removeprefix("file:")).is_absolute()
    assert Path(given.removeprefix("file:")).name == "-odd name.mp4"
    assert kwargs["timeout"] == 7.5
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs.get("shell", False) is False


@pytest.mark.usefixtures("present")
def test_ffprobe_is_limited_to_files_and_the_mp4_and_mov_demuxers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeRun(completed(listing([packet(0, 20, "K__")])))
    monkeypatch.setattr(subprocess, "run", run)

    read_container_durations(Path("a.mp4"))

    ((argv, _),) = run.calls
    assert argv[argv.index("-protocol_whitelist") + 1] == "file"
    demuxers = argv[argv.index("-format_whitelist") + 1]
    assert demuxers == "mov,mp4,m4a,3gp,3g2,mj2"
    assert argv.index("-protocol_whitelist") < argv.index("-i")
    assert argv.index("-format_whitelist") < argv.index("-i")


@pytest.mark.usefixtures("present")
def test_a_working_folder_that_is_gone_gives_no_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gone(self: Path) -> Path:
        raise FileNotFoundError("the working folder is gone")

    monkeypatch.setattr(Path, "absolute", gone)
    monkeypatch.setattr(subprocess, "run", FakeRun(completed("")))

    assert read_container_durations(Path("a.mp4")) is None


def test_without_ffprobe_nothing_is_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeRun(completed(""))
    monkeypatch.setattr(durations, "ffprobe_path", lambda: None)
    monkeypatch.setattr(subprocess, "run", run)

    assert read_container_durations(Path("a.mp4")) is None
    assert run.calls == []


@pytest.mark.usefixtures("present")
@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired("ffprobe", 60.0),
        subprocess.CalledProcessError(1, "ffprobe"),
        FileNotFoundError("ffprobe"),
        PermissionError("ffprobe"),
        completed(listing([{"pts": 0, "flags": "K__"}])),
        completed("not a packet list"),
        completed(""),
    ],
    ids=[
        "timeout",
        "exit status",
        "missing",
        "denied",
        "no duration",
        "junk",
        "empty",
    ],
)
def test_a_failing_ffprobe_gives_no_durations_and_no_error(
    monkeypatch: pytest.MonkeyPatch, failure: object
) -> None:
    monkeypatch.setattr(subprocess, "run", FakeRun(failure))

    assert read_container_durations(Path("a.mp4")) is None


@pytest.mark.skipif(ffprobe_path() is None, reason="no ffprobe")
def test_a_real_clip_has_one_duration_per_frame(make_video: MakeVideo) -> None:
    path = make_video(n_frames=12)

    result = read_container_durations(path)

    assert result is not None
    assert len(result) == 12
    assert result == pytest.approx([1 / 30] * 12, abs=1e-4)


@pytest.mark.skipif(ffprobe_path() is None, reason="no ffprobe")
def test_a_real_clip_is_read_by_a_relative_name_that_looks_like_an_option(
    make_video: MakeVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_video(n_frames=12, name="-odd & name.mp4")
    monkeypatch.chdir(path.parent)

    result = read_container_durations(Path(path.name))

    assert result is not None
    assert len(result) == 12


@pytest.mark.skipif(ffprobe_path() is None, reason="no ffprobe")
def test_ffprobe_does_not_follow_a_script_to_a_real_clip(
    script_naming_a_clip: Path,
) -> None:
    assert read_container_durations(script_naming_a_clip) is None


@pytest.mark.skipif(ffprobe_path() is None, reason="no ffprobe")
def test_a_playlist_or_script_saved_as_mp4_gives_no_durations_and_no_error(
    crafted_video: Path,
) -> None:
    assert read_container_durations(crafted_video) is None
