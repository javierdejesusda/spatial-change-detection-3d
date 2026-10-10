import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from spatial_change.capture.video import probe_video, read_frames

MakeVideo = Callable[..., Path]


def test_probe_reports_written_properties(make_video: MakeVideo) -> None:
    info = probe_video(make_video(n_frames=60))

    assert info.fps == pytest.approx(30.0)
    assert info.frame_count == 60
    assert (info.width, info.height) == (160, 120)


def test_probe_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "absent.mp4")


def test_probe_unreadable_file_raises_value_error(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")

    with pytest.raises(ValueError):
        probe_video(junk)


def test_read_frames_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(read_frames(tmp_path / "absent.mp4"))


def test_read_frames_stride_halves_count(make_video: MakeVideo) -> None:
    frames = list(read_frames(make_video(n_frames=60), stride=2))

    assert abs(len(frames) - 30) <= 1
    assert [f.index for f in frames] == list(range(0, 2 * len(frames), 2))
    stamps = [f.timestamp_s for f in frames]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    assert stamps[1] == pytest.approx(2 / 30, abs=0.01)


def test_read_frames_respects_max_frames(make_video: MakeVideo) -> None:
    frames = list(read_frames(make_video(n_frames=60), max_frames=5))

    assert len(frames) == 5


def test_read_frames_rejects_bad_stride(make_video: MakeVideo) -> None:
    with pytest.raises(ValueError):
        list(read_frames(make_video(n_frames=10), stride=0))


def test_frames_are_rgb_uint8(
    tmp_path: Path, write_video: Callable[..., None]
) -> None:
    path = tmp_path / "red.mp4"
    bgr_red = np.zeros((120, 160, 3), dtype=np.uint8)
    bgr_red[..., 2] = 255
    write_video(path, [bgr_red] * 10)

    frame = next(read_frames(path))

    assert frame.image.dtype == np.uint8
    assert frame.image.shape == (120, 160, 3)
    means = frame.image.reshape(-1, 3).mean(axis=0)
    assert means[0] > 200
    assert means[1] < 60
    assert means[2] < 60


def test_read_frames_without_timing_uses_nominal_fps(
    make_video: MakeVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_video(n_frames=3)
    real_capture = cv2.VideoCapture

    class NoTimingCapture:
        """Delegates to a real capture but reports no timing metadata.

        A plain wrapper is used because subclassing ``cv2.VideoCapture``
        crashes the interpreter when the subclass is garbage collected.
        """

        def __init__(self, filename: str) -> None:
            self._cap = real_capture(filename)

        def get(self, prop: int) -> float:
            if prop in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_POS_MSEC):
                return 0.0
            return float(self._cap.get(prop))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", NoTimingCapture)

    stamps = [frame.timestamp_s for frame in read_frames(path)]

    assert stamps == pytest.approx([0.0, 1 / 30, 2 / 30])


def _tag_rotation(source: Path, target: Path) -> bool:
    """Writes ``source`` to ``target`` tagged as rotated by 90 degrees.

    Newer ffmpeg builds ignore the legacy ``rotate`` metadata tag and need
    ``-display_rotation``, so both are tried.

    Returns:
        True if OpenCV reports rotation metadata for ``target``.
    """
    base = ["ffmpeg", "-y", "-loglevel", "error"]
    attempts = (
        [*base, "-display_rotation", "90", "-i", str(source)]
        + ["-c", "copy", str(target)],
        [*base, "-i", str(source), "-c", "copy"]
        + ["-metadata:s:v:0", "rotate=90", str(target)],
    )
    for argv in attempts:
        try:
            subprocess.run(argv, check=True)
        except subprocess.CalledProcessError:
            continue
        cap = cv2.VideoCapture(str(target))
        try:
            if cap.get(cv2.CAP_PROP_ORIENTATION_META):
                return True
        finally:
            cap.release()
    return False


def test_rotation_metadata_swaps_the_reported_and_decoded_size(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    rotated = tmp_path / "rotated.mp4"
    if not _tag_rotation(make_video(n_frames=10), rotated):
        pytest.skip("ffmpeg could not write rotation metadata OpenCV reads")

    info = probe_video(rotated)
    frame = next(read_frames(rotated))

    assert (info.width, info.height) == (120, 160)
    assert frame.image.shape == (160, 120, 3)
