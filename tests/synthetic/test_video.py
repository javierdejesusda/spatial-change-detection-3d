from pathlib import Path

import numpy as np
import pytest

from spatial_change.capture.video import probe_video, read_frames
from spatial_change.domain import SE3, Epoch, Frame, Intrinsics
from spatial_change.synthetic.video import write_video

INTRINSICS = Intrinsics(fx=40.0, fy=40.0, cx=31.5, cy=23.5, width=64, height=48)


SMALL_INTRINSICS = Intrinsics(
    fx=20.0, fy=20.0, cx=15.5, cy=11.5, width=32, height=24
)


def make_frame(
    index: int, image: bool = True, intrinsics: Intrinsics = INTRINSICS
) -> Frame:
    rng = np.random.default_rng(index)
    pixels = rng.integers(
        0, 256, size=(intrinsics.height, intrinsics.width, 3), dtype=np.uint8
    )
    return Frame(
        epoch=Epoch.BEFORE,
        index=index,
        timestamp_s=index / 10.0,
        intrinsics=intrinsics,
        world_from_camera=SE3.identity(),
        image=pixels if image else None,
    )


def test_written_video_round_trips_through_capture(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"

    write_video([make_frame(i) for i in range(6)], path, fps=10.0)
    info = probe_video(path)
    decoded = list(read_frames(path))

    assert info.frame_count == 6
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10.0)
    assert len(decoded) == 6
    assert decoded[0].image.shape == (48, 64, 3)


def test_write_video_rejects_empty_input_and_non_positive_fps(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        write_video([], tmp_path / "a.mp4", fps=10.0)
    with pytest.raises(ValueError):
        write_video([make_frame(0)], tmp_path / "c.mp4", fps=0.0)


def test_failed_write_leaves_no_file_behind(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    with pytest.raises(ValueError, match="no image"):
        write_video(
            [make_frame(0), make_frame(1, image=False)], missing, fps=10.0
        )
    assert not missing.exists()

    mismatched = tmp_path / "mismatched.mp4"
    with pytest.raises(ValueError, match="same size"):
        write_video(
            [make_frame(0), make_frame(1, intrinsics=SMALL_INTRINSICS)],
            mismatched,
            fps=10.0,
        )
    assert not mismatched.exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_write_keeps_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    write_video([make_frame(i) for i in range(3)], path, fps=10.0)
    original = path.read_bytes()

    with pytest.raises(ValueError):
        write_video([make_frame(0), make_frame(1, image=False)], path, 10.0)

    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["clip.mp4"]
