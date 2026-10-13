from collections.abc import Callable
from pathlib import Path

import pytest

from spatial_change.capture.container import require_iso_base_media

MakeBox = Callable[..., bytes]

REFUSED = r"not an MP4 or MOV video: .*"
DASH_MANIFEST = b'<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static">'
MPEG_TS = b"\x47" + b"\0" * 187


def write(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(content)
    return path


@pytest.mark.parametrize("kind", [b"ftyp", b"moov", b"mdat", b"pnot"])
def test_a_file_starting_with_an_mp4_or_mov_box_is_accepted(
    tmp_path: Path, box: MakeBox, kind: bytes
) -> None:
    require_iso_base_media(write(tmp_path, box(kind)))


@pytest.mark.parametrize("size", [0, 8, 2**24 - 1])
def test_a_box_that_runs_to_the_end_or_is_under_16_mib_is_accepted(
    tmp_path: Path, box: MakeBox, size: int
) -> None:
    require_iso_base_media(write(tmp_path, box(b"mdat", size)))


@pytest.mark.parametrize("kind", [b"ftyp", b"moov", b"mdat", b"pnot"])
@pytest.mark.parametrize("size", [2**24, 2**32 - 1])
def test_a_first_box_of_16_mib_or_more_is_refused(
    tmp_path: Path, box: MakeBox, kind: bytes, size: int
) -> None:
    path = write(tmp_path, box(kind, size))

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


def test_a_manifest_whose_text_reads_as_an_ftyp_box_is_refused(
    tmp_path: Path,
) -> None:
    content = b"<!--ftypjp2 -->\n" + DASH_MANIFEST + b"<BaseURL>http://x/"
    path = write(tmp_path, content)

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


@pytest.mark.parametrize("largesize", [16, 2**40])
def test_a_box_with_a_64_bit_size_of_at_least_16_is_accepted(
    tmp_path: Path, box: MakeBox, largesize: int
) -> None:
    payload = largesize.to_bytes(8, "big") + bytes(8)

    require_iso_base_media(write(tmp_path, box(b"mdat", 1, payload)))


@pytest.mark.parametrize("largesize", [0, 8, 15])
def test_a_box_with_a_64_bit_size_below_16_is_refused(
    tmp_path: Path, box: MakeBox, largesize: int
) -> None:
    payload = largesize.to_bytes(8, "big") + bytes(8)
    path = write(tmp_path, box(b"mdat", 1, payload))

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


@pytest.mark.parametrize("cut", [bytes(4), b"\xff" * 4])
def test_a_box_with_a_64_bit_size_that_is_cut_short_is_refused(
    tmp_path: Path, box: MakeBox, cut: bytes
) -> None:
    path = write(tmp_path, box(b"mdat", 1, cut))

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


@pytest.mark.parametrize(
    ("skipped", "kind"),
    [(b"wide", b"mdat"), (b"free", b"ftyp"), (b"skip", b"moov")],
)
def test_a_padding_box_is_stepped_over_to_the_box_after_it(
    tmp_path: Path, box: MakeBox, skipped: bytes, kind: bytes
) -> None:
    require_iso_base_media(write(tmp_path, box(skipped) + box(kind)))


def test_a_padding_box_of_only_a_header_is_stepped_over(
    tmp_path: Path, box: MakeBox
) -> None:
    require_iso_base_media(write(tmp_path, box(b"wide", 8, b"") + box(b"mdat")))


def test_up_to_four_padding_boxes_may_come_first(
    tmp_path: Path, box: MakeBox
) -> None:
    require_iso_base_media(write(tmp_path, box(b"free") * 4 + box(b"ftyp")))


def test_five_padding_boxes_are_refused(tmp_path: Path, box: MakeBox) -> None:
    path = write(tmp_path, box(b"free") * 5 + box(b"ftyp"))

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


@pytest.mark.parametrize(
    "payload",
    [DASH_MANIFEST, MPEG_TS, b"#EXTM3U\n", b"\0" * 24],
    ids=["dash manifest", "mpeg ts", "hls playlist", "zeros"],
)
def test_a_padding_box_followed_by_another_format_is_refused(
    tmp_path: Path, box: MakeBox, payload: bytes
) -> None:
    path = write(tmp_path, box(b"free") + payload)

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


def test_a_padding_box_with_nothing_after_it_is_refused(
    tmp_path: Path, box: MakeBox
) -> None:
    path = write(tmp_path, box(b"free"))

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


def test_padding_that_ends_at_the_limit_is_accepted(
    tmp_path: Path, box: MakeBox
) -> None:
    padding = box(b"free", 1024, bytes(1016))

    require_iso_base_media(write(tmp_path, padding + box(b"ftyp")))


@pytest.mark.parametrize(
    "payload",
    [(MPEG_TS * 22)[:4088], bytes(4088), bytes(1017)],
    ids=["mpeg ts packets", "zeros", "one byte past the limit"],
)
def test_padding_that_runs_past_the_limit_is_refused(
    tmp_path: Path, box: MakeBox, payload: bytes
) -> None:
    padding = box(b"free", 8 + len(payload), payload)
    path = write(tmp_path, padding + box(b"ftyp"))

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


@pytest.mark.parametrize("size", [0, 1, 4, 7, 1000, 2**32 - 1])
def test_a_padding_box_that_cannot_be_stepped_over_is_refused(
    tmp_path: Path, box: MakeBox, size: int
) -> None:
    path = write(tmp_path, box(b"free", size) + box(b"ftyp"))

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


def test_a_playlist_or_script_saved_as_mp4_is_refused(
    crafted_video: Path,
) -> None:
    with pytest.raises(ValueError, match=REFUSED + r"evil\.mp4"):
        require_iso_base_media(crafted_video)


@pytest.mark.parametrize(
    "content",
    [
        b"this is not a video",
        b"RIFF\x24\0\0\0AVI LIST",
        b"\x1a\x45\xdf\xa3" + b"\0" * 16,
        MPEG_TS,
        b"\0\0\0\x18abcd" + b"\0" * 16,
        b"\0\0\0\x04ftyp" + b"\0" * 16,
        b"\0\0\0\x07ftyp",
        b"\0\0\0\x18fty",
        b"",
    ],
    ids=[
        "text",
        "avi",
        "matroska",
        "mpeg ts",
        "unknown box type",
        "size 4",
        "size 7",
        "header cut short",
        "empty",
    ],
)
def test_any_other_start_is_refused_with_the_file_named(
    tmp_path: Path, content: bytes
) -> None:
    path = write(tmp_path, content)

    with pytest.raises(ValueError, match=REFUSED + r"clip\.mp4"):
        require_iso_base_media(path)


def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        require_iso_base_media(tmp_path / "absent.mp4")
