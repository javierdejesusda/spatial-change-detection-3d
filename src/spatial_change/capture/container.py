"""The container format a video must have before any decoder opens it.

FFmpeg picks the demuxer for a file by sniffing its content, not by its name,
so a file called ``.mp4`` that is really an HLS playlist or an ffconcat script
makes FFmpeg read other local files or fetch network addresses. Phone
recordings are ISO base media files (MP4 and MOV), so only those are read:
the start of the file is checked here, before OpenCV is given the path.
"""

import struct
from pathlib import Path

ISO_BMFF_DEMUXERS = "mov,mp4,m4a,3gp,3g2,mj2"
"""FFmpeg's name for the one demuxer that reads ISO base media files."""

_START_BOX_TYPES = frozenset({b"ftyp", b"moov", b"mdat", b"pnot"})
"""Types of the box an MP4 or MOV file starts with.

``ftyp`` opens every MP4 and modern MOV file. Older QuickTime files open with
one of the others.
"""

_PADDING_BOX_TYPES = frozenset({b"wide", b"free", b"skip"})
"""Types of the padding boxes a file may have before its first real box."""

_MAX_PADDING_BOXES = 4
"""Most padding boxes that may come before the first real box."""

_MAX_START_OFFSET = 1024
"""Furthest byte at which the first real box may start.

FFmpeg probes a file by its first 2048 bytes, and a long run of padding lets
another demuxer score higher than the MP4 one. Real padding is an 8-byte
``wide`` box or a small ``free`` one.
"""

_BOX_HEADER_SIZE = 8
"""Bytes of a box header: a 32-bit size and a four-letter type."""

_LARGE_BOX_HEADER_SIZE = 16
"""Bytes of a header whose size is 1: the usual one and a 64-bit size."""


def require_iso_base_media(path: Path) -> None:
    """Checks that a file starts like an MP4 or MOV file.

    A box starts with its size as a 32-bit big-endian number, where 0 means
    the box runs to the end of the file and 1 means a 64-bit size follows
    the type, and then its four-letter type. The file must start with a box
    of a type an MP4 or MOV file starts with, whose 64-bit size, if it has
    one, is at least 16. Padding boxes may come first, at most four, each
    stepped over by its size, so the type that follows them is the one that
    decides. The first real box must start within the first 1024 bytes. A
    padding box must have a size of at least 8: a size of 0 would leave
    nothing after it, and the 64-bit size of 1 is not used for padding.

    Args:
        path: Video file path.

    Raises:
        ValueError: If the file does not start with a box of a type an MP4 or
            MOV file starts with.
        OSError: If the file cannot be read.
    """
    offset = 0
    with path.open("rb") as handle:
        for _ in range(_MAX_PADDING_BOXES + 1):
            handle.seek(offset)
            header = handle.read(_LARGE_BOX_HEADER_SIZE)
            if len(header) < _BOX_HEADER_SIZE:
                break
            size, kind = struct.unpack(">I4s", header[:_BOX_HEADER_SIZE])
            if kind in _START_BOX_TYPES:
                if size == 1:
                    large_size = int.from_bytes(
                        header[_BOX_HEADER_SIZE:], "big"
                    )
                    fits = large_size >= _LARGE_BOX_HEADER_SIZE
                    fits = fits and len(header) == _LARGE_BOX_HEADER_SIZE
                else:
                    fits = size == 0 or size >= _BOX_HEADER_SIZE
                if fits:
                    return
                break
            if kind not in _PADDING_BOX_TYPES or size < _BOX_HEADER_SIZE:
                break
            offset += size
            if offset > _MAX_START_OFFSET:
                break
    raise ValueError(f"not an MP4 or MOV video: {path}")
