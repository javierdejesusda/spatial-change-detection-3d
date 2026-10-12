"""Selective download of SceneDiff pairs from a remote zip archive.

The benchmark is one 11.9 GB zip, but evaluation needs a few pairs. The
fetcher reads the zip central directory with HTTP Range requests and
extracts only the chosen members.
"""

import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Buffer, Callable, Sequence
from dataclasses import dataclass
from http.client import IncompleteRead
from pathlib import Path
from typing import Literal

from spatial_change.evaluation.scenediff import indoor_pairs

SCENEDIFF_REVISION = "d9e2115fae13daf219ed547bfb072a875aae4d7c"
SCENEDIFF_URL = (
    f"https://huggingface.co/datasets/yuqun/SceneDiff/resolve/{SCENEDIFF_REVISION}/"
    "scenediff_benchmark.zip"
)

DEMO_PAIRS = (
    "living_room_3_living_room_4",
    "living_room_37_living_room_38",
    "living_room_9_living_room_10",
    "bedroom_28_bedroom_29",
    "closet_1_closet_2",
)

_ROOT = "scenediff_benchmark"
_PAIR_RE = re.compile(r"[A-Za-z0-9_-]+")
_PROBE_RANGE_RE = re.compile(r"bytes 0-0/(\d+)")
_VIDEO_EXTS = frozenset({".mp4", ".mov"})
_BUFFER = 1 << 20
_COPY_CHUNK = 8 << 20
MAX_PICKLE_BYTES = 64 << 20
MAX_VIDEO_BYTES = 4 << 30
_TIMEOUT_S = 60.0
_ATTEMPTS = 3
_RETRY_STATUS_MIN = 500
_TOO_MANY_REQUESTS = 429
_FORBIDDEN = 403


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests."""

    def __init__(
        self,
        url: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Learns the size and final URL with a one-byte Range request.

        The size comes from the ``Content-Range`` header of a ``bytes=0-0``
        request, because a HEAD request is reissued as a GET after a
        redirect. The final URL is cached for later reads.

        Args:
            url: URL of the resource. Redirects are followed.
            sleep: Function used to wait between retries.

        Raises:
            RuntimeError: If the server does not honor Range requests, does
                not report the total size, or the request keeps failing.
        """
        super().__init__()
        self._sleep = sleep
        self._pos = 0
        self._origin = url
        self._url = url
        self._size = 0
        self._resolve()

    def _resolve(self) -> None:
        _, header, url = self._retry(
            lambda: self._range_get(self._origin, 0, 0)
        )
        match = _PROBE_RANGE_RE.fullmatch(header)
        if match is None:
            raise RuntimeError(
                f"cannot read the total size from Content-Range {header!r}"
            )
        self._url, self._size = url, int(match[1])

    def _retry[T](self, op: Callable[[], T]) -> T:
        for attempt in range(_ATTEMPTS):
            try:
                return op()
            except urllib.error.HTTPError as exc:
                if (
                    exc.code != _TOO_MANY_REQUESTS
                    and exc.code < _RETRY_STATUS_MIN
                ):
                    # zipfile swallows OSError (HTTPError is one) as BadZipFile.
                    raise RuntimeError(
                        f"HTTP {exc.code} for {exc.url}"
                    ) from exc
                self._backoff(attempt, exc)
            except (OSError, IncompleteRead) as exc:
                self._backoff(attempt, exc)
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int, exc: Exception) -> None:
        if attempt == _ATTEMPTS - 1:
            # zipfile swallows OSError (URLError is one) as BadZipFile.
            raise RuntimeError(
                f"HTTP request failed after {_ATTEMPTS} attempts: {exc}"
            ) from exc
        self._sleep(2.0**attempt)

    def readable(self) -> bool:
        """Returns True."""
        return True

    def seekable(self) -> bool:
        """Returns True."""
        return True

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Moves the offset.

        Args:
            offset: Offset relative to ``whence``.
            whence: ``os.SEEK_SET``, ``os.SEEK_CUR`` or ``os.SEEK_END``.

        Returns:
            The new absolute offset.

        Raises:
            ValueError: If ``whence`` is unknown or the result is negative.
        """
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._pos + offset
        elif whence == os.SEEK_END:
            target = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self._pos = target
        return target

    def readinto(self, buffer: Buffer) -> int:
        """Reads up to ``len(buffer)`` bytes at the current offset.

        Args:
            buffer: Writable buffer to fill.

        Returns:
            Number of bytes read, 0 at end of file.

        Raises:
            RuntimeError: If the server ignores the Range header, or keeps
                answering with an HTTP error.
        """
        view = memoryview(buffer).cast("B")
        end = min(self._pos + len(view), self._size) - 1
        if end < self._pos:
            return 0
        start = self._pos
        data = self._retry(lambda: self._get(start, end))
        view[: len(data)] = data
        self._pos += len(data)
        return len(data)

    @staticmethod
    def _range_get(url: str, start: int, end: int) -> tuple[bytes, str, str]:
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp:
            if resp.status != 206:
                raise RuntimeError(
                    f"server answered {resp.status} to a Range request; "
                    "Range support is required"
                )
            length = end - start + 1
            data = resp.read(length)
            if len(data) != length:
                # A clean close mid-body reads as EOF instead of raising.
                raise IncompleteRead(data, length - len(data))
            return data, resp.headers.get("Content-Range", ""), resp.geturl()

    def _get(self, start: int, end: int) -> bytes:
        try:
            data, content_range, _ = self._range_get(self._url, start, end)
        except urllib.error.HTTPError as exc:
            if exc.code != _FORBIDDEN or self._url == self._origin:
                raise
            # A signed redirect target expires, so resolve it again once.
            self._resolve()
            data, content_range, _ = self._range_get(self._url, start, end)
        expected = f"bytes {start}-"
        if not content_range.startswith(expected):
            raise RuntimeError(
                f"bad Content-Range {content_range!r}, expected {expected!r}"
            )
        return data


def open_remote_zip(
    url: str = SCENEDIFF_URL,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> zipfile.ZipFile:
    """Opens a remote zip, reading only the parts that are needed.

    Args:
        url: URL of the zip archive.
        sleep: Function used to wait between HTTP retries.

    Returns:
        An open ``ZipFile`` backed by HTTP Range requests.
    """
    return zipfile.ZipFile(
        io.BufferedReader(HttpRangeFile(url, sleep=sleep), _BUFFER)
    )


@dataclass(frozen=True)
class PairMembers:
    """Archive member paths needed for one pair.

    Attributes:
        video1: Member path of the first original video.
        video2: Member path of the second original video.
        segments: Member path of the segmentation annotations.
    """

    video1: str
    video2: str
    segments: str


def _check_pair(pair: str) -> None:
    if _PAIR_RE.fullmatch(pair) is None:
        raise ValueError(f"invalid pair name: {pair!r}")


def find_pair_members(zf: zipfile.ZipFile, pair: str) -> PairMembers:
    """Locates the members of a pair, ignoring extension case.

    Args:
        zf: Open archive.
        pair: Pair directory name.

    Returns:
        The member paths for the pair.

    Raises:
        KeyError: If any needed member is missing.
    """
    prefix = f"{_ROOT}/data/{pair}/"
    names = zf.namelist()

    def original(index: int) -> str:
        stem = f"original_video{index}."
        for name in names:
            leaf = name[len(prefix) :]
            if (
                name.startswith(prefix)
                and leaf.lower().startswith(stem)
                and "/" not in leaf
                and Path(leaf).suffix.lower() in _VIDEO_EXTS
            ):
                return name
        raise KeyError(f"{pair}: missing original_video{index}")

    segments = prefix + "segments.pkl"
    if segments not in names:
        raise KeyError(f"{pair}: missing segments.pkl")
    return PairMembers(original(1), original(2), segments)


def load_split(
    zf: zipfile.ZipFile, split: Literal["val", "test"]
) -> dict[str, list[str]]:
    """Reads a split file.

    Args:
        zf: Open archive.
        split: Split name.

    Returns:
        Mapping from scene category to pair names.
    """
    raw = zf.read(f"{_ROOT}/splits/{split}_split.json")
    splits: dict[str, list[str]] = json.loads(raw)
    return splits


def _extract_member(
    zf: zipfile.ZipFile, member: str, dest: Path, limit: int
) -> None:
    info = zf.getinfo(member)
    if info.file_size > limit:
        raise ValueError(
            f"{member}: declared size {info.file_size} exceeds limit {limit}"
        )
    if dest.is_file() and dest.stat().st_size == info.file_size:
        return
    tmp = dest.with_name(dest.name + ".part")
    try:
        with zf.open(info) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, _COPY_CHUNK)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def extract_pair(zf: zipfile.ZipFile, pair: str, dest_root: Path) -> Path:
    """Extracts the videos and annotations of one pair.

    Files already present with the expected size are skipped, so an
    interrupted run can be resumed. Output names are derived from the
    validated pair name, never from archive member names.

    Args:
        zf: Open archive.
        pair: Pair directory name.
        dest_root: Directory that receives ``<pair>/``.

    Returns:
        The pair directory.

    Raises:
        ValueError: If the pair name contains characters other than
            letters, digits, underscore and hyphen, or a member's declared
            size exceeds the limit for its kind.
        KeyError: If the pair is missing from the archive.
    """
    _check_pair(pair)
    members = find_pair_members(zf, pair)
    pair_dir = dest_root / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        (
            members.video1,
            "video1" + Path(members.video1).suffix.lower(),
            MAX_VIDEO_BYTES,
        ),
        (
            members.video2,
            "video2" + Path(members.video2).suffix.lower(),
            MAX_VIDEO_BYTES,
        ),
        (members.segments, "segments.pkl", MAX_PICKLE_BYTES),
    )
    for member, name, limit in targets:
        _extract_member(zf, member, pair_dir / name, limit)
    return pair_dir


def fetch_pairs(
    pairs: Sequence[str], dest_root: Path, *, url: str = SCENEDIFF_URL
) -> list[Path]:
    """Downloads the given pairs from the remote archive.

    Args:
        pairs: Pair directory names.
        dest_root: Directory that receives one folder per pair.
        url: URL of the zip archive.

    Returns:
        The pair directories, in input order.

    Raises:
        RuntimeError: If the server does not support Range requests or an
            HTTP request keeps failing.
        ValueError: If a pair name is invalid or a member exceeds its size
            limit.
        KeyError: If a pair is missing from the archive.
    """
    with open_remote_zip(url) as zf:
        return [extract_pair(zf, pair, dest_root) for pair in pairs]


def fetch_indoor_split(
    split: Literal["val", "test"],
    dest_root: Path,
    *,
    url: str = SCENEDIFF_URL,
) -> list[Path]:
    """Downloads the indoor pairs of a split.

    Args:
        split: ``val`` (28 pairs) or ``test`` (92 pairs).
        dest_root: Directory that receives one folder per pair.
        url: URL of the zip archive.

    Returns:
        The pair directories, in split order.

    Raises:
        RuntimeError: If the server does not support Range requests or an
            HTTP request keeps failing.
        ValueError: If a pair name is invalid or a member exceeds its size
            limit.
        KeyError: If a pair is missing from the archive.
    """
    with open_remote_zip(url) as zf:
        pairs = indoor_pairs(load_split(zf, split))
        return [extract_pair(zf, pair, dest_root) for pair in pairs]
