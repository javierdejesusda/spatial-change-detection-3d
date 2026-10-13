"""Tests for the selective SceneDiff fetcher using a local Range server."""

import http.server
import io
import json
import math
import pickle
import re
import socket
import struct
import threading
import urllib.parse
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from spatial_change.evaluation import scenediff_fetch
from spatial_change.evaluation.scenediff_fetch import (
    SCENEDIFF_REVISION,
    SCENEDIFF_URL,
    extract_pair,
    find_pair_members,
    load_split,
    open_remote_zip,
)

ROOT = "scenediff_benchmark"
PAIR_MOV = "living_room_1_living_room_2"
PAIR_MP4 = "bedroom_1_bedroom_2"
SEGMENTS = pickle.dumps({"scene_type": "x", "objects": []})


@dataclass
class _ServerState:
    data: bytes
    honor_range: bool = True
    fail_first_gets: int = 0
    fail_status: int = 503
    bad_content_range: bool = False
    reset_first_gets: int = 0
    hide_size: bool = False
    short_body_gets: int = 0
    short_body_keep: int = 0
    redirect: bool = False
    generation: int = 0
    reads_per_generation: int = 0
    reads_in_generation: int = 0
    forbid_reads: bool = False
    origin_gets: int = 0
    forbidden: int = 0
    gets: list[str] = field(default_factory=list)


def _handler(state: _ServerState) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            header = self.headers.get("Range", "")
            state.gets.append(header)
            if state.redirect and self._redirect_or_forbid(header):
                return
            if state.fail_first_gets > 0:
                state.fail_first_gets -= 1
                self.send_error(state.fail_status)
                return
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", header)
            if not state.honor_range or match is None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(state.data)))
                self.end_headers()
                self.wfile.write(state.data)
                return
            start, end = int(match[1]), int(match[2])
            end = min(end, len(state.data) - 1)
            body = state.data[start : end + 1]
            self.send_response(206)
            shown = start + 1 if state.bad_content_range and start else start
            total = "*" if state.hide_size else str(len(state.data))
            self.send_header("Content-Range", f"bytes {shown}-{end}/{total}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.short_body_gets > 0 and len(body) > 1:
                state.short_body_gets -= 1
                self.wfile.write(body[: state.short_body_keep])
                return
            if state.reset_first_gets > 0 and len(body) > 1:
                state.reset_first_gets -= 1
                self.wfile.write(body[:1])
                self.wfile.flush()
                self.connection.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_LINGER,
                    struct.pack("ii", 1, 0),
                )
                self.close_connection = True
                self.connection.close()
                return
            self.wfile.write(body)
            if state.reads_per_generation:
                state.reads_in_generation += 1
                if state.reads_in_generation >= state.reads_per_generation:
                    state.generation += 1
                    state.reads_in_generation = 0

        def _redirect_or_forbid(self, header: str) -> bool:
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/z.zip":
                state.origin_gets += 1
                self.send_response(302)
                self.send_header("Location", f"/signed?gen={state.generation}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True
            generation = int(urllib.parse.parse_qs(url.query)["gen"][0])
            if generation != state.generation or (
                state.forbid_reads and header != "bytes=0-0"
            ):
                state.forbidden += 1
                self.send_error(403)
                return True
            return False

    return Handler


def _make_zip(
    files: Mapping[str, str | bytes],
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _build_zip() -> bytes:
    files: dict[str, str | bytes] = {}
    for pair, ext in ((PAIR_MOV, "MOV"), (PAIR_MP4, "mp4")):
        base = f"{ROOT}/data/{pair}"
        files[f"{base}/original_video1.{ext}"] = f"{pair}-v1" * 50
        files[f"{base}/original_video2.{ext}"] = f"{pair}-v2" * 50
        files[f"{base}/segments.pkl"] = SEGMENTS
        files[f"{base}/video1.mp4"] = "distractor"
        files[f"{base}/video2.mp4"] = "distractor"
        files[f"{base}/inference_data.pkl"] = "distractor"
        files[f"{base}/review_meta.json"] = "{}"
    splits = {"varied": [PAIR_MOV], "kitchen": [PAIR_MP4]}
    files[f"{ROOT}/splits/val_split.json"] = json.dumps(splits)
    files[f"{ROOT}/splits/test_split.json"] = json.dumps(splits)
    return _make_zip(files)


@pytest.fixture
def served() -> Iterator[tuple[str, _ServerState]]:
    state = _ServerState(data=_build_zip())
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/z.zip", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _no_sleep(_: float) -> None:
    return None


def test_lists_members_over_http_range(
    served: tuple[str, _ServerState],
) -> None:
    url, state = served

    with open_remote_zip(url, sleep=_no_sleep) as zf:
        names = zf.namelist()

    assert f"{ROOT}/data/{PAIR_MOV}/segments.pkl" in names
    assert len(names) == 16
    assert state.gets
    assert all(h.startswith("bytes=") for h in state.gets)


def test_find_pair_members_resolves_extension_case(
    served: tuple[str, _ServerState],
) -> None:
    url, _ = served
    with open_remote_zip(url, sleep=_no_sleep) as zf:
        mov = find_pair_members(zf, PAIR_MOV)
        mp4 = find_pair_members(zf, PAIR_MP4)
        with pytest.raises(KeyError, match="nope_1_nope_2"):
            find_pair_members(zf, "nope_1_nope_2")

    assert mov.video1.endswith("original_video1.MOV")
    assert mp4.video2.endswith("original_video2.mp4")
    assert mov.segments == f"{ROOT}/data/{PAIR_MOV}/segments.pkl"


def test_load_split_parses_json(served: tuple[str, _ServerState]) -> None:
    url, _ = served
    with open_remote_zip(url, sleep=_no_sleep) as zf:
        val = load_split(zf, "val")

    assert val == {"varied": [PAIR_MOV], "kitchen": [PAIR_MP4]}


def test_extract_pair_writes_only_the_three_files(
    served: tuple[str, _ServerState], tmp_path: Path
) -> None:
    url, _ = served
    dest = tmp_path / "out"
    with open_remote_zip(url, sleep=_no_sleep) as zf:
        pair_dir = extract_pair(zf, PAIR_MOV, dest)

    assert pair_dir == dest / PAIR_MOV
    assert sorted(p.name for p in pair_dir.iterdir()) == [
        "segments.pkl",
        "video1.mov",
        "video2.mov",
    ]
    assert (pair_dir / "video1.mov").read_text() == f"{PAIR_MOV}-v1" * 50
    assert (pair_dir / "segments.pkl").read_bytes() == SEGMENTS


def test_existing_file_with_same_size_is_skipped(
    served: tuple[str, _ServerState], tmp_path: Path
) -> None:
    url, _ = served
    with open_remote_zip(url, sleep=_no_sleep) as zf:
        pair_dir = extract_pair(zf, PAIR_MP4, tmp_path)
        marked = {}
        for path in pair_dir.iterdir():
            marked[path] = b"Z" * path.stat().st_size
            path.write_bytes(marked[path])
        extract_pair(zf, PAIR_MP4, tmp_path)

    for path, content in marked.items():
        assert path.read_bytes() == content


def test_existing_file_with_different_size_is_re_extracted(
    served: tuple[str, _ServerState], tmp_path: Path
) -> None:
    url, _ = served
    with open_remote_zip(url, sleep=_no_sleep) as zf:
        pair_dir = extract_pair(zf, PAIR_MP4, tmp_path)
        (pair_dir / "video1.mp4").write_bytes(b"short")
        extract_pair(zf, PAIR_MP4, tmp_path)

    assert (pair_dir / "video1.mp4").read_text() == f"{PAIR_MP4}-v1" * 50


@pytest.mark.parametrize(
    "bad", ["../evil", "a/b", "", "pair name", "x\\y", "..", "a.b"]
)
def test_bad_pair_name_is_rejected(
    served: tuple[str, _ServerState], tmp_path: Path, bad: str
) -> None:
    url, _ = served
    with (
        open_remote_zip(url, sleep=_no_sleep) as zf,
        pytest.raises(ValueError, match="pair"),
    ):
        extract_pair(zf, bad, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_server_ignoring_range_raises(
    served: tuple[str, _ServerState],
) -> None:
    url, state = served
    state.honor_range = False

    with pytest.raises(RuntimeError, match="Range"):
        open_remote_zip(url, sleep=_no_sleep)


@pytest.mark.parametrize(
    ("faults", "counter"),
    [
        pytest.param({"fail_first_gets": 1}, "fail_first_gets", id="http-503"),
        pytest.param(
            {"reset_first_gets": 1}, "reset_first_gets", id="connection-reset"
        ),
        pytest.param(
            {"short_body_gets": 1, "short_body_keep": 0},
            "short_body_gets",
            id="short-body-empty",
        ),
        pytest.param(
            {"short_body_gets": 1, "short_body_keep": 5},
            "short_body_gets",
            id="short-body-partial",
        ),
    ],
)
def test_transient_fault_is_retried_once_with_backoff(
    served: tuple[str, _ServerState],
    tmp_path: Path,
    faults: dict[str, int],
    counter: str,
) -> None:
    url, state = served
    for name, value in faults.items():
        setattr(state, name, value)
    sleeps: list[float] = []

    with open_remote_zip(url, sleep=sleeps.append) as zf:
        pair_dir = extract_pair(zf, PAIR_MOV, tmp_path)

    # BufferedReader re-reads after a short non-empty chunk, so only the
    # retry backoff proves a short body was rejected.
    assert sleeps == [1.0]
    assert getattr(state, counter) == 0
    assert (pair_dir / "video1.mov").read_text() == f"{PAIR_MOV}-v1" * 50
    assert (pair_dir / "segments.pkl").read_bytes() == SEGMENTS


@pytest.mark.parametrize(
    ("status", "expected_gets", "expected_sleeps", "message"),
    [
        pytest.param(404, 1, 0, "404", id="404-not-retried"),
        pytest.param(503, 3, 2, "3 attempts", id="503-uses-all-attempts"),
    ],
)
def test_error_status_retry_policy(
    served: tuple[str, _ServerState],
    status: int,
    expected_gets: int,
    expected_sleeps: int,
    message: str,
) -> None:
    url, state = served
    state.fail_first_gets = 100
    state.fail_status = status
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match=message):
        open_remote_zip(url, sleep=sleeps.append)

    assert len(state.gets) == expected_gets
    assert len(sleeps) == expected_sleeps


def test_expired_signed_url_is_re_resolved_per_failing_read(
    served: tuple[str, _ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, state = served
    monkeypatch.setattr(scenediff_fetch, "_BUFFER", 256)
    monkeypatch.setattr(scenediff_fetch, "_COPY_CHUNK", 256)
    state.redirect = True
    state.reads_per_generation = 3

    with open_remote_zip(url, sleep=_no_sleep) as zf:
        pair_dir = extract_pair(zf, PAIR_MOV, tmp_path)

    assert (pair_dir / "video1.mov").read_text() == f"{PAIR_MOV}-v1" * 50
    assert (pair_dir / "segments.pkl").read_bytes() == SEGMENTS
    assert state.generation >= 2
    assert state.forbidden >= 2
    assert state.origin_gets >= 3


def test_persistent_403_after_re_resolve_raises(
    served: tuple[str, _ServerState], tmp_path: Path
) -> None:
    url, state = served
    state.redirect = True

    with open_remote_zip(url, sleep=_no_sleep) as zf:
        state.forbid_reads = True
        with pytest.raises(RuntimeError, match="403"):
            extract_pair(zf, PAIR_MOV, tmp_path)

    assert state.origin_gets == 2


def test_403_without_redirect_is_not_re_resolved(
    served: tuple[str, _ServerState], tmp_path: Path
) -> None:
    url, state = served

    with open_remote_zip(url, sleep=_no_sleep) as zf:
        reads = len(state.gets)
        state.fail_first_gets = 100
        state.fail_status = 403
        with pytest.raises(RuntimeError, match="403"):
            extract_pair(zf, PAIR_MOV, tmp_path)

    assert len(state.gets) == reads + 1


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        pytest.param("hide_size", "total size", id="unknown-total-size"),
        pytest.param(
            "bad_content_range", "Content-Range", id="wrong-range-start"
        ),
    ],
)
def test_malformed_range_response_raises_clear_error(
    served: tuple[str, _ServerState], fault: str, message: str
) -> None:
    url, state = served
    setattr(state, fault, True)

    with pytest.raises(RuntimeError, match=message):
        open_remote_zip(url, sleep=_no_sleep)


def test_failed_copy_leaves_no_part_or_final_file(
    served: tuple[str, _ServerState], tmp_path: Path
) -> None:
    url, state = served
    payload = b"payload-" * 50
    base = f"{ROOT}/data/{PAIR_MP4}"
    raw = bytearray(
        _make_zip(
            {
                f"{base}/original_video1.mp4": payload,
                f"{base}/original_video2.mp4": payload,
                f"{base}/segments.pkl": SEGMENTS,
            },
            zipfile.ZIP_STORED,
        )
    )
    raw[raw.index(payload)] ^= 0xFF
    state.data = bytes(raw)

    with (
        open_remote_zip(url, sleep=_no_sleep) as remote,
        pytest.raises(zipfile.BadZipFile, match="CRC"),
    ):
        extract_pair(remote, PAIR_MP4, tmp_path)

    assert list((tmp_path / PAIR_MP4).iterdir()) == []


@pytest.mark.parametrize(
    ("limit_name", "message", "leftover"),
    [
        pytest.param(
            "MAX_PICKLE_BYTES", "segments.pkl", "segments.pkl", id="pickle"
        ),
        pytest.param(
            "MAX_VIDEO_BYTES", "original_video1", "video1.mp4", id="video"
        ),
    ],
)
def test_oversized_member_is_rejected_without_output(
    served: tuple[str, _ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    message: str,
    leftover: str,
) -> None:
    url, _ = served
    monkeypatch.setattr(scenediff_fetch, limit_name, 10)

    with (
        open_remote_zip(url, sleep=_no_sleep) as zf,
        pytest.raises(ValueError, match=message),
    ):
        extract_pair(zf, PAIR_MP4, tmp_path)

    names = [p.name for p in (tmp_path / PAIR_MP4).iterdir()]
    assert leftover not in names
    assert not any(name.endswith(".tmp") for name in names)


def test_url_is_pinned_to_a_dataset_revision() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", SCENEDIFF_REVISION)
    assert f"/resolve/{SCENEDIFF_REVISION}/" in SCENEDIFF_URL


def test_non_video_extension_is_not_selected(
    served: tuple[str, _ServerState],
) -> None:
    url, state = served
    pair = "x_1_x_2"
    base = f"{ROOT}/data/{pair}"
    state.data = _make_zip(
        {
            f"{base}/original_video1.exe": "a",
            f"{base}/original_video1.txt": "a",
            f"{base}/original_video2.MOV": "b",
            f"{base}/segments.pkl": SEGMENTS,
        },
        zipfile.ZIP_STORED,
    )

    with (
        open_remote_zip(url, sleep=_no_sleep) as remote,
        pytest.raises(KeyError, match="original_video1"),
    ):
        find_pair_members(remote, pair)


def test_copy_chunk_sets_the_range_request_size(
    served: tuple[str, _ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, state = served
    buffer, copy_chunk, size = 256, 64 << 10, 256 << 10
    monkeypatch.setattr(scenediff_fetch, "_BUFFER", buffer)
    monkeypatch.setattr(scenediff_fetch, "_COPY_CHUNK", copy_chunk)
    base = f"{ROOT}/data/{PAIR_MP4}"
    video = bytes(range(256)) * (size // 256)
    state.data = _make_zip(
        {
            f"{base}/original_video1.mp4": video,
            f"{base}/original_video2.mp4": b"v2",
            f"{base}/segments.pkl": SEGMENTS,
        },
        zipfile.ZIP_STORED,
    )

    with open_remote_zip(url, sleep=_no_sleep) as zf:
        opened = len(state.gets)
        pair_dir = extract_pair(zf, PAIR_MP4, tmp_path)

    assert (pair_dir / "video1.mp4").read_bytes() == video
    # Each copy chunk costs one pass-through read plus one buffer refill.
    assert len(state.gets) - opened <= 2 * math.ceil(size / copy_chunk) + 4
