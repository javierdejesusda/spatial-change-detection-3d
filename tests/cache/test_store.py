import io
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.cache import (
    CachedArtifact,
    CacheKey,
    JsonValue,
    StageCache,
    fingerprint_array,
    fingerprint_file,
)
from spatial_change.cache import store as store_module

BASE: dict[str, Any] = {
    "stage": "recon",
    "version": "1",
    "config": {"a": 1, "b": [1, 2]},
    "inputs": {"video": "f" * 64},
}


def make_key(**overrides: Any) -> CacheKey:
    return CacheKey.build(**{**BASE, **overrides})


def load(cache: StageCache, key: CacheKey) -> CachedArtifact:
    got = cache.get(key)
    assert got is not None
    return got


def named_members_npz() -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, a=np.zeros(1), b=np.zeros(1), c=np.zeros(1), d=np.zeros(1))
    return buffer.getvalue()


def sample_arrays() -> dict[str, NDArray[Any]]:
    rng = np.random.default_rng(0)
    return {
        "f32": rng.random((3, 4)).astype(np.float32),
        "u8": rng.integers(0, 255, (5,), dtype=np.uint8),
        "flags": np.array([True, False, True]),
        "scalar": np.array(7.5),
    }


def test_fingerprint_file_stable_and_content_sensitive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v.bin"
    path.write_bytes(b"x" * 3_000_000)
    first = fingerprint_file(path)
    assert fingerprint_file(path) == first
    assert len(first) == 64
    path.write_bytes(b"x" * 1_500_000 + b"y" + b"x" * 1_499_999)
    assert fingerprint_file(path) != first


def test_fingerprint_array_depends_on_dtype_and_shape() -> None:
    base = np.zeros(4, dtype=np.float32)
    assert fingerprint_array(base) == fingerprint_array(base.copy())
    assert fingerprint_array(base) != fingerprint_array(base.view(np.int32))
    assert fingerprint_array(base) != fingerprint_array(base.reshape(2, 2))


def test_fingerprint_array_ignores_memory_layout() -> None:
    arr = np.arange(6).reshape(2, 3)
    assert fingerprint_array(arr) == fingerprint_array(np.asfortranarray(arr))


def test_key_is_order_independent() -> None:
    one = CacheKey.build("s", "1", {"a": 1, "b": 2}, {"x": "1", "y": "2"})
    two = CacheKey.build("s", "1", {"b": 2, "a": 1}, {"y": "2", "x": "1"})
    assert one == two


@pytest.mark.parametrize(
    "other",
    [
        make_key(stage="seg"),
        make_key(version="2"),
        make_key(config={"a": 2, "b": [1, 2]}),
        make_key(inputs={"video": "e" * 64}),
    ],
)
def test_key_changes_with_any_component(other: CacheKey) -> None:
    assert other.digest != make_key().digest


@pytest.mark.parametrize("stage", ["", "..", "../x", "A", "a/b", "a b", "é"])
def test_unsafe_stage_rejected(stage: str) -> None:
    with pytest.raises(ValueError):
        CacheKey(stage, "0" * 64)


@pytest.mark.parametrize("digest", ["", "../x", "0" * 63, "A" * 64, "g" * 64])
def test_directly_built_key_with_bad_digest_rejected(digest: str) -> None:
    with pytest.raises(ValueError):
        CacheKey("recon", digest)


def test_round_trip(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    arrays = sample_arrays()
    meta: dict[str, JsonValue] = {"n": 3, "tags": ["a", None], "ok": True}
    cache.put(key, arrays, MappingProxyType(meta))
    got = load(cache, key)
    assert got.meta == meta
    assert got.arrays.keys() == arrays.keys()
    for name, arr in arrays.items():
        assert got.arrays[name].dtype == arr.dtype
        assert got.arrays[name].shape == arr.shape
        np.testing.assert_array_equal(got.arrays[name], arr)


def test_any_string_is_a_valid_array_name(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    names = ["a", "file", "allow_pickle", "a.npy", "a/b", "é", ""]
    arrays = {name: np.full(2, float(i)) for i, name in enumerate(names)}
    cache.put(key, arrays, {})
    got = load(cache, key)
    assert list(got.arrays) == names
    for name, arr in arrays.items():
        np.testing.assert_array_equal(got.arrays[name], arr)


def test_path_layout(tmp_path: Path) -> None:
    key = make_key()
    expected = tmp_path / "recon" / key.digest[:2] / key.digest
    assert StageCache(tmp_path).path_for(key) == expected


def test_miss_returns_none(tmp_path: Path) -> None:
    assert StageCache(tmp_path).get(make_key()) is None


def test_get_or_compute_hits_second_time(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    calls = 0

    def compute() -> tuple[dict[str, NDArray[Any]], dict[str, JsonValue]]:
        nonlocal calls
        calls += 1
        return {"a": np.arange(3)}, {"k": 1}

    first, hit1 = cache.get_or_compute(make_key(), compute)
    second, hit2 = cache.get_or_compute(make_key(), compute)
    assert (hit1, hit2) == (False, True)
    assert calls == 1
    np.testing.assert_array_equal(first.arrays["a"], second.arrays["a"])
    assert second.meta == {"k": 1}


@pytest.mark.parametrize(
    ("filename", "replacement"),
    [
        pytest.param("COMPLETE", None, id="missing-marker"),
        pytest.param("arrays.npz", None, id="missing-npz"),
        pytest.param(
            "arrays.npz", b"PK\x03\x04" + bytes(6), id="truncated-npz"
        ),
        pytest.param("meta.json", b"{not json", id="malformed-json"),
        pytest.param(
            "arrays.npz", named_members_npz(), id="members-not-positional"
        ),
        pytest.param("meta.json", b"[]", id="document-not-object"),
        pytest.param(
            "meta.json",
            b'{"arrays": [1, 2, 3, 4], "meta": {}}',
            id="names-not-strings",
        ),
        pytest.param(
            "meta.json",
            b'{"arrays": ["f32"], "meta": {}}',
            id="names-count-differs-from-members",
        ),
        pytest.param(
            "meta.json",
            b'{"arrays": ["f32", "f32", "u8", "flags"], "meta": {}}',
            id="names-duplicated",
        ),
        pytest.param(
            "meta.json",
            b'{"arrays": ["f32", "u8", "flags", "scalar"], "meta": []}',
            id="meta-not-object",
        ),
    ],
)
def test_corrupt_entry_is_miss_removed_and_recoverable(
    tmp_path: Path, filename: str, replacement: bytes | None
) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    cache.put(key, sample_arrays(), {"n": 1})
    entry = cache.path_for(key)
    if replacement is None:
        (entry / filename).unlink()
    else:
        (entry / filename).write_bytes(replacement)
    assert cache.get(key) is None
    assert not entry.exists()
    cache.put(key, sample_arrays(), {"n": 2})
    assert load(cache, key).meta == {"n": 2}


@pytest.mark.parametrize("error", [PermissionError, MemoryError])
def test_transient_load_error_propagates_and_keeps_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    cache.put(key, sample_arrays(), {"n": 1})

    def failing_load(entry: Path) -> CachedArtifact:
        raise error("transient")

    monkeypatch.setattr(store_module, "_load", failing_load)
    with pytest.raises(error):
        cache.get(key)
    assert cache.path_for(key).exists()


def test_object_dtype_rejected_on_put(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    bad = {"o": np.array([{"a": 1}, None], dtype=object)}
    with pytest.raises(TypeError):
        cache.put(make_key(), bad, {})
    assert cache.get(make_key()) is None


@pytest.mark.parametrize("meta", [{"x": object()}, {"s": {1, 2}}])
def test_unserializable_meta_rejected_without_creating_directories(
    tmp_path: Path, meta: dict[str, Any]
) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    with pytest.raises(TypeError):
        cache.put(key, {"a": np.zeros(2)}, meta)
    assert cache.get(key) is None
    assert list(tmp_path.rglob(".tmp-*")) == []
    assert not (tmp_path / "recon").exists()


def test_pickled_npz_is_never_loaded(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    entry = cache.path_for(key)
    entry.mkdir(parents=True)
    np.savez(entry / "arrays.npz", np.array([{"a": 1}], dtype=object))
    (entry / "meta.json").write_text('{"arrays": ["o"], "meta": {}}')
    (entry / "COMPLETE").write_text("")
    assert cache.get(key) is None
    assert not entry.exists()


@pytest.mark.parametrize("race", [False, True])
def test_put_keeps_first_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: bool
) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    cache.put(key, {"a": np.zeros(2)}, {"v": "first"})
    if race:
        real_get = cache.get
        calls = 0

        def get_missing_once(candidate: CacheKey) -> CachedArtifact | None:
            nonlocal calls
            calls += 1
            return None if calls == 1 else real_get(candidate)

        monkeypatch.setattr(cache, "get", get_missing_once)
    result = cache.put(key, {"a": np.ones(2)}, {"v": "second"})
    assert result.meta == {"v": "first"}
    got = load(cache, key)
    assert got.meta == {"v": "first"}
    np.testing.assert_array_equal(got.arrays["a"], np.zeros(2))
    parent = cache.path_for(key).parent
    assert [p.name for p in parent.iterdir()] == [key.digest]


def test_clear_stage_only_removes_that_stage(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    keys = [make_key(config={"i": i}) for i in range(3)]
    other = make_key(stage="seg")
    for key in [*keys, other]:
        cache.put(key, {"a": np.zeros(1)}, {})
    assert cache.clear_stage("recon") == 3
    assert all(cache.get(key) is None for key in keys)
    assert cache.get(other) is not None
    assert cache.clear_stage("recon") == 0


def test_clear_stage_rejects_unsafe_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        StageCache(tmp_path).clear_stage("../x")


def test_discard_removes_only_that_entry(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    first, second = make_key(config={"i": 1}), make_key(config={"i": 2})
    for key in (first, second):
        cache.put(key, {"a": np.zeros(1)}, {})

    cache.discard(first)

    assert cache.get(first) is None
    assert cache.get(second) is not None


def test_discarding_a_missing_entry_does_nothing(tmp_path: Path) -> None:
    StageCache(tmp_path).discard(make_key())


def test_discard_does_not_raise_when_the_entry_cannot_be_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = StageCache(tmp_path)
    key = make_key()
    cache.put(key, {"a": np.zeros(1)}, {})

    def locked(path: Path, ignore_errors: bool = False) -> None:
        if not ignore_errors:
            raise PermissionError("locked")

    monkeypatch.setattr(store_module.shutil, "rmtree", locked)

    cache.discard(key)
