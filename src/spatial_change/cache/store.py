"""Content-addressed on-disk store for stage artifacts.

Artifacts are named numpy arrays plus JSON metadata. Entries are keyed by
a digest of the stage name, stage version, stage config and input
fingerprints, so any change to those invalidates exactly the entries it
affects. Arrays are stored positionally and their names live in the JSON
document, so a name can be any string. Pickled data is never loaded.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue as JsonValue

_STAGE_PATTERN = re.compile(r"[a-z0-9_-]+")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARRAYS_FILE = "arrays.npz"
_META_FILE = "meta.json"
_MARKER_FILE = "COMPLETE"
_TEMP_PREFIX = ".tmp-"
_CORRUPTION_ERRORS = (
    ValueError,
    KeyError,
    EOFError,
    zipfile.BadZipFile,
    FileNotFoundError,
)


def fingerprint_file(path: Path) -> str:
    """Hashes a file's content without reading it into memory at once.

    Args:
        path: File to hash.

    Returns:
        The sha256 hex digest of the file content.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint_array(array: NDArray[Any]) -> str:
    """Hashes an array's dtype, shape and C-ordered bytes.

    Args:
        array: Array to hash.

    Returns:
        The sha256 hex digest.
    """
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _check_stage(stage: str) -> None:
    if _STAGE_PATTERN.fullmatch(stage) is None:
        raise ValueError(
            f"Unsafe stage name {stage!r}: only [a-z0-9_-]+ is allowed."
        )


@dataclass(frozen=True)
class CacheKey:
    """Identifies one cache entry.

    Attributes:
        stage: Pipeline stage name, a safe path component.
        digest: Hex digest of stage, version, config and inputs.
    """

    stage: str
    digest: str

    def __post_init__(self) -> None:
        """Validates that the key is safe to use as a relative path.

        Raises:
            ValueError: If stage is not a safe path component or digest is
                not 64 lowercase hex characters.
        """
        _check_stage(self.stage)
        if _DIGEST_PATTERN.fullmatch(self.digest) is None:
            raise ValueError(
                f"Invalid digest {self.digest!r}: expected 64 lowercase hex "
                "characters."
            )

    @classmethod
    def build(
        cls,
        stage: str,
        version: str,
        config: Mapping[str, JsonValue],
        inputs: Mapping[str, str],
    ) -> "CacheKey":
        """Builds a key from everything that determines a stage's output.

        Args:
            stage: Pipeline stage name, matching [a-z0-9_-]+.
            version: Version of the stage implementation.
            config: JSON-serializable stage configuration.
            inputs: Logical input names mapped to their fingerprints.

        Returns:
            The cache key.

        Raises:
            ValueError: If stage is not a safe path component.
        """
        document = json.dumps(
            {
                "stage": stage,
                "version": version,
                "config": config,
                "inputs": inputs,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(stage, hashlib.sha256(document.encode("ascii")).hexdigest())


@dataclass(frozen=True)
class CachedArtifact:
    """Arrays and metadata produced by one stage run.

    Attributes:
        arrays: Named numpy arrays.
        meta: JSON metadata.
    """

    arrays: dict[str, NDArray[Any]]
    meta: dict[str, JsonValue]


class StageCache:
    """Directory-backed cache of stage artifacts."""

    def __init__(self, root: Path) -> None:
        """Initializes the cache.

        Args:
            root: Directory holding all entries; created on first write.
        """
        self._root = root

    def path_for(self, key: CacheKey) -> Path:
        """Returns the entry directory for a key.

        Args:
            key: Cache key.

        Returns:
            root / stage / digest[:2] / digest.
        """
        return self._root / key.stage / key.digest[:2] / key.digest

    def get(self, key: CacheKey) -> CachedArtifact | None:
        """Loads an entry.

        Incomplete or corrupt entries, including entries written in an
        older layout, are removed and reported as misses.

        Args:
            key: Cache key.

        Returns:
            The artifact, or None on a miss.

        Raises:
            OSError: On other I/O failures such as PermissionError; the
                entry is kept because it may still be valid.
        """
        entry = self.path_for(key)
        if not entry.exists():
            return None
        try:
            return _load(entry)
        except _CORRUPTION_ERRORS:
            shutil.rmtree(entry, ignore_errors=True)
            return None

    def put(
        self,
        key: CacheKey,
        arrays: Mapping[str, NDArray[Any]],
        meta: Mapping[str, JsonValue],
    ) -> CachedArtifact:
        """Stores an entry atomically, keeping an existing valid one.

        Args:
            key: Cache key.
            arrays: Named arrays; names may be any string. Object dtype is
                not allowed.
            meta: JSON-serializable metadata.

        Returns:
            The artifact now stored under the key.

        Raises:
            TypeError: If an array has object dtype or meta is not JSON
                serializable.
            OSError: If the entry is unreadable after the write.
        """
        for name, array in arrays.items():
            if array.dtype.hasobject:
                raise TypeError(f"Array {name!r} has object dtype.")
        document = json.dumps({"arrays": list(arrays), "meta": dict(meta)})
        entry = self.path_for(key)
        existing = self.get(key)
        if existing is not None:
            return existing
        entry.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX, dir=entry.parent))
        try:
            np.savez(temp / _ARRAYS_FILE, *arrays.values())
            (temp / _META_FILE).write_text(document)
            (temp / _MARKER_FILE).write_text("")
            try:
                os.replace(temp, entry)
            except OSError:
                # A concurrent writer won the race: keep its entry.
                if not entry.exists():
                    raise
        finally:
            shutil.rmtree(temp, ignore_errors=True)
        stored = self.get(key)
        if stored is None:
            raise OSError(f"Cache entry {entry} is unreadable after write.")
        return stored

    def get_or_compute(
        self,
        key: CacheKey,
        compute: Callable[
            [], tuple[dict[str, NDArray[Any]], dict[str, JsonValue]]
        ],
    ) -> tuple[CachedArtifact, bool]:
        """Returns a cached entry, computing and storing it on a miss.

        Args:
            key: Cache key.
            compute: Produces (arrays, meta) when the entry is missing.

        Returns:
            The artifact and True if it was a cache hit.
        """
        cached = self.get(key)
        if cached is not None:
            return cached, True
        arrays, meta = compute()
        return self.put(key, arrays, meta), False

    def clear_stage(self, stage: str) -> int:
        """Removes every entry of one stage.

        Args:
            stage: Stage name.

        Returns:
            The number of entries removed.

        Raises:
            ValueError: If stage is not a safe path component.
        """
        _check_stage(stage)
        stage_dir = self._root / stage
        if not stage_dir.is_dir():
            return 0
        removed = sum(
            1
            for entry in stage_dir.glob("*/*")
            if not entry.name.startswith(_TEMP_PREFIX)
        )
        shutil.rmtree(stage_dir)
        return removed


def _load(entry: Path) -> CachedArtifact:
    if not (entry / _MARKER_FILE).is_file():
        raise ValueError("missing completion marker")
    document = json.loads((entry / _META_FILE).read_text())
    if not isinstance(document, dict):
        raise ValueError("entry document is not an object")
    names = document.get("arrays")
    meta = document.get("meta")
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("array names are not a list of strings")
    if not isinstance(meta, dict):
        raise ValueError("metadata is not an object")
    with (
        (entry / _ARRAYS_FILE).open("rb") as handle,
        np.load(handle, allow_pickle=False) as npz,
    ):
        arrays = {name: npz[f"arr_{i}"] for i, name in enumerate(names)}
        if len(arrays) != len(npz.files):
            raise ValueError("array names do not match stored arrays")
    return CachedArtifact(arrays=arrays, meta=meta)
