"""Tests for the restricted unpickler."""

import collections
import pickle
from pathlib import Path

import pytest

from spatial_change.evaluation.safe_pickle import safe_load, safe_loads


def _annotations() -> dict[str, object]:
    rle = {"size": [4, 6], "counts": "abc123"}
    return {
        "scene_type": "living_room",
        "objects": [{"id": 1, "name": "mug"}],
        "video1_objects": {"1": {"0": rle, "5": rle}},
        "video2_objects": {},
        "comments": None,
        "light_change": False,
    }


class _Evil:
    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self) -> tuple[object, tuple[str, str]]:
        return (open, (str(self.target), "w"))


def test_builtin_payload_round_trips(tmp_path: Path) -> None:
    payload = _annotations()
    data = pickle.dumps(payload)

    assert safe_loads(data) == payload

    path = tmp_path / "segments.pkl"
    path.write_bytes(data)
    assert safe_load(path) == payload


def test_payload_is_live_under_plain_pickle(tmp_path: Path) -> None:
    control = tmp_path / "control"

    pickle.loads(pickle.dumps(_Evil(control))).close()  # noqa: S301

    assert control.exists()


def test_malicious_pickle_is_rejected_without_side_effect(
    tmp_path: Path,
) -> None:
    target = tmp_path / "pwned"
    data = pickle.dumps(_Evil(target))

    with pytest.raises(pickle.UnpicklingError, match="open"):
        safe_loads(data)

    assert not target.exists()


def test_any_class_reference_is_rejected() -> None:
    data = pickle.dumps(collections.OrderedDict(a=1))

    with pytest.raises(pickle.UnpicklingError, match="OrderedDict"):
        safe_loads(data)
