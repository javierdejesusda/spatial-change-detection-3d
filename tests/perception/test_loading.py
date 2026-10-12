import sys
from typing import Any

import pytest
import torch

from spatial_change.perception.loading import (
    inference_context,
    load_with_fallback,
)


class Builder:
    """Builds a fake model per id, failing for the ids in ``failing``."""

    def __init__(self, failing: dict[str, Exception]) -> None:
        self.failing = failing
        self.requested: list[str] = []

    def __call__(self, model_id: str) -> str:
        self.requested.append(model_id)
        if model_id in self.failing:
            raise self.failing[model_id]
        return f"model:{model_id}"


def test_the_primary_is_used_when_it_loads() -> None:
    build = Builder({})

    loaded, loaded_id = load_with_fallback("primary", "fallback", build)

    assert (loaded, loaded_id) == ("model:primary", "primary")
    assert build.requested == ["primary"]


def test_the_fallback_is_used_when_the_primary_raises_os_error() -> None:
    build = Builder({"primary": OSError("gated")})

    loaded, loaded_id = load_with_fallback("primary", "fallback", build)

    assert (loaded, loaded_id) == ("model:fallback", "fallback")
    assert build.requested == ["primary", "fallback"]


def test_the_error_is_raised_without_a_fallback() -> None:
    error = OSError("gated")
    build = Builder({"primary": error})

    with pytest.raises(OSError) as raised:
        load_with_fallback("primary", None, build)

    assert raised.value is error
    assert build.requested == ["primary"]


def test_both_errors_are_chained_when_the_fallback_fails_too() -> None:
    first, second = OSError("gated"), OSError("offline")
    build = Builder({"primary": first, "fallback": second})

    with pytest.raises(OSError, match="primary.*fallback") as raised:
        load_with_fallback("primary", "fallback", build)

    assert raised.value.__cause__ is second
    assert second.__context__ is first


def test_errors_other_than_os_error_do_not_trigger_the_fallback() -> None:
    build = Builder({"primary": ValueError("unsupported")})

    with pytest.raises(ValueError, match="unsupported"):
        load_with_fallback("primary", "fallback", build)

    assert build.requested == ["primary"]


def test_the_context_runs_in_inference_mode_with_cpu_autocast_off() -> None:
    seen: dict[str, Any] = {}

    with inference_context(torch.device("cpu")):
        seen["inference"] = torch.is_inference_mode_enabled()
        seen["autocast"] = torch.is_autocast_enabled("cpu")

    assert seen == {"inference": True, "autocast": False}
    assert not torch.is_inference_mode_enabled()


def test_importing_the_module_does_not_need_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "spatial_change.perception.loading")
    monkeypatch.setitem(sys.modules, "torch", None)

    import spatial_change.perception.loading as loading

    assert callable(loading.load_with_fallback)
