import dataclasses
import gc
import weakref
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from spatial_change.perception.dino import (
    DINOV2_MODEL_ID,
    DINOV3_MODEL_ID,
    DinoConfig,
    DinoFeatureEncoder,
)
from spatial_change.ports import FeatureEncoder

HIDDEN = 5
SENTINEL = 999.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_CPU = DinoConfig(device="cpu")


class FakeDino(nn.Module):
    """Stand-in for a Hugging Face DINO backbone.

    Patch token ``(i, j)`` carries ``i * 1000 + j`` in channel 0 and the
    mean normalized pixel value of its patch in channels 1 to 3, so the
    tests can read both the grid order and the preprocessing from the
    output. The prefix tokens hold a sentinel.
    """

    instances: ClassVar[list["FakeDino"]] = []
    requested: ClassVar[list[str]] = []
    failing: ClassVar[dict[str, Exception]] = {}
    patch_sizes: ClassVar[dict[str, int]] = {}
    load_kwargs: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, patch_size: int, registers: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            patch_size=patch_size,
            num_register_tokens=registers,
            hidden_size=HIDDEN,
        )
        self.eval_called = False
        self.inputs: list[torch.Tensor] = []
        self.modes: list[tuple[bool, bool]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeDino":
        cls.requested.append(model_id)
        cls.load_kwargs.append(kwargs)
        if model_id in cls.failing:
            raise cls.failing[model_id]
        registers = 0 if model_id == DINOV2_MODEL_ID else 4
        model = cls(cls.patch_sizes.get(model_id, 16), registers)
        cls.instances.append(model)
        return model

    def eval(self) -> "FakeDino":
        self.eval_called = True
        return self

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.inputs.append(pixel_values.clone())
        self.modes.append(
            (
                torch.is_inference_mode_enabled(),
                torch.is_autocast_enabled("cpu"),
            )
        )
        p = self.config.patch_size
        batch, _, height, width = pixel_values.shape
        rows, cols = height // p, width // p
        pooled = nn.functional.avg_pool2d(pixel_values, p)
        tokens = torch.zeros(batch, rows * cols, HIDDEN)
        for i in range(rows):
            for j in range(cols):
                tokens[:, i * cols + j, 0] = i * 1000 + j
                tokens[:, i * cols + j, 1:4] = pooled[:, :, i, j]
        prefix = torch.full(
            (batch, 1 + self.config.num_register_tokens, HIDDEN), SENTINEL
        )
        return SimpleNamespace(last_hidden_state=torch.cat([prefix, tokens], 1))


@pytest.fixture
def fake_dino(monkeypatch: pytest.MonkeyPatch) -> type[FakeDino]:
    import transformers

    FakeDino.instances.clear()
    FakeDino.requested.clear()
    FakeDino.failing.clear()
    FakeDino.patch_sizes.clear()
    FakeDino.load_kwargs.clear()
    monkeypatch.setattr(transformers, "AutoModel", FakeDino)
    return FakeDino


def _image(height: int = 32, width: int = 48) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), np.uint8)


def _normalized(rgb: tuple[int, int, int]) -> NDArray[np.float32]:
    return (np.array(rgb, np.float32) / 255.0 - MEAN) / STD


def test_config_defaults() -> None:
    config = DinoConfig()

    assert config.model_id == DINOV3_MODEL_ID
    assert config.fallback_model_id == DINOV2_MODEL_ID
    assert config.device == "cuda"
    assert DINOV3_MODEL_ID == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert DINOV2_MODEL_ID == "facebook/dinov2-base"


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DinoConfig().device = "cpu"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"device": ""},
        {"fallback_model_id": ""},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DinoConfig(**overrides)


def test_config_allows_no_fallback() -> None:
    assert DinoConfig(fallback_model_id=None).fallback_model_id is None


def test_encoder_satisfies_the_port() -> None:
    assert isinstance(DinoFeatureEncoder(), FeatureEncoder)


@pytest.fixture
def model_must_not_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: DinoFeatureEncoder) -> None:
        raise AssertionError("the model must not load for invalid input")

    monkeypatch.setattr(DinoFeatureEncoder, "load", fail)


@pytest.mark.usefixtures("model_must_not_load")
@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8), np.uint8),
        np.zeros((8, 8, 4), np.uint8),
        np.zeros((8, 8, 3), np.float32),
        np.zeros((0, 8, 3), np.uint8),
        np.zeros((8, 0, 3), np.uint8),
    ],
)
def test_invalid_images_are_rejected_without_loading(
    image: NDArray[Any],
) -> None:
    with pytest.raises(ValueError, match="image"):
        DinoFeatureEncoder().encode(image)


def test_nothing_is_loaded_before_first_use(fake_dino: type[FakeDino]) -> None:
    encoder = DinoFeatureEncoder(_CPU)

    assert encoder.loaded_model_id is None
    assert fake_dino.requested == []


def test_load_requests_the_configured_model(fake_dino: type[FakeDino]) -> None:
    encoder = DinoFeatureEncoder(DinoConfig(model_id="org/m", device="cpu"))

    encoder.load()

    assert fake_dino.requested == ["org/m"]
    assert encoder.loaded_model_id == "org/m"
    assert fake_dino.instances[0].eval_called


@pytest.mark.parametrize(
    ("height", "width", "patch", "grid"),
    [(294, 518, 16, (19, 33)), (294, 518, 14, (21, 37)), (32, 48, 16, (2, 3))],
)
def test_output_grid_covers_the_padded_image(
    fake_dino: type[FakeDino],
    height: int,
    width: int,
    patch: int,
    grid: tuple[int, int],
) -> None:
    fake_dino.patch_sizes[DINOV3_MODEL_ID] = patch

    features = DinoFeatureEncoder(_CPU).encode(_image(height, width))

    assert features.shape == (*grid, HIDDEN)
    assert features.dtype == np.float32


def test_prefix_tokens_are_dropped_and_the_order_is_row_major(
    fake_dino: type[FakeDino],
) -> None:
    features = DinoFeatureEncoder(_CPU).encode(_image(32, 48))

    assert not np.any(features == SENTINEL)
    expected = np.add.outer(np.arange(2) * 1000, np.arange(3))
    np.testing.assert_array_equal(features[:, :, 0], expected)


def test_a_model_without_register_tokens_drops_only_the_cls_token(
    fake_dino: type[FakeDino],
) -> None:
    config = DinoConfig(model_id=DINOV2_MODEL_ID, device="cpu")
    fake_dino.patch_sizes[DINOV2_MODEL_ID] = 14

    features = DinoFeatureEncoder(config).encode(_image(28, 28))

    assert features.shape == (2, 2, HIDDEN)
    assert not np.any(features == SENTINEL)


def test_input_is_edge_padded_scaled_and_normalized(
    fake_dino: type[FakeDino],
) -> None:
    image = np.zeros((20, 20, 3), np.uint8)
    image[:, :] = (255, 0, 0)
    image[19, 19] = (0, 255, 0)

    features = DinoFeatureEncoder(_CPU).encode(image)

    (recorded,) = fake_dino.instances[0].inputs
    assert recorded.shape == (1, 3, 32, 32)
    assert recorded.dtype == torch.float32
    red = _normalized((255, 0, 0))
    green = _normalized((0, 255, 0))
    for row, col, want in [
        (0, 0, red),
        (19, 19, green),
        (31, 19, green),
        (19, 31, green),
        (31, 5, red),
    ]:
        np.testing.assert_allclose(
            recorded[0, :, row, col].numpy(), want, atol=1e-5
        )
    np.testing.assert_allclose(features[0, 0, 1:4], red, atol=1e-5)


def test_weights_load_in_float32_whatever_the_checkpoint_dtype(
    fake_dino: type[FakeDino],
) -> None:
    DinoFeatureEncoder(_CPU).load()

    assert fake_dino.load_kwargs == [{"dtype": torch.float32}]


def test_a_view_with_negative_strides_is_encoded(
    fake_dino: type[FakeDino],
) -> None:
    image = np.zeros((32, 48, 3), np.uint8)
    image[:, :] = (10, 20, 30)

    features = DinoFeatureEncoder(_CPU).encode(image[..., ::-1])

    np.testing.assert_allclose(
        features[0, 0, 1:4], _normalized((30, 20, 10)), atol=1e-5
    )


def test_inference_runs_without_autocast_on_cpu(
    fake_dino: type[FakeDino],
) -> None:
    DinoFeatureEncoder(_CPU).encode(_image())

    assert fake_dino.instances[0].modes == [(True, False)]


def test_patch_size_loads_lazily_and_reports_the_loaded_model(
    fake_dino: type[FakeDino],
) -> None:
    fake_dino.patch_sizes[DINOV3_MODEL_ID] = 16
    encoder = DinoFeatureEncoder(_CPU)

    assert fake_dino.requested == []
    assert encoder.patch_size == 16
    assert fake_dino.requested == [DINOV3_MODEL_ID]


def test_the_fallback_is_used_when_the_primary_cannot_be_loaded(
    fake_dino: type[FakeDino],
) -> None:
    fake_dino.failing[DINOV3_MODEL_ID] = OSError("gated repo")
    fake_dino.patch_sizes[DINOV2_MODEL_ID] = 14
    encoder = DinoFeatureEncoder(_CPU)

    features = encoder.encode(_image(28, 28))

    assert fake_dino.requested == [DINOV3_MODEL_ID, DINOV2_MODEL_ID]
    assert encoder.loaded_model_id == DINOV2_MODEL_ID
    assert encoder.patch_size == 14
    assert features.shape == (2, 2, HIDDEN)


def test_the_error_is_raised_without_a_fallback(
    fake_dino: type[FakeDino],
) -> None:
    fake_dino.failing[DINOV3_MODEL_ID] = OSError("gated repo")
    encoder = DinoFeatureEncoder(
        DinoConfig(device="cpu", fallback_model_id=None)
    )

    with pytest.raises(OSError, match="gated repo"):
        encoder.encode(_image())

    assert fake_dino.requested == [DINOV3_MODEL_ID]
    assert encoder.loaded_model_id is None


def test_both_errors_are_chained_when_the_fallback_fails_too(
    fake_dino: type[FakeDino],
) -> None:
    primary = OSError("gated repo")
    secondary = OSError("offline")
    fake_dino.failing[DINOV3_MODEL_ID] = primary
    fake_dino.failing[DINOV2_MODEL_ID] = secondary

    with pytest.raises(OSError) as raised:
        DinoFeatureEncoder(_CPU).load()

    assert raised.value.__cause__ is secondary
    assert secondary.__context__ is primary
    assert DINOV3_MODEL_ID in str(raised.value)
    assert DINOV2_MODEL_ID in str(raised.value)


def test_errors_other_than_os_error_do_not_trigger_the_fallback(
    fake_dino: type[FakeDino],
) -> None:
    fake_dino.failing[DINOV3_MODEL_ID] = RuntimeError("corrupt")

    with pytest.raises(RuntimeError, match="corrupt"):
        DinoFeatureEncoder(_CPU).load()

    assert fake_dino.requested == [DINOV3_MODEL_ID]


def test_the_model_loads_once(fake_dino: type[FakeDino]) -> None:
    encoder = DinoFeatureEncoder(_CPU)

    encoder.encode(_image())
    encoder.encode(_image())
    encoder.load()

    assert len(fake_dino.requested) == 1


def test_unload_releases_the_weights_and_a_later_call_reloads(
    fake_dino: type[FakeDino],
) -> None:
    encoder = DinoFeatureEncoder(_CPU)
    encoder.encode(_image())
    ref = weakref.ref(fake_dino.instances[0])
    fake_dino.instances.clear()

    encoder.unload()
    gc.collect()

    assert ref() is None
    assert encoder.loaded_model_id is None
    encoder.encode(_image())
    assert len(fake_dino.requested) == 2


def test_unload_before_load_does_nothing(fake_dino: type[FakeDino]) -> None:
    encoder = DinoFeatureEncoder(_CPU)

    encoder.unload()
    encoder.unload()

    assert fake_dino.requested == []
