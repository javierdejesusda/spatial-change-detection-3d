import dataclasses
import gc
import warnings
import weakref
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from spatial_change.ports import Reconstructor
from spatial_change.reconstruction import pi3x as adapter_module
from spatial_change.reconstruction.pi3x import Pi3XConfig, Pi3XReconstructor

_CPU = Pi3XConfig(device="cpu", cap_allocator_at_free_vram=False)
FAKE_FOCAL = 600.0


def _image(width: int = 64, height: int = 48) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), np.uint8)


@pytest.fixture(autouse=True)
def pi3_attention(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Restores Pi3's attention backends, which loading may replace."""
    import pi3.models.layers.attention as module

    monkeypatch.setattr(module, "SDPBackend", module.SDPBackend)
    return module


def test_config_defaults_follow_the_benchmark_gate() -> None:
    config = Pi3XConfig()

    assert config.model_id == "yyfz233/Pi3X"
    assert config.pixel_limit == 153_000
    assert config.max_frames == 128
    assert config.head_chunk_size == 8
    assert config.device == "cuda"
    assert config.cap_allocator_at_free_vram is True


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Pi3XConfig().max_frames = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"pixel_limit": 195},
        {"max_frames": 0},
        {"head_chunk_size": 0},
        {"device": ""},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Pi3XConfig(**overrides)  # type: ignore[arg-type]


@pytest.fixture
def model_must_not_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: Pi3XReconstructor) -> None:
        raise AssertionError("the model must not load for invalid input")

    monkeypatch.setattr(Pi3XReconstructor, "load", fail)


def test_adapter_satisfies_the_reconstructor_protocol() -> None:
    assert isinstance(Pi3XReconstructor(), Reconstructor)


@pytest.mark.usefixtures("model_must_not_load")
class TestInputValidation:
    def test_rejects_no_images(self) -> None:
        with pytest.raises(ValueError, match="at least one image"):
            Pi3XReconstructor().reconstruct([])

    def test_rejects_more_images_than_max_frames(self) -> None:
        adapter = Pi3XReconstructor(Pi3XConfig(max_frames=2))

        with pytest.raises(ValueError, match="max_frames"):
            adapter.reconstruct([_image()] * 3)

    def test_accepts_exactly_max_frames_images(self) -> None:
        adapter = Pi3XReconstructor(Pi3XConfig(max_frames=2))

        with pytest.raises(AssertionError, match="must not load"):
            adapter.reconstruct([_image()] * 2)

    def test_rejects_images_of_different_sizes(self) -> None:
        with pytest.raises(ValueError, match="same size"):
            Pi3XReconstructor().reconstruct([_image(64, 48), _image(48, 64)])

    @pytest.mark.parametrize(
        "image",
        [
            np.zeros((48, 64), np.uint8),
            np.zeros((48, 64, 4), np.uint8),
            np.zeros((48, 64, 3), np.float32),
        ],
    )
    def test_rejects_images_that_are_not_rgb_uint8(
        self, image: NDArray[np.generic]
    ) -> None:
        with pytest.raises(ValueError, match="image"):
            Pi3XReconstructor().reconstruct([image])


class FakePi3X(nn.Module):
    """Small stand-in with the module layout and hooks the adapter touches."""

    instances: ClassVar[list["FakePi3X"]] = []
    events: ClassVar[list[str]] = []

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.ModuleList([nn.Linear(2, 2)])
        self.point_decoder = nn.Linear(2, 2)
        self.camera_decoder = nn.Linear(2, 2)
        self.conf_decoder = nn.Linear(2, 2)
        self.metric_decoder = nn.Linear(2, 2)
        self.register_token = nn.Parameter(torch.zeros(1, 1, 5, 4))
        self.metric_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.point_head = nn.Linear(2, 2)
        self.conf_head = nn.Linear(2, 2)
        self.depth_encoder = nn.Linear(2, 2)
        self.model_id = ""
        self.device_arg: object = None
        self.multimodal_kwargs: dict[str, object] = {}
        self.eval_called = False
        self.forward_inputs: list[torch.Tensor] = []
        self.forward_modes: list[tuple[bool, bool, torch.dtype]] = []

    @classmethod
    def from_pretrained(cls, model_id: str) -> "FakePi3X":
        cls.events.append("from_pretrained")
        model = cls()
        model.model_id = model_id
        cls.instances.append(model)
        return model

    def eval(self) -> "FakePi3X":
        self.eval_called = True
        return self

    def disable_multimodal(self, free_cuda_cache: bool = True) -> None:
        self.multimodal_kwargs = {"free_cuda_cache": free_cuda_cache}
        del self.depth_encoder

    def to(self, *args: Any, **kwargs: Any) -> "FakePi3X":  # noqa: D102
        self.events.append("to")
        self.device_arg = args[0] if args else kwargs.get("device")
        return self

    def _chunked_conv_head(
        self, head: object, feat: object, ph: int, pw: int, chunk_size: int = 64
    ) -> int:
        return chunk_size

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns a plane at depth ``2 + i`` for view ``i``."""
        self.forward_inputs.append(imgs.clone())
        self.forward_modes.append(
            (
                torch.is_inference_mode_enabled(),
                torch.is_autocast_enabled("cpu"),
                torch.get_autocast_dtype("cpu"),
            )
        )
        _, n, _, h, w = imgs.shape
        poses = torch.eye(4).repeat(1, n, 1, 1)
        points = torch.zeros(1, n, h, w, 3)
        cols = torch.arange(w, dtype=torch.float32) - (w - 1) / 2
        rows = torch.arange(h, dtype=torch.float32) - (h - 1) / 2
        for i in range(n):
            z = 2.0 + i
            poses[0, i, 0, 3] = 0.25 * i
            points[0, i, :, :, 0] = cols / FAKE_FOCAL * z
            points[0, i, :, :, 1] = (rows / FAKE_FOCAL * z)[:, None]
            points[0, i, :, :, 2] = z
        return {
            "camera_poses": poses,
            "local_points": points,
            "conf": torch.full((1, n, h, w, 1), 2.0),
        }


@pytest.fixture
def fake_pi3x(monkeypatch: pytest.MonkeyPatch) -> type[FakePi3X]:
    import pi3.models.pi3x as pi3x_module

    FakePi3X.instances.clear()
    FakePi3X.events.clear()
    monkeypatch.setattr(pi3x_module, "Pi3X", FakePi3X)
    return FakePi3X


def test_load_builds_the_model_from_the_configured_repository(
    fake_pi3x: type[FakePi3X],
) -> None:
    Pi3XReconstructor(
        Pi3XConfig(
            model_id="org/model", device="cpu", cap_allocator_at_free_vram=False
        )
    ).load()

    (model,) = fake_pi3x.instances
    assert model.model_id == "org/model"
    assert model.eval_called


def test_load_frees_the_multimodal_branches_without_emptying_the_cache(
    fake_pi3x: type[FakePi3X],
) -> None:
    Pi3XReconstructor(_CPU).load()

    assert fake_pi3x.instances[0].multimodal_kwargs == {
        "free_cuda_cache": False
    }


def test_load_casts_the_transformer_stack_to_bfloat16_and_keeps_heads_fp32(
    fake_pi3x: type[FakePi3X],
) -> None:
    Pi3XReconstructor(_CPU).load()

    model = fake_pi3x.instances[0]
    for name in (
        "encoder",
        "decoder",
        "point_decoder",
        "camera_decoder",
        "conf_decoder",
        "metric_decoder",
    ):
        dtypes = {p.dtype for p in getattr(model, name).parameters()}
        assert dtypes == {torch.bfloat16}, name
    assert model.register_token.dtype == torch.bfloat16
    assert model.metric_token.dtype == torch.bfloat16
    for name in ("point_head", "conf_head"):
        dtypes = {p.dtype for p in getattr(model, name).parameters()}
        assert dtypes == {torch.float32}, name


def test_load_moves_the_model_to_the_configured_device(
    fake_pi3x: type[FakePi3X],
) -> None:
    Pi3XReconstructor(_CPU).load()

    assert fake_pi3x.instances[0].device_arg == torch.device("cpu")


def test_load_chunks_the_convolutional_heads(
    fake_pi3x: type[FakePi3X],
) -> None:
    config = Pi3XConfig(
        device="cpu", cap_allocator_at_free_vram=False, head_chunk_size=3
    )

    Pi3XReconstructor(config).load()

    model = fake_pi3x.instances[0]
    assert model._chunked_conv_head(None, None, 1, 1) == 3


@pytest.fixture
def cyclic_gc_disabled() -> Iterator[None]:
    """Leaves only reference counting to free objects."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.usefixtures("cyclic_gc_disabled")
def test_dropping_the_adapter_frees_the_model_without_a_gc_pass(
    fake_pi3x: type[FakePi3X],
) -> None:
    adapter = Pi3XReconstructor(_CPU)
    adapter.load()
    model_ref = weakref.ref(fake_pi3x.instances.pop())

    del adapter

    assert model_ref() is None


def test_load_happens_once(fake_pi3x: type[FakePi3X]) -> None:
    adapter = Pi3XReconstructor(_CPU)

    adapter.load()
    adapter.load()

    assert len(fake_pi3x.instances) == 1


def _colored_images(n: int) -> list[NDArray[np.uint8]]:
    images = []
    for i in range(n):
        image = np.zeros((360, 640, 3), np.uint8)
        image[...] = (10 + 20 * i, 100, 200)
        images.append(image)
    return images


def test_reconstruct_resizes_to_the_working_resolution_in_one_batch(
    fake_pi3x: type[FakePi3X],
) -> None:
    Pi3XReconstructor(_CPU).reconstruct(_colored_images(3))

    (batch,) = fake_pi3x.instances[0].forward_inputs
    assert batch.shape == (1, 3, 3, 294, 518)
    assert batch.dtype == torch.float32


def test_reconstruct_feeds_rgb_channels_scaled_to_the_unit_range(
    fake_pi3x: type[FakePi3X],
) -> None:
    Pi3XReconstructor(_CPU).reconstruct(_colored_images(2))

    (batch,) = fake_pi3x.instances[0].forward_inputs
    for i in range(2):
        expected = torch.tensor([10 + 20 * i, 100, 200]) / 255.0
        torch.testing.assert_close(
            batch[0, i].mean(dim=(1, 2)), expected, atol=1e-6, rtol=0
        )
        assert torch.equal(
            batch[0, i, 0], batch[0, i, 0, :1, :1].expand(294, 518)
        )


@pytest.fixture
def float16_autocast_default() -> Iterator[None]:
    """Makes bfloat16, the CPU default, differ from the ambient dtype."""
    previous = torch.get_autocast_dtype("cpu")
    torch.set_autocast_dtype("cpu", torch.float16)
    try:
        yield
    finally:
        torch.set_autocast_dtype("cpu", previous)


@pytest.mark.usefixtures("float16_autocast_default")
def test_reconstruct_runs_under_inference_mode_and_bfloat16_autocast(
    fake_pi3x: type[FakePi3X],
) -> None:
    Pi3XReconstructor(_CPU).reconstruct(_colored_images(2))

    assert fake_pi3x.instances[0].forward_modes == [
        (True, True, torch.bfloat16)
    ]


def test_reconstruct_returns_the_model_geometry(
    fake_pi3x: type[FakePi3X],
) -> None:
    result = Pi3XReconstructor(_CPU).reconstruct(_colored_images(3))

    assert len(result.depth) == 3
    for i in range(3):
        k = result.intrinsics[i]
        assert (k.width, k.height) == (518, 294)
        assert (k.fx, k.fy) == (pytest.approx(FAKE_FOCAL, rel=1e-4),) * 2
        assert result.depth[i].shape == (294, 518)
        np.testing.assert_allclose(result.depth[i], 2.0 + i)
        np.testing.assert_allclose(
            result.confidence[i], 1 / (1 + np.exp(-2.0)), atol=1e-6
        )
        np.testing.assert_allclose(
            result.world_from_camera[i].translation, [0.25 * i, 0.0, 0.0]
        )


def test_reconstruct_loads_the_model_once(fake_pi3x: type[FakePi3X]) -> None:
    adapter = Pi3XReconstructor(_CPU)

    adapter.reconstruct(_colored_images(1))
    adapter.reconstruct(_colored_images(2))

    assert len(fake_pi3x.instances) == 1
    assert len(fake_pi3x.instances[0].forward_inputs) == 2


@pytest.fixture
def empty_cache_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    return calls


def test_unload_releases_the_model_and_the_cached_gpu_memory(
    fake_pi3x: type[FakePi3X], empty_cache_calls: list[int]
) -> None:
    adapter = Pi3XReconstructor(_CPU)
    adapter.load()
    model_ref = weakref.ref(fake_pi3x.instances.pop())

    adapter.unload()

    assert model_ref() is None
    assert len(empty_cache_calls) == 1


def test_unload_is_idempotent(
    fake_pi3x: type[FakePi3X], empty_cache_calls: list[int]
) -> None:
    adapter = Pi3XReconstructor(_CPU)
    adapter.load()

    adapter.unload()
    adapter.unload()

    assert len(empty_cache_calls) == 1


def test_unload_before_load_does_nothing(
    fake_pi3x: type[FakePi3X], empty_cache_calls: list[int]
) -> None:
    Pi3XReconstructor(_CPU).unload()

    assert fake_pi3x.instances == []
    assert empty_cache_calls == []


def test_reconstruct_after_unload_loads_the_model_again(
    fake_pi3x: type[FakePi3X],
) -> None:
    adapter = Pi3XReconstructor(_CPU)
    adapter.reconstruct(_colored_images(1))

    adapter.unload()
    result = adapter.reconstruct(_colored_images(1))

    assert len(fake_pi3x.instances) == 2
    assert len(result.depth) == 1


def _set_flash_available(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: available
    )


class AttentionProbe:
    """Stands in for the kernel probe and records what it was asked."""

    def __init__(self) -> None:
        self.runnable: set[Any] = set()
        self.calls: list[tuple[Any, torch.device]] = []

    def __call__(self, backend: Any, device: torch.device) -> bool:
        self.calls.append((backend, device))
        return backend in self.runnable


@pytest.fixture
def attention_probe(monkeypatch: pytest.MonkeyPatch) -> AttentionProbe:
    """Replaces the probe so no test depends on the attached hardware.

    Every kernel runs unless a test narrows ``runnable``.
    """
    from torch.nn.attention import SDPBackend

    probe = AttentionProbe()
    probe.runnable = {
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    }
    monkeypatch.setattr(adapter_module, "_kernel_runs", probe)
    return probe


def test_load_keeps_flash_attention_when_the_probe_runs_it(
    fake_pi3x: type[FakePi3X],
    pi3_attention: Any,
    attention_probe: AttentionProbe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.nn.attention import SDPBackend

    _set_flash_available(monkeypatch, False)

    Pi3XReconstructor(_CPU).load()

    assert pi3_attention.SDPBackend is SDPBackend
    assert attention_probe.calls == [
        (SDPBackend.FLASH_ATTENTION, torch.device("cpu"))
    ]


def test_load_keeps_flash_attention_on_cpu_whatever_the_build_reports(
    fake_pi3x: type[FakePi3X],
    pi3_attention: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.nn.attention import SDPBackend

    _set_flash_available(monkeypatch, False)

    Pi3XReconstructor(_CPU).load()

    assert pi3_attention.SDPBackend is SDPBackend


@pytest.mark.parametrize(
    ("runnable", "expected"),
    [
        pytest.param(
            ("EFFICIENT_ATTENTION", "MATH"),
            "EFFICIENT_ATTENTION",
            id="efficient-when-flash-fails",
        ),
        pytest.param(("MATH",), "MATH", id="math-when-no-fused-kernel-runs"),
    ],
)
def test_load_redirects_only_the_flash_name_to_the_first_kernel_that_runs(
    fake_pi3x: type[FakePi3X],
    pi3_attention: Any,
    attention_probe: AttentionProbe,
    monkeypatch: pytest.MonkeyPatch,
    runnable: tuple[str, ...],
    expected: str,
) -> None:
    from torch.nn.attention import SDPBackend

    _set_flash_available(monkeypatch, True)
    attention_probe.runnable = {getattr(SDPBackend, name) for name in runnable}

    Pi3XReconstructor(_CPU).load()

    backends = pi3_attention.SDPBackend
    fallback = getattr(SDPBackend, expected)
    for name, member in SDPBackend.__members__.items():
        want = fallback if name == "FLASH_ATTENTION" else member
        assert getattr(backends, name) is want, name
    assert [backend for backend, _ in attention_probe.calls] == [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
    ]


def test_kernel_probe_runs_a_small_bfloat16_attention_on_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.nn.attention import SDPBackend

    seen: list[torch.Tensor] = []

    def attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        seen.append(query)
        return query

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", attention
    )

    ran = adapter_module._kernel_runs(
        SDPBackend.FLASH_ATTENTION, torch.device("cpu")
    )

    assert ran
    (query,) = seen
    assert query.dtype == torch.bfloat16
    assert query.device == torch.device("cpu")
    assert query.shape == (1, 1, 8, 64)


def test_kernel_probe_reports_a_kernel_that_runs() -> None:
    from torch.nn.attention import SDPBackend

    assert adapter_module._kernel_runs(SDPBackend.MATH, torch.device("cpu"))


def test_kernel_probe_reports_a_kernel_that_raises_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.nn.attention import SDPBackend

    def attention(*args: Any) -> torch.Tensor:
        raise RuntimeError("No available kernel")

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", attention
    )

    assert not adapter_module._kernel_runs(
        SDPBackend.FLASH_ATTENTION, torch.device("cpu")
    )


def test_kernel_probe_hides_the_warnings_of_a_kernel_that_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.nn.attention import SDPBackend

    def attention(*args: Any) -> torch.Tensor:
        warnings.warn("kernel not used", UserWarning, stacklevel=2)
        raise RuntimeError("No available kernel")

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", attention
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter_module._kernel_runs(
            SDPBackend.FLASH_ATTENTION, torch.device("cpu")
        )

    assert caught == []


GIB = 1 << 30


@pytest.fixture
def cuda_memory(
    monkeypatch: pytest.MonkeyPatch, attention_probe: AttentionProbe
) -> list[tuple[float, object]]:
    """Fakes 6 GiB free and 1 GiB reserved of 8 GiB, recording fractions.

    The probe is replaced too, so no test touches a real CUDA device. Like
    torch, the fakes reject a device without an index.
    """
    fractions: list[tuple[float, object]] = []

    def mem_get_info(device: torch.device) -> tuple[int, int]:
        assert device.index is not None
        return 6 * GIB, 8 * GIB

    def memory_reserved(device: torch.device) -> int:
        assert device.index is not None
        return GIB

    def set_fraction(fraction: float, device: torch.device) -> None:
        assert device.index is not None
        FakePi3X.events.append("cap")
        fractions.append((fraction, device))

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "memory_reserved", memory_reserved)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", set_fraction
    )
    return fractions


CUDA_0 = torch.device("cuda", 0)


def test_load_caps_the_allocator_at_free_plus_reserved_vram_before_loading(
    fake_pi3x: type[FakePi3X], cuda_memory: list[tuple[float, object]]
) -> None:
    Pi3XReconstructor(Pi3XConfig(device="cuda")).load()

    assert cuda_memory == [(0.875, CUDA_0)]
    assert fake_pi3x.events == ["cap", "from_pretrained", "to"]


def test_unload_lifts_the_allocator_cap(
    fake_pi3x: type[FakePi3X],
    cuda_memory: list[tuple[float, object]],
    empty_cache_calls: list[int],
) -> None:
    adapter = Pi3XReconstructor(Pi3XConfig(device="cuda"))
    adapter.load()

    adapter.unload()

    assert cuda_memory == [(0.875, CUDA_0), (1.0, CUDA_0)]


def test_unload_leaves_the_allocator_alone_when_capping_is_disabled(
    fake_pi3x: type[FakePi3X],
    cuda_memory: list[tuple[float, object]],
    empty_cache_calls: list[int],
) -> None:
    config = Pi3XConfig(device="cuda", cap_allocator_at_free_vram=False)
    adapter = Pi3XReconstructor(config)
    adapter.load()

    adapter.unload()

    assert cuda_memory == []


def test_a_later_load_caps_the_allocator_again(
    fake_pi3x: type[FakePi3X],
    cuda_memory: list[tuple[float, object]],
    empty_cache_calls: list[int],
) -> None:
    adapter = Pi3XReconstructor(Pi3XConfig(device="cuda"))
    adapter.load()
    adapter.unload()

    adapter.load()

    assert [fraction for fraction, _ in cuda_memory] == [0.875, 1.0, 0.875]


def _fail(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("out of memory")


@pytest.mark.parametrize(
    ("target", "attribute"),
    [
        pytest.param(adapter_module, "_kernel_runs", id="probe"),
        pytest.param(FakePi3X, "from_pretrained", id="from_pretrained"),
        pytest.param(FakePi3X, "to", id="to"),
    ],
)
def test_a_failed_load_lifts_the_allocator_cap_and_reraises(
    fake_pi3x: type[FakePi3X],
    cuda_memory: list[tuple[float, object]],
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    attribute: str,
) -> None:
    monkeypatch.setattr(target, attribute, _fail)
    adapter = Pi3XReconstructor(Pi3XConfig(device="cuda"))

    with pytest.raises(RuntimeError, match="out of memory"):
        adapter.load()

    assert cuda_memory == [(0.875, CUDA_0), (1.0, CUDA_0)]


def test_a_failed_load_without_the_cap_leaves_the_allocator_alone(
    fake_pi3x: type[FakePi3X],
    cuda_memory: list[tuple[float, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FakePi3X, "from_pretrained", _fail)
    config = Pi3XConfig(device="cuda", cap_allocator_at_free_vram=False)

    with pytest.raises(RuntimeError, match="out of memory"):
        Pi3XReconstructor(config).load()

    assert cuda_memory == []


@pytest.mark.parametrize(
    "config",
    [
        Pi3XConfig(device="cuda", cap_allocator_at_free_vram=False),
        Pi3XConfig(device="cpu"),
    ],
    ids=["capping-disabled", "cpu-device"],
)
def test_load_does_not_cap_the_allocator_when_there_is_nothing_to_cap(
    fake_pi3x: type[FakePi3X],
    cuda_memory: list[tuple[float, object]],
    config: Pi3XConfig,
) -> None:
    Pi3XReconstructor(config).load()

    assert cuda_memory == []
