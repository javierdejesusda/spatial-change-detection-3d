import time
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.perception import DinoFeatureEncoder
from spatial_change.synthetic.scene import SceneConfig, generate_scene_pair

pytestmark = pytest.mark.gpu

WIDTH, HEIGHT = 518, 294
# Measured with the DINOv3 ViT-S/16 weights on a rendered frame: a 0.8x
# brightness copy keeps a mean per-patch cosine of 0.995 and a minimum of
# 0.984.
MIN_MEAN_BRIGHTNESS_COSINE = 0.98


@pytest.fixture(scope="module")
def encoder() -> Iterator[DinoFeatureEncoder]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    dino = DinoFeatureEncoder()
    started = time.perf_counter()
    try:
        dino.load()
    except OSError as error:
        pytest.skip(f"no DINO model can be loaded: {error}")
    print(
        f"loaded {dino.loaded_model_id} in "
        f"{time.perf_counter() - started:.1f} s"
    )
    yield dino
    dino.unload()


@pytest.fixture(scope="module")
def frame_image() -> NDArray[np.uint8]:
    pair = generate_scene_pair(
        0, SceneConfig(width=WIDTH, height=HEIGHT, frames_per_epoch=4)
    )
    image = pair.before.frames[0].image
    assert image is not None
    return image


def _cosine(
    first: NDArray[np.float32], second: NDArray[np.float32]
) -> NDArray[np.float32]:
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.asarray((first * second).sum(axis=-1), dtype=np.float32)


def test_features_have_the_expected_grid_and_are_finite(
    encoder: DinoFeatureEncoder, frame_image: NDArray[np.uint8]
) -> None:
    patch = encoder.patch_size

    features = encoder.encode(frame_image)

    assert features.shape[:2] == (-(-HEIGHT // patch), -(-WIDTH // patch))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_the_same_image_gives_identical_features(
    encoder: DinoFeatureEncoder, frame_image: NDArray[np.uint8]
) -> None:
    first = encoder.encode(frame_image)
    started = time.perf_counter()
    second = encoder.encode(frame_image)
    print(f"encode {time.perf_counter() - started:.3f} s per frame")

    np.testing.assert_array_equal(first, second)


def test_a_dimmer_copy_keeps_the_features_close(
    encoder: DinoFeatureEncoder, frame_image: NDArray[np.uint8]
) -> None:
    dimmer = (frame_image.astype(np.float32) * 0.8).round().astype(np.uint8)

    cosine = _cosine(encoder.encode(frame_image), encoder.encode(dimmer))

    print(f"brightness cosine mean {cosine.mean():.4f} min {cosine.min():.4f}")
    assert cosine.mean() >= MIN_MEAN_BRIGHTNESS_COSINE
