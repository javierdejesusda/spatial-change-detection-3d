import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.perception.fake import FakeFeatureEncoder
from spatial_change.ports import FeatureEncoder


def _image(height: int, width: int, rgb: tuple[int, int, int]) -> NDArray:
    return np.full((height, width, 3), rgb, np.uint8)


def test_fake_encoder_satisfies_the_port() -> None:
    assert isinstance(FakeFeatureEncoder(), FeatureEncoder)


def test_default_patch_size_is_14() -> None:
    assert FakeFeatureEncoder().patch_size == 14


@pytest.mark.parametrize("patch_size", [0, -1])
def test_invalid_patch_sizes_are_rejected(patch_size: int) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        FakeFeatureEncoder(patch_size)


@pytest.mark.parametrize(
    ("shape", "patch", "grid"),
    [
        ((294, 518), 16, (19, 33)),
        ((294, 518), 14, (21, 37)),
        ((8, 8), 4, (2, 2)),
    ],
)
def test_grid_is_the_padded_image_over_the_patch_size(
    shape: tuple[int, int], patch: int, grid: tuple[int, int]
) -> None:
    features = FakeFeatureEncoder(patch).encode(_image(*shape, (9, 9, 9)))

    assert features.shape == (*grid, 3)
    assert features.dtype == np.float32


def test_feature_is_the_unit_direction_of_the_shifted_mean_color() -> None:
    image = np.zeros((4, 8, 3), np.uint8)
    image[:, :4] = (0, 0, 0)
    image[:, 4:] = (254, 0, 0)

    features = FakeFeatureEncoder(4).encode(image)

    np.testing.assert_allclose(np.linalg.norm(features, axis=2), 1.0, atol=1e-6)
    expected_black = np.ones(3) / np.sqrt(3)
    np.testing.assert_allclose(features[0, 0], expected_black, atol=1e-6)
    red = np.array([255.0, 1.0, 1.0])
    np.testing.assert_allclose(
        features[0, 1], red / np.linalg.norm(red), atol=1e-6
    )


def test_patch_feature_averages_the_colors_of_its_pixels() -> None:
    image = np.zeros((2, 2, 3), np.uint8)
    image[0] = (100, 0, 0)
    image[1] = (0, 100, 0)

    feature = FakeFeatureEncoder(2).encode(image)[0, 0]

    mean = np.array([51.0, 51.0, 1.0])
    np.testing.assert_allclose(feature, mean / np.linalg.norm(mean), atol=1e-6)


def test_edge_patches_are_padded_by_edge_replication() -> None:
    image = np.zeros((3, 3, 3), np.uint8)
    image[:, :] = (200, 10, 10)

    features = FakeFeatureEncoder(2).encode(image)

    np.testing.assert_allclose(features[1, 1], features[0, 0], atol=1e-6)


def test_cosine_ignores_brightness_but_not_hue() -> None:
    encoder = FakeFeatureEncoder(4)
    red = encoder.encode(_image(4, 4, (200, 40, 40)))[0, 0]
    dark_red = encoder.encode(_image(4, 4, (99, 19, 19)))[0, 0]
    blue = encoder.encode(_image(4, 4, (40, 40, 200)))[0, 0]

    assert float(red @ dark_red) > 0.999
    assert float(red @ blue) < 0.5


def test_calls_are_counted() -> None:
    encoder = FakeFeatureEncoder(4)

    encoder.encode(_image(4, 4, (1, 2, 3)))
    encoder.encode(_image(4, 4, (1, 2, 3)))
    encoder.unload()

    assert encoder.encode_count == 2
    assert encoder.unload_count == 1


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), np.uint8),
        np.zeros((4, 4, 4), np.uint8),
        np.zeros((4, 4, 3), np.float32),
        np.zeros((0, 4, 3), np.uint8),
    ],
)
def test_invalid_images_are_rejected(image: NDArray) -> None:
    with pytest.raises(ValueError, match="image"):
        FakeFeatureEncoder(4).encode(image)
