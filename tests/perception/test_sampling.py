import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.perception.sampling import sample_features

PATCH = 4


def _grid() -> NDArray[np.float32]:
    """A 3x4 grid of 2-channel features with distinct values."""
    rows = np.arange(3, dtype=np.float32)[:, None, None]
    cols = np.arange(4, dtype=np.float32)[None, :, None]
    return np.concatenate(
        [np.broadcast_to(rows * 10 + cols, (3, 4, 1)), rows - cols * 2],
        axis=2,
    ).astype(np.float32)


def _center(i: int, j: int) -> tuple[float, float]:
    return (j + 0.5) * PATCH - 0.5, (i + 0.5) * PATCH - 0.5


def test_sampling_at_patch_centers_returns_the_patch_feature() -> None:
    grid = _grid()
    uv = np.array([_center(i, j) for i in range(3) for j in range(4)])

    sampled = sample_features(grid, uv, PATCH)

    assert sampled.shape == (12, 2)
    assert sampled.dtype == np.float32
    np.testing.assert_allclose(sampled, grid.reshape(12, 2), atol=1e-5)


def test_sampling_midway_between_centers_averages_the_neighbours() -> None:
    grid = _grid()
    x0, y0 = _center(1, 1)
    uv = np.array([[x0 + PATCH / 2, y0], [x0, y0 + PATCH / 2]])

    sampled = sample_features(grid, uv, PATCH)

    np.testing.assert_allclose(
        sampled[0], (grid[1, 1] + grid[1, 2]) / 2, atol=1e-5
    )
    np.testing.assert_allclose(
        sampled[1], (grid[1, 1] + grid[2, 1]) / 2, atol=1e-5
    )


def test_sampling_matches_a_hand_computed_bilinear_value() -> None:
    grid = _grid()
    x0, y0 = _center(0, 1)
    uv = np.array([[x0 + 0.25 * PATCH, y0 + 0.75 * PATCH]])

    sampled = sample_features(grid, uv, PATCH)

    # Weights: x 0.75 / 0.25 along columns 1 and 2, y 0.25 / 0.75 along
    # rows 0 and 1.
    expected = (
        0.75 * 0.25 * grid[0, 1]
        + 0.25 * 0.25 * grid[0, 2]
        + 0.75 * 0.75 * grid[1, 1]
        + 0.25 * 0.75 * grid[1, 2]
    )
    np.testing.assert_allclose(sampled[0], expected, atol=1e-5)


def test_positions_beyond_the_outer_centers_clamp_to_the_border() -> None:
    grid = _grid()
    uv = np.array([[-0.5, -0.5], [100.0, 100.0], [-0.5, 100.0]])

    sampled = sample_features(grid, uv, PATCH)

    np.testing.assert_allclose(sampled[0], grid[0, 0], atol=1e-5)
    np.testing.assert_allclose(sampled[1], grid[2, 3], atol=1e-5)
    np.testing.assert_allclose(sampled[2], grid[2, 0], atol=1e-5)


def test_sampling_no_positions_returns_an_empty_array() -> None:
    sampled = sample_features(_grid(), np.zeros((0, 2)), PATCH)

    assert sampled.shape == (0, 2)
    assert sampled.dtype == np.float32


@pytest.mark.parametrize(
    ("features", "uv", "patch_size"),
    [
        (np.zeros((3, 4), np.float32), np.zeros((1, 2)), 4),
        (np.zeros((0, 4, 2), np.float32), np.zeros((1, 2)), 4),
        (np.zeros((3, 4, 2), np.float32), np.zeros((1, 3)), 4),
        (np.zeros((3, 4, 2), np.float32), np.zeros((1, 2)), 0),
    ],
)
def test_invalid_inputs_are_rejected(
    features: NDArray[np.float32], uv: NDArray[np.float64], patch_size: int
) -> None:
    with pytest.raises(ValueError):
        sample_features(features, uv, patch_size)
