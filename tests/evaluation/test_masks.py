import numpy as np
import pytest

from spatial_change.evaluation._masks import resize_nearest, splat

SIZE = 12


def box(top: int, left: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top : top + 4, left : left + 4] = True
    return mask


def test_resize_nearest_downsamples_like_a_floor_lookup() -> None:
    mask = np.arange(16).reshape(4, 4) % 2 == 0

    out = resize_nearest(mask, 2, 2)

    np.testing.assert_array_equal(out, mask[np.ix_([0, 2], [0, 2])])


def test_resize_nearest_upsamples_by_repetition() -> None:
    mask = np.array([[True, False], [False, True]])

    out = resize_nearest(mask, 4, 4)

    np.testing.assert_array_equal(out, np.kron(mask, np.ones((2, 2), bool)))


def test_resize_nearest_returns_the_same_size_unchanged() -> None:
    mask = box(0, 0)

    np.testing.assert_array_equal(resize_nearest(mask, SIZE, SIZE), mask)


def test_resize_nearest_matches_the_float32_scale_of_the_reference() -> None:
    # Integer arithmetic would read source row 7 for destination row 23 of
    # a 14 -> 46 resize; float32 scale arithmetic reads row 6.
    src = (np.arange(14) % 2 == 0).reshape(14, 1)

    out = resize_nearest(src, 46, 1)

    assert out[23, 0] == src[6, 0]
    assert src[6, 0] != src[7, 0]


def test_splat_draws_clipped_disks() -> None:
    rows = np.array([0, 5])
    cols = np.array([0, 5])

    out = splat(rows, cols, 8, 8, 2)

    grid_r, grid_c = np.mgrid[:8, :8]
    expected = ((grid_r - 0) ** 2 + (grid_c - 0) ** 2 <= 4) | (
        (grid_r - 5) ** 2 + (grid_c - 5) ** 2 <= 4
    )
    np.testing.assert_array_equal(out, expected)


def test_splat_with_radius_zero_marks_only_the_points() -> None:
    out = splat(np.array([1]), np.array([2]), 4, 4, 0)

    assert out.sum() == 1
    assert out[1, 2]


def test_splat_larger_than_the_image_fills_it() -> None:
    assert splat(np.array([1]), np.array([1]), 3, 3, 50).all()


def test_splat_without_points_is_empty() -> None:
    empty = np.array([], dtype=np.intp)

    assert not splat(empty, empty, 3, 3, 2).any()


@pytest.mark.parametrize("radius", [1, 3])
def test_splat_matches_the_disk_formula(radius: int) -> None:
    out = splat(np.array([4]), np.array([6]), 9, 9, radius)

    grid_r, grid_c = np.mgrid[:9, :9]
    np.testing.assert_array_equal(
        out, (grid_r - 4) ** 2 + (grid_c - 6) ** 2 <= radius**2
    )
