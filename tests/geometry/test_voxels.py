import numpy as np
import pytest

from spatial_change.geometry import label_voxel_cells, voxel_cells


def test_points_map_to_the_voxel_that_contains_them() -> None:
    points = np.array(
        [[0.0, 0.04, 0.05], [-0.01, 0.1, 0.149], [1.0, -0.05, -0.051]]
    )

    cells = voxel_cells(points, 0.05)

    assert cells.dtype == np.int64
    assert cells.tolist() == [[0, 0, 1], [-1, 2, 2], [20, -1, -2]]


def test_no_points_give_no_cells() -> None:
    cells = voxel_cells(np.empty((0, 3)), 0.05)

    assert cells.shape == (0, 3)
    assert cells.dtype == np.int64


def test_cells_label_by_component_within_the_radius() -> None:
    near = np.array([[0, 0, 0], [2, 2, 2], [3, 3, 3]], np.int64)
    far = np.array([[10, 0, 0], [-10, 5, 5]], np.int64)

    labels = label_voxel_cells(np.concatenate([near, far]), 2)

    assert len(set(labels[:3])) == 1
    assert len(set(labels)) == 3
    assert len(set(labels[3:])) == 2


def test_cells_at_a_distance_of_the_radius_plus_one_are_apart() -> None:
    cells = np.array([[0, 0, 0], [3, 0, 0]], np.int64)

    assert len(set(label_voxel_cells(cells, 2))) == 2
    assert len(set(label_voxel_cells(cells, 3))) == 1


def test_repeated_cells_share_a_label_and_keep_their_rows() -> None:
    cells = np.array([[5, 5, 5], [0, 0, 0], [5, 5, 5], [0, 0, 1]], np.int64)

    labels = label_voxel_cells(cells, 1)

    assert labels.shape == (4,)
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_no_cells_give_no_labels() -> None:
    labels = label_voxel_cells(np.empty((0, 3), np.int64), 2)

    assert labels.shape == (0,)


def test_cells_spanning_beyond_the_key_range_are_rejected() -> None:
    cells = np.array([[0, 0, 0], [10**9, 10**9, 10**9]], np.int64)

    with pytest.raises(ValueError, match="too many voxels"):
        label_voxel_cells(cells, 2)


def test_cells_beyond_float_precision_are_labelled_exactly() -> None:
    base = 2**60
    cells = np.array(
        [[base, 0, 0], [base + 1, 0, 0], [base + 5, 0, 0]], np.int64
    )

    labels = label_voxel_cells(cells, 1)

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
