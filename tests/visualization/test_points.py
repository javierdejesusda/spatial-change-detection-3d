import numpy as np
import pytest

from spatial_change.visualization.points import ScenePoints, subsample


def _cloud(count: int) -> ScenePoints:
    points = np.arange(count * 3, dtype=np.float64).reshape(count, 3)
    colors = (np.arange(count * 3) % 256).astype(np.uint8).reshape(count, 3)
    return ScenePoints(points, colors)


def test_scene_points_reject_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="differ"):
        ScenePoints(np.zeros((3, 3)), np.zeros((2, 3), dtype=np.uint8))


def test_scene_points_reject_wrong_dtypes_and_shapes() -> None:
    with pytest.raises(ValueError, match="colors"):
        ScenePoints(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="points"):
        ScenePoints(np.zeros((2, 2)), np.zeros((2, 3), dtype=np.uint8))


def test_subsample_below_the_cap_keeps_every_point() -> None:
    cloud = _cloud(10)

    kept = subsample(cloud, 10)

    np.testing.assert_array_equal(kept.points, cloud.points)
    np.testing.assert_array_equal(kept.colors, cloud.colors)


def test_subsample_is_even_ordered_and_deterministic() -> None:
    cloud = _cloud(1000)

    first = subsample(cloud, 100)
    second = subsample(cloud, 100)

    assert len(first.points) == 100
    np.testing.assert_array_equal(first.points, second.points)
    rows = first.points[:, 0] / 3
    assert np.all(np.diff(rows) > 0)
    assert np.ptp(np.diff(rows)) <= 1.0
    index = (first.points[:, 0] / 3).astype(int)
    np.testing.assert_array_equal(first.colors, cloud.colors[index])


def test_subsample_drops_points_that_are_not_finite() -> None:
    points = np.array([[0, 0, 0], [np.nan, 0, 0], [1, 1, np.inf], [2, 2, 2]])
    cloud = ScenePoints(points, np.zeros((4, 3), dtype=np.uint8))

    kept = subsample(cloud, 10)

    np.testing.assert_array_equal(kept.points, [[0, 0, 0], [2, 2, 2]])


def test_subsample_of_nothing_is_empty() -> None:
    kept = subsample(ScenePoints.empty(), 5)

    assert len(kept.points) == 0
    assert kept.colors.shape == (0, 3)


def test_subsample_rejects_a_cap_below_one() -> None:
    with pytest.raises(ValueError, match="cap"):
        subsample(_cloud(3), 0)
