import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.geometry.walls import WallConfig, find_walls

VOXEL = 0.05


def _wall_grid(
    x0: float, x1: float, y: float = 0.0, z1: float = 2.2
) -> NDArray[np.float64]:
    xs, zs = np.meshgrid(np.arange(x0, x1, 0.05), np.arange(0.05, z1, 0.05))
    return np.stack([xs.ravel(), np.full(xs.size, y), zs.ravel()], axis=1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"plane_distance": 0.0},
        {"min_width": 0.0},
        {"min_height": -1.0},
        {"max_gap": 0.0},
        {"min_fill": 0.0},
        {"min_fill": 1.5},
        {"max_planes": -1},
        {"ransac_iterations": 0},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        WallConfig(**overrides)  # type: ignore[arg-type]


def test_a_wall_is_found_and_contains_its_points() -> None:
    points = _wall_grid(0.0, 3.0)

    (wall,) = find_walls(points, VOXEL, WallConfig())

    assert wall.contains(points, 0.03).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_wall_is_found_whatever_the_seed(seed: int) -> None:
    points = _wall_grid(0.0, 3.0)

    walls = find_walls(points, VOXEL, WallConfig(seed=seed))

    assert len(walls) == 1


def test_no_planes_are_removed_when_max_planes_is_zero() -> None:
    points = _wall_grid(0.0, 3.0)

    assert find_walls(points, VOXEL, WallConfig(max_planes=0)) == []


def test_furniture_in_line_with_a_wall_beyond_its_end_is_not_cut() -> None:
    wall = _wall_grid(0.0, 3.0)
    cabinet = _wall_grid(6.0, 7.2)[:400]
    points = np.concatenate([wall, cabinet])

    (found,) = find_walls(points, VOXEL, WallConfig())

    removed = found.contains(points, 0.03)
    assert removed[: len(wall)].all()
    assert not removed[len(wall) :].any()


def test_a_sparse_plane_through_separate_pieces_is_not_a_wall() -> None:
    pieces = np.concatenate(
        [
            _wall_grid(0.0, 0.5, z1=1.5),
            _wall_grid(0.7, 1.2, z1=1.5),
            _wall_grid(1.4, 1.9, z1=1.5),
        ]
    )

    config = dataclasses.replace(WallConfig(), max_gap=0.5, min_fill=0.9)

    assert find_walls(pieces, VOXEL, config) == []
