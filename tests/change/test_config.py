import math

import pytest

from spatial_change.change import CandidateConfig


def test_defaults() -> None:
    config = CandidateConfig()

    assert config.voxel_size == 0.05
    assert config.link_radius == 2
    assert config.min_points == 10
    assert config.min_voxels == 3
    assert config.min_changed_fraction == 0.3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_size": 0.0},
        {"voxel_size": -0.05},
        {"voxel_size": math.nan},
        {"voxel_size": math.inf},
        {"link_radius": 0},
        {"link_radius": -1},
        {"min_points": 0},
        {"min_voxels": 0},
        {"min_changed_fraction": -0.1},
        {"min_changed_fraction": 1.1},
        {"min_changed_fraction": math.nan},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be"):
        CandidateConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"link_radius": 1},
        {"min_points": 1},
        {"min_voxels": 1},
        {"min_changed_fraction": 0.0},
        {"min_changed_fraction": 1.0},
    ],
)
def test_boundary_config_is_accepted(kwargs: dict[str, object]) -> None:
    CandidateConfig(**kwargs)  # type: ignore[arg-type]
