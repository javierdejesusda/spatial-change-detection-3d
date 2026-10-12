import dataclasses
import math

import pytest

from spatial_change.domain import ChangeType
from spatial_change.visualization import RerunConfig
from spatial_change.visualization.palette import CHANGE_COLORS


def test_every_change_type_has_its_own_color() -> None:
    colors = RerunConfig().change_colors

    assert set(colors) == set(ChangeType)
    assert len(set(colors.values())) == len(ChangeType)


def test_the_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        RerunConfig().seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"application_id": ""},
        {"point_radius": 0.0},
        {"point_radius": -0.1},
        {"point_radius": math.nan},
        {"point_radius": math.inf},
        {"line_radius": 0.0},
        {"line_radius": math.nan},
        {"max_points": 0},
        {"seed": -1},
        {"seed": 1.5},
        {"max_unobserved_points": -1},
        {"ghost_alpha": 0.0},
        {"ghost_alpha": 1.0},
        {"ghost_alpha": math.nan},
        {"unknown_alpha": 0.0},
        {"unknown_alpha": 1.0},
        {"image_downscale": 0},
        {"max_image_panels": -1},
        {"panel_points": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RerunConfig(**kwargs)  # type: ignore[arg-type]


def test_colors_must_cover_every_type() -> None:
    colors = dict(RerunConfig().change_colors)
    del colors[ChangeType.MOVED]

    with pytest.raises(ValueError, match="change_colors"):
        RerunConfig(change_colors=colors)


def test_the_config_is_hashable_and_its_colors_are_immutable() -> None:
    config = RerunConfig()
    colors = dict(config.change_colors)
    copied = RerunConfig(change_colors=colors)

    colors[ChangeType.ADDED] = (1, 2, 3)

    assert hash(config) == hash(RerunConfig())
    assert config == RerunConfig()
    assert RerunConfig(seed=1) != config
    assert copied.change_colors[ChangeType.ADDED] != (1, 2, 3)
    for mapping in (config.change_colors, CHANGE_COLORS):
        with pytest.raises(TypeError):
            mapping[ChangeType.ADDED] = (0, 0, 0)  # type: ignore[index]


@pytest.mark.parametrize(
    "bad",
    [
        (1.0, 2.0, 3.0),
        (True, False, True),
        (1, 2),
        (256, 0, 0),
        (-1, 0, 0),
        (0, 0, 0, 0),
        7,
        None,
        "abc",
        (1, 2, "3"),
    ],
)
def test_colors_must_be_three_int_bytes(bad: object) -> None:
    colors = dict(RerunConfig().change_colors)
    colors[ChangeType.ADDED] = bad  # type: ignore[assignment]

    with pytest.raises(ValueError, match="change_colors"):
        RerunConfig(change_colors=colors)


def test_default_box_colors_are_the_shared_palette() -> None:
    assert dict(RerunConfig().change_colors) == dict(CHANGE_COLORS)
