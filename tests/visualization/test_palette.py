import re

import pytest

from spatial_change.domain import ChangeType
from spatial_change.visualization.palette import (
    CHANGE_COLORS,
    UNOBSERVED_COLOR,
    change_color_hex,
)


def test_every_change_type_has_one_distinct_color() -> None:
    assert set(CHANGE_COLORS) == set(ChangeType)
    assert len(set(CHANGE_COLORS.values())) == len(ChangeType)


def test_colors_are_rgb_bytes() -> None:
    for color in CHANGE_COLORS.values():
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


@pytest.mark.parametrize("kind", list(ChangeType))
def test_hex_form_matches_the_rgb_color(kind: ChangeType) -> None:
    text = change_color_hex(kind)

    assert re.fullmatch(r"#[0-9a-f]{6}", text)
    red, green, blue = CHANGE_COLORS[kind]
    assert text == f"#{red:02x}{green:02x}{blue:02x}"


def test_the_palette_cannot_be_modified() -> None:
    with pytest.raises(TypeError):
        CHANGE_COLORS[ChangeType.ADDED] = (0, 0, 0)  # type: ignore[index]


def test_unobserved_gray_is_a_neutral_apart_from_every_type() -> None:
    assert UNOBSERVED_COLOR not in CHANGE_COLORS.values()
    assert max(UNOBSERVED_COLOR) - min(UNOBSERVED_COLOR) < 40
