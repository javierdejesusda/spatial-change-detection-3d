import pytest

from spatial_change.semantics.quantities import angles_deg, distances_m

DISTANCES = [
    ("The chair moved 300 mm.", [0.3]),
    ("The chair moved 1 foot.", [0.3048]),
    ("The chair moved 3 feet.", [0.9144]),
    ("The chair moved 3 ft.", [0.9144]),
    ("The chair moved 12 inches.", [0.3048]),
    ("The chair moved 12 in.", [0.3048]),
    ("The chair moved 1 inch.", [0.0254]),
    ("The chair moved 1.25 m.", [1.25]),
    ("The chair moved 30 cm and then 2 m.", [0.3, 2.0]),
    ("The chair moved thirty centimeters.", [0.3]),
    ("The chair moved three meters.", [3.0]),
    ("The chair moved twenty-five cm.", [0.25]),
    ("The chair moved twenty five centimetres.", [0.25]),
    ("The chair moved ninety-nine cm.", [0.99]),
    ("The chair moved a half meter.", [0.5]),
    ("The chair moved half a meter.", [0.5]),
    ("The chair moved one half meter.", [0.5]),
    ("The chair moved half a metre.", [0.5]),
    ("The chair moved one and a half meters.", [1.5]),
    ("The chair moved One and a half meters.", [1.5]),
    ("The chair moved two and a half metres.", [2.5]),
    ("The chair moved 30-40 cm.", [0.3, 0.4]),
    ("The chair moved 30 to 40 cm.", [0.3, 0.4]),
    ("The chair moved 0.3-0.4 m.", [0.3, 0.4]),
    ("The chair moved 30-90 cm.", [0.3, 0.9]),
    ("The chair moved .5 m.", [0.5]),
    ("The chair moved a 30-centimeter step.", [0.3]),
]


@pytest.mark.parametrize(("text", "expected"), DISTANCES)
def test_distances_are_read_in_meters(text: str, expected: list[float]) -> None:
    assert distances_m(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "The chair moved.",
        "One chair moved.",
        "The chair moved, one in the corner.",
        "The chair moved 2 in front of the table.",
        "The chair moved 3 in a row.",
        "The chair moved in a small way.",
        "The chair moved once.",
        "The chair turned 70 degrees.",
    ],
)
def test_words_that_are_not_distances_are_not_read(text: str) -> None:
    assert distances_m(text) == []


def test_a_leading_dot_is_a_decimal_not_a_whole_number() -> None:
    assert distances_m("moved .5 m") == [0.5]


ANGLES = [
    ("The chair turned 70°.", [70.0]),
    ("The chair moved with a 90-degree turn.", [90.0]),
    ("The chair turned ninety degrees.", [90.0]),
    ("The chair turned fifteen degrees.", [15.0]),
    ("The chair turned 90 deg.", [90.0]),
    ("The chair turned 30-40 degrees.", [30.0, 40.0]),
    ("The chair turned 30 to 40 degrees.", [30.0, 40.0]),
    ("The chair made a quarter turn.", [90.0]),
    ("The chair made a quarter-turn.", [90.0]),
    ("The chair made a quarter of a turn.", [90.0]),
    ("The chair turned half a turn.", [180.0]),
    ("The chair made a half turn.", [180.0]),
    ("The chair made a three-quarter turn.", [270.0]),
    ("The chair made a three quarter turn.", [270.0]),
    ("The chair made a three-quarters turn.", [270.0]),
]


@pytest.mark.parametrize(("text", "expected"), ANGLES)
def test_angles_are_read_in_degrees(text: str, expected: list[float]) -> None:
    assert angles_deg(text) == expected


def test_a_three_quarter_turn_is_not_also_a_quarter_turn() -> None:
    assert angles_deg("a three-quarter turn") == [270.0]


@pytest.mark.parametrize(
    "text",
    ["The chair turned.", "The chair moved 70 cm.", "A 30 cm turn."],
)
def test_text_without_an_angle_gives_none(text: str) -> None:
    assert angles_deg(text) == []


def test_distances_and_angles_do_not_mix() -> None:
    text = "The chair moved 30 cm and turned 70 degrees."

    assert distances_m(text) == pytest.approx([0.3])
    assert angles_deg(text) == [70.0]
