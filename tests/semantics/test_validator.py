import pytest
from facts import make_fact, moved_fact, near

from spatial_change.domain import ChangeType
from spatial_change.semantics.template import template_sentence
from spatial_change.semantics.validator import (
    ValidationResult,
    repair_description,
    validate_description,
)


def _distance_ok(text: str, displacement_m: float) -> bool:
    fact = moved_fact(displacement_m=displacement_m, rotation_deg=None)
    return validate_description(text, fact).ok


def _rotation_ok(text: str, rotation_deg: float) -> bool:
    return validate_description(text, moved_fact(rotation_deg=rotation_deg)).ok


def test_a_faithful_sentence_is_ok() -> None:
    result = validate_description(
        "The chair was moved about 30 cm to the left and turned a bit.",
        moved_fact(),
    )

    assert result == ValidationResult(ok=True, reasons=())


def test_the_result_is_immutable() -> None:
    result = validate_description("The chair moved.", moved_fact())

    with pytest.raises(AttributeError):
        result.ok = False  # type: ignore[misc]


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_an_empty_sentence_is_rejected(text: str) -> None:
    result = validate_description(text, moved_fact())

    assert not result.ok
    assert result.reasons == ("empty description",)


def test_every_violation_is_reported() -> None:
    fact = moved_fact(direction="to the left", toward=None)

    result = validate_description(
        "The chair was added 2 m to the right and turned 170 degrees.", fact
    )

    kinds = {reason.split()[0] for reason in result.reasons}
    assert {"type", "direction", "distance", "rotation"} <= kinds


@pytest.mark.parametrize(
    ("text", "consistent", "contradicting"),
    [
        ("The chair moved 300 mm.", 0.3, 3.0),
        ("The chair moved 3 feet.", 0.9, 0.3),
        ("The chair moved 12 inches.", 0.3, 3.0),
        ("The chair moved thirty centimeters.", 0.3, 3.0),
        ("The chair moved one and a half meters.", 1.5, 0.3),
        ("The chair moved 30-40 cm.", 0.35, 2.0),
        ("The chair moved .5 m.", 0.5, 5.0),
    ],
)
def test_distances_in_several_forms_are_compared_with_the_fact(
    text: str, consistent: float, contradicting: float
) -> None:
    assert _distance_ok(text, consistent)
    assert not _distance_ok(text, contradicting)


def test_every_number_of_a_range_is_checked() -> None:
    assert not _distance_ok("The chair moved 30-90 cm.", 0.3)
    assert not _distance_ok("The chair moved 10 to 30 cm.", 0.5)


@pytest.mark.parametrize(
    "text",
    [
        "The chair moved about 30 cm.",
        "The chair moved 0.3 m to the left.",
        "The chair moved 35 centimeters.",
        "The chair moved 38 cm.",
        "The chair moved 22 cm.",
        "The chair moved 40 cm.",
    ],
)
def test_a_distance_inside_the_tolerance_is_accepted(text: str) -> None:
    assert _distance_ok(text, 0.3)


@pytest.mark.parametrize(
    "text",
    [
        "The chair moved about 1 m.",
        "The chair moved 2 meters.",
        "The chair moved 80 cm.",
        "The chair moved 5 cm.",
    ],
)
def test_a_distance_outside_the_tolerance_is_rejected(text: str) -> None:
    result = validate_description(text, moved_fact(displacement_m=0.3))

    assert not result.ok
    assert any(reason.startswith("distance") for reason in result.reasons)


def test_the_distance_tolerance_is_ten_centimeters_or_half() -> None:
    near_one_meter = moved_fact(displacement_m=1.0, displacement_text=None)

    assert validate_description("It moved 1.5 m.", near_one_meter).ok
    assert not validate_description("It moved 1.6 m.", near_one_meter).ok
    assert validate_description("It moved 50 cm.", near_one_meter).ok
    assert not validate_description("It moved 45 cm.", near_one_meter).ok
    tiny = moved_fact(displacement_m=0.05, displacement_text=None)
    assert validate_description("It moved 15 cm.", tiny).ok
    assert not validate_description("It moved 16 cm.", tiny).ok


def test_a_landmark_distance_is_a_valid_reference() -> None:
    fact = make_fact(
        ChangeType.ADDED,
        label="crate",
        near_after=[near("shelf", "near", 0.6)],
    )

    assert validate_description(
        "A crate was added about 60 cm from the shelf.", fact
    ).ok


def test_a_distance_without_any_reference_is_rejected() -> None:
    fact = make_fact(ChangeType.ADDED, label="crate")

    result = validate_description("A crate was added 2 m from the wall.", fact)

    assert not result.ok
    assert any(reason.startswith("distance") for reason in result.reasons)


@pytest.mark.parametrize(
    ("text", "consistent", "contradicting"),
    [
        ("The chair moved with a 90-degree turn.", 90.0, 20.0),
        ("The chair moved and turned ninety degrees.", 90.0, 20.0),
        ("The chair moved and made a quarter turn.", 90.0, 20.0),
        ("The chair moved and made a three-quarter turn.", 270.0, 90.0),
        ("The chair moved and turned half a turn.", 180.0, 90.0),
        ("The chair moved and turned 30-40 degrees.", 35.0, 120.0),
        ("The chair moved and turned 70°.", 70.0, 20.0),
    ],
)
def test_rotations_in_several_forms_are_compared_with_the_fact(
    text: str, consistent: float, contradicting: float
) -> None:
    assert _rotation_ok(text, consistent)
    assert not _rotation_ok(text, contradicting)


def test_the_rotation_tolerance_is_fifteen_degrees_or_half() -> None:
    small = moved_fact(rotation_deg=10.0, rotation_text=None)

    assert validate_description("It moved and turned 25 degrees.", small).ok
    assert not validate_description("It moved and turned 26 degrees.", small).ok
    large = moved_fact(rotation_deg=150.0)
    assert validate_description("It moved and turned 225 degrees.", large).ok
    assert not validate_description(
        "It moved and turned 226 degrees.", large
    ).ok


def test_a_rotation_without_a_reference_is_rejected() -> None:
    fact = make_fact(ChangeType.ADDED, label="crate")

    result = validate_description("A crate was added at 45 degrees.", fact)

    assert not result.ok
    assert any(reason.startswith("rotation") for reason in result.reasons)


def test_distances_and_rotations_are_not_mixed_up() -> None:
    fact = moved_fact(displacement_m=0.3, rotation_deg=70.0)

    assert validate_description(
        "The chair moved 30 cm and turned 70 degrees.", fact
    ).ok
    assert not validate_description(
        "The chair moved 70 cm and turned 30 degrees.", fact
    ).ok


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("3 m cable", "A 3 m cable was added."),
        ("90-degree pipe", "A 90-degree pipe was added."),
        ("2 in hinge", "A 2 in hinge was added."),
    ],
)
def test_a_quantity_inside_a_label_is_not_read_as_a_number(
    label: str, text: str
) -> None:
    fact = make_fact(ChangeType.ADDED, label=label)

    assert validate_description(text, fact).ok


def test_a_quantity_outside_the_label_is_still_checked() -> None:
    fact = make_fact(ChangeType.ADDED, label="3 m cable")

    result = validate_description("A 3 m cable was added 5 m away.", fact)

    assert not result.ok
    assert any(reason.startswith("distance") for reason in result.reasons)


def test_repair_keeps_a_valid_sentence() -> None:
    fact = moved_fact()
    text = "The chair slid to the left."

    assert repair_description(text, fact) == text


def test_repair_replaces_an_invalid_sentence_with_the_template() -> None:
    fact = moved_fact()

    assert repair_description("The chair moved to the right.", fact) == (
        template_sentence(fact)
    )
