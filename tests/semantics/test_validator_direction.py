import pytest
from facts import moved_fact

from spatial_change.semantics.validator import validate_description


def _ok(text: str, direction: str) -> bool:
    fact = moved_fact(
        direction=direction,
        toward=None,
        displacement_m=None,
        displacement_text=None,
        rotation_deg=None,
        rotation_text=None,
    )
    return validate_description(text, fact).ok


@pytest.mark.parametrize(
    ("direction", "text"),
    [
        ("to the left", "The chair moved right next to the table."),
        ("to the left", "The chair moved right where the sofa was."),
        ("to the left", "The chair moved right up against the wall."),
        ("to the right", "The chair moved and ended up by the window."),
        ("to the right", "The chair moved and was left near the door."),
        ("to the right", "The chair moved, leaving a stool left behind."),
        ("down", "The chair was moved and set up by the window."),
        ("up", "The chair was moved and put down by the window."),
        ("up", "The chair moved to the lower shelf."),
        ("down", "The chair moved to the higher shelf."),
        ("to the right", "The chair moved to the room on the left."),
        ("farther", "The chair moved closer to the table."),
        ("closer", "The chair moved farther from the table."),
        ("closer", "The chair moved farther to the wall."),
    ],
)
def test_bare_words_that_are_not_directions_do_not_contradict(
    direction: str, text: str
) -> None:
    assert _ok(text, direction)


@pytest.mark.parametrize(
    ("direction", "text"),
    [
        ("to the right", "The chair moved to the left."),
        ("to the left", "The chair moved to the right."),
        ("to the right", "The chair slid leftward."),
        ("to the left", "The chair slid rightwards."),
        ("to the right", "The chair moved left."),
        ("to the left", "The chair was pushed right."),
        ("down", "The chair moved upward."),
        ("down", "The chair moved up."),
        ("down", "The chair was lifted."),
        ("down", "The chair was raised."),
        ("up", "The chair moved downward."),
        ("up", "The chair moved down."),
        ("up", "The chair was lowered."),
        ("farther", "The chair moved closer."),
        ("farther", "The chair moved nearer."),
        ("farther", "The chair moved closer to the camera."),
        ("farther", "The chair moved toward the viewpoint."),
        ("closer", "The chair moved farther."),
        ("closer", "The chair moved further back."),
        ("closer", "The chair moved away from the camera."),
        ("closer", "The chair moved farther from the viewer."),
    ],
)
def test_motion_phrases_in_the_opposite_direction_are_rejected(
    direction: str, text: str
) -> None:
    assert not _ok(text, direction)


@pytest.mark.parametrize(
    ("direction", "text"),
    [
        ("to the left", "The chair moved to the left."),
        ("to the left", "The chair slid leftward."),
        ("to the left", "The chair moved left."),
        ("to the right", "The chair moved rightward."),
        ("up", "The chair moved upward."),
        ("up", "The chair moved up."),
        ("down", "The chair moved downward."),
        ("closer", "The chair moved toward the camera."),
        ("farther", "The chair moved away from the camera."),
        ("closer and to the left", "The chair moved closer and left."),
    ],
)
def test_motion_phrases_in_the_same_direction_are_accepted(
    direction: str, text: str
) -> None:
    assert _ok(text, direction)


@pytest.mark.parametrize(
    ("direction", "text", "expected"),
    [
        (
            "closer and to the right",
            "It moved to the left of the table.",
            False,
        ),
        ("closer and to the left", "The chair moved farther away.", False),
        ("closer and to the left", "The chair moved to the right.", False),
        ("to the left", "The chair moved left and closer.", True),
        (None, "The chair moved to the right.", True),
    ],
)
def test_composite_and_missing_directions(
    direction: str | None, text: str, expected: bool
) -> None:
    fact = moved_fact(
        direction=direction,
        toward=None,
        displacement_m=None,
        displacement_text=None,
        rotation_deg=None,
        rotation_text=None,
    )

    assert validate_description(text, fact).ok is expected


def test_saying_both_sides_of_an_axis_is_not_a_contradiction() -> None:
    fact = moved_fact(direction="to the left", toward=None)

    assert validate_description(
        "The chair moved from the right side and ended up to the left.", fact
    ).ok
    assert validate_description(
        "The chair moved to the left, not to the right.", fact
    ).ok


def test_moving_away_from_the_target_landmark_is_rejected() -> None:
    fact = moved_fact(direction="to the left", toward="table")

    result = validate_description("The chair moved away from the table.", fact)

    assert not result.ok
    assert any(reason.startswith("toward") for reason in result.reasons)


def test_moving_toward_the_target_landmark_is_accepted() -> None:
    fact = moved_fact(direction="to the left", toward="table")

    assert validate_description("The chair moved toward the table.", fact).ok
