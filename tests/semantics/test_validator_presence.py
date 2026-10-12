import pytest
from facts import make_fact, moved_fact

from spatial_change.domain import ChangeType
from spatial_change.semantics.validator import validate_description


@pytest.mark.parametrize(
    "text",
    [
        "The crate is now on the floor, removed from the shelf.",
        "The crate now rests on the floor and was removed.",
        "The crate is now resting there and was removed.",
        "The crate now sits there but was removed.",
        "The crate now stands there, removed.",
        "The crate remains there, removed from the shelf.",
        "The crate was removed but still there.",
    ],
)
def test_a_removed_object_cannot_be_said_to_be_still_there(text: str) -> None:
    result = validate_description(
        text, make_fact(ChangeType.REMOVED, label="crate")
    )

    assert not result.ok
    assert any(reason.startswith("presence") for reason in result.reasons)


@pytest.mark.parametrize(
    "text",
    [
        "The crate was removed from the floor.",
        "The crate was removed and the floor is empty.",
        "The crate was removed and the shelf stays put.",
    ],
)
def test_a_removed_object_can_be_described_leaving(text: str) -> None:
    fact = make_fact(ChangeType.REMOVED, label="crate")

    assert validate_description(text, fact).ok


@pytest.mark.parametrize(
    "text",
    [
        "A drum was added and was previously elsewhere.",
        "A drum was added where one used to be.",
        "A drum was added and is no longer absent.",
        "A drum was added; it was already there.",
    ],
)
def test_an_added_object_cannot_be_said_to_be_there_before(text: str) -> None:
    result = validate_description(
        text, make_fact(ChangeType.ADDED, label="drum")
    )

    assert not result.ok
    assert any(reason.startswith("presence") for reason in result.reasons)


@pytest.mark.parametrize(
    "text",
    [
        "A drum was added and is now on the table.",
        "A drum was added and it remains there.",
        "A drum was added on the table, previously empty.",
    ],
)
def test_an_added_object_can_be_described_arriving(text: str) -> None:
    fact = make_fact(ChangeType.ADDED, label="drum")

    assert validate_description(text, fact).ok


def test_presence_phrases_do_not_constrain_a_moved_object() -> None:
    fact = moved_fact()

    assert validate_description(
        "The chair moved and is now on the left, still near the table.",
        fact,
    ).ok
