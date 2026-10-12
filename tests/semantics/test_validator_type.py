import pytest
from facts import make_fact, moved_fact, near

from spatial_change.domain import ChangeType
from spatial_change.semantics.relations import ChangeFact
from spatial_change.semantics.validator import (
    _TURN_VERBS,
    _TYPE_WORDS,
    validate_description,
)

CHECKED = [ChangeType.ADDED, ChangeType.REMOVED, ChangeType.MOVED]


def _contradicting_words(change_type: ChangeType) -> list[str]:
    words = [
        verb
        for other, type_words in _TYPE_WORDS.items()
        if other is not change_type
        for verb in type_words.verbs
    ]
    return words if change_type is ChangeType.MOVED else [*words, *_TURN_VERBS]


def _own_words(change_type: ChangeType) -> list[str]:
    type_words = _TYPE_WORDS[change_type]
    return [*type_words.verbs, *type_words.own_extra, *type_words.phrases]


def _type_reasons(text: str, fact: ChangeFact) -> list[str]:
    result = validate_description(text, fact)
    return [r for r in result.reasons if r.startswith("type")]


def _pure_rotation() -> ChangeFact:
    return moved_fact(
        direction=None,
        toward=None,
        displacement_m=0.01,
        displacement_text=None,
    )


@pytest.mark.parametrize(
    ("change_type", "word"),
    [(t, w) for t in CHECKED for w in _contradicting_words(t)],
)
def test_every_word_of_another_type_contradicts(
    change_type: ChangeType, word: str
) -> None:
    fact = make_fact(change_type, label="x")

    reasons = _type_reasons(f"The thing {word} near it.", fact)

    assert any(reason.startswith("type contradicts") for reason in reasons)


@pytest.mark.parametrize(
    ("change_type", "word"),
    [(t, w) for t in CHECKED for w in _own_words(t)],
)
def test_every_word_of_the_own_type_names_the_change(
    change_type: ChangeType, word: str
) -> None:
    fact = make_fact(change_type, label="x")

    assert validate_description(f"The thing {word} near it.", fact).ok


@pytest.mark.parametrize(
    ("change_type", "text"),
    [
        (ChangeType.MOVED, "A chair was added to the room."),
        (ChangeType.MOVED, "The chair disappeared from view."),
        (ChangeType.ADDED, "The chair moved to the left."),
        (ChangeType.ADDED, "An added chair was pushed in."),
        (ChangeType.ADDED, "A drum was dragged in."),
        (ChangeType.REMOVED, "A chair appeared by the wall."),
        (ChangeType.REMOVED, "The crate was repositioned."),
    ],
)
def test_a_sentence_naming_another_type_is_rejected(
    change_type: ChangeType, text: str
) -> None:
    assert _type_reasons(text, make_fact(change_type, label="chair"))


@pytest.mark.parametrize(
    ("change_type", "text"),
    [
        (ChangeType.MOVED, "The chair is by the table."),
        (ChangeType.ADDED, "A drum is on the table."),
        (ChangeType.REMOVED, "A crate stands near the shelf."),
        (
            ChangeType.REMOVED,
            "A brown crate is now resting on the floor and was previously "
            "located near a shelf.",
        ),
    ],
)
def test_a_sentence_must_name_its_own_change(
    change_type: ChangeType, text: str
) -> None:
    reasons = _type_reasons(text, make_fact(change_type, label="x"))

    assert any(reason.startswith("type not stated") for reason in reasons)


def test_an_uncertain_change_allows_any_type_word() -> None:
    fact = make_fact(ChangeType.UNCERTAIN, label="chair")

    assert validate_description("The chair was added or removed.", fact).ok
    assert validate_description("Something may differ.", fact).ok


@pytest.mark.parametrize("word", _TURN_VERBS)
def test_a_turn_names_the_change_only_for_a_pure_rotation(word: str) -> None:
    text = f"The chair {word} 70 degrees."

    assert validate_description(text, _pure_rotation()).ok
    assert _type_reasons(text, moved_fact())


@pytest.mark.parametrize(
    "text",
    [
        "The gadded drum stands there.",
        "A putty drum stands there.",
        "A drum is addingly there.",
    ],
)
def test_words_match_whole_words_only(text: str) -> None:
    assert _type_reasons(text, make_fact(ChangeType.ADDED, label="x"))


@pytest.mark.parametrize(
    "text",
    [
        "The chair was moved and placed near the table.",
        "The chair was pushed and put by the window.",
        "The chair was moved to a new spot.",
    ],
)
def test_ambiguous_synonyms_do_not_contradict_a_moved_object(
    text: str,
) -> None:
    assert validate_description(text, moved_fact(direction=None)).ok


def test_a_label_is_not_read_as_a_change_word() -> None:
    fact = make_fact(
        ChangeType.ADDED,
        label="chair",
        support_after="floor",
        near_after=[near("sliding door", "next to", 0.2)],
    )

    assert validate_description(
        "A chair was added next to the sliding door.", fact
    ).ok


@pytest.mark.parametrize(
    ("change_type", "label", "text"),
    [
        (ChangeType.ADDED, "moving box", "A moving box was added."),
        (ChangeType.REMOVED, "turned table", "The turned table was removed."),
        (
            ChangeType.ADDED,
            "missing person poster",
            "A missing person poster was added.",
        ),
    ],
)
def test_the_own_label_is_not_read_as_a_change_word(
    change_type: ChangeType, label: str, text: str
) -> None:
    assert validate_description(text, make_fact(change_type, label=label)).ok


def test_a_change_word_outside_the_labels_still_counts() -> None:
    fact = make_fact(
        ChangeType.ADDED, label="box", near_after=[near("sliding door")]
    )

    result = validate_description(
        "A box was added and the door was removed.", fact
    )

    assert not result.ok
