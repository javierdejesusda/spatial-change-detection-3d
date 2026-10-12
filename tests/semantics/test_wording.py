import pytest

from spatial_change.semantics.wording import (
    NUMBER_WORDS,
    count_phrase,
    join_and,
    pluralize,
    with_article,
)


def test_the_number_table_covers_zero_to_ninety_nine() -> None:
    assert len(NUMBER_WORDS) == 100
    assert len(set(NUMBER_WORDS)) == 100
    assert NUMBER_WORDS[0] == "zero"
    assert NUMBER_WORDS[13] == "thirteen"
    assert NUMBER_WORDS[20] == "twenty"
    assert NUMBER_WORDS[21] == "twenty-one"
    assert NUMBER_WORDS[40] == "forty"
    assert NUMBER_WORDS[99] == "ninety-nine"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("chair", "a chair"),
        ("armchair", "an armchair"),
        ("TV", "a TV"),
        ("iPad", "an iPad"),
        ("", "a "),
    ],
)
def test_with_article(label: str, expected: str) -> None:
    assert with_article(label) == expected


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], ""),
        (["a"], "a"),
        (["a", "b"], "a and b"),
        (["a", "b", "c"], "a, b, and c"),
        (["a", "b", "c", "d"], "a, b, c, and d"),
    ],
)
def test_join_and(items: list[str], expected: str) -> None:
    assert join_and(items) == expected


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("chair", "chairs"),
        ("bus", "buses"),
        ("box", "boxes"),
        ("buzz", "buzzes"),
        ("church", "churches"),
        ("dish", "dishes"),
        ("lady", "ladies"),
        ("toy", "toys"),
        ("shelf", "shelves"),
        ("bookshelf", "bookshelves"),
        ("Bookshelf", "Bookshelves"),
        ("knife", "knives"),
        ("leaf", "leaves"),
        ("wolf", "wolves"),
        ("person", "people"),
        ("salesperson", "salespeople"),
        ("child", "children"),
        ("roof", "roofs"),
        ("coffee table", "coffee tables"),
        ("side shelf", "side shelves"),
        ("Box", "Boxes"),
    ],
)
def test_pluralize(singular: str, plural: str) -> None:
    assert pluralize(singular) == plural


@pytest.mark.parametrize(
    ("count", "noun", "expected"),
    [
        (1, "chair", "a chair"),
        (1, "armchair", "an armchair"),
        (2, "chair", "two chairs"),
        (10, "shelf", "ten shelves"),
        (11, "chair", "11 chairs"),
        (120, "person", "120 people"),
    ],
)
def test_count_phrase(count: int, noun: str, expected: str) -> None:
    assert count_phrase(count, noun) == expected


def test_count_phrase_can_leave_a_single_noun_without_an_article() -> None:
    assert count_phrase(1, "image", indefinite=False) == "image"
    assert count_phrase(3, "image", indefinite=False) == "three images"


@pytest.mark.parametrize("count", [0, -1])
def test_count_phrase_needs_a_positive_count(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        count_phrase(count, "chair")
