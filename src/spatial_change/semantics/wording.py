"""English morphology shared by the sentence builders and the readers.

One place owns articles, plurals, lists, and the number words, so the
template, the summary, the prompts, and the number reader agree.
"""

from collections.abc import Sequence

VOWELS = "aeiou"

_SMALL = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]

NUMBER_WORDS: tuple[str, ...] = (
    *_SMALL,
    *(
        tens if unit == 0 else f"{tens}-{_SMALL[unit]}"
        for tens in _TENS
        for unit in range(10)
    ),
)
"""The words for 0 to 99, such as ``"seven"`` and ``"twenty-five"``."""

_MAX_WORD_COUNT = 10
_IRREGULAR_PLURALS = {
    "shelf": "shelves",
    "knife": "knives",
    "leaf": "leaves",
    "wolf": "wolves",
    "person": "people",
    "child": "children",
}
_ES_ENDINGS = ("s", "x", "z", "ch", "sh")


def with_article(label: str) -> str:
    """Prefixes a label with "a" or "an" by its first letter.

    Args:
        label: A noun phrase.

    Returns:
        The label with its article.
    """
    article = "an" if label.lower().startswith(tuple(VOWELS)) else "a"
    return f"{article} {label}"


def join_and(items: Sequence[str]) -> str:
    """Joins items as an English list.

    Args:
        items: The items.

    Returns:
        ``"a"``, ``"a and b"``, or ``"a, b, and c"``; empty for no items.
    """
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def pluralize(label: str) -> str:
    """Pluralizes a noun phrase.

    Args:
        label: A singular noun phrase such as ``"coffee table"``.

    Returns:
        The plural: an irregular form matched by suffix (shelf, knife,
        leaf, wolf, person, child, so also bookshelf), ``-es`` after s, x,
        z, ch, or sh, ``-ies`` for y after a consonant, otherwise ``-s``.
    """
    lowered = label.lower()
    for singular, plural in _IRREGULAR_PLURALS.items():
        if lowered.endswith(singular):
            stem = len(label) - len(singular)
            return label[:stem] + label[stem : stem + 1] + plural[1:]
    if lowered.endswith(_ES_ENDINGS):
        return label + "es"
    if lowered.endswith("y") and lowered[-2:-1] not in VOWELS:
        return label[:-1] + "ies"
    return label + "s"


def count_phrase(count: int, noun: str, indefinite: bool = True) -> str:
    """Words a count of a noun.

    Args:
        count: How many, at least 1.
        noun: The singular noun phrase.
        indefinite: Whether a single noun gets "a" or "an".

    Returns:
        ``"a chair"`` (or ``"chair"`` without the article) for one,
        ``"three chairs"`` up to ten, and ``"11 chairs"`` above.

    Raises:
        ValueError: If ``count`` is below 1.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if count == 1:
        return with_article(noun) if indefinite else noun
    number = NUMBER_WORDS[count] if count <= _MAX_WORD_COUNT else str(count)
    return f"{number} {pluralize(noun)}"
