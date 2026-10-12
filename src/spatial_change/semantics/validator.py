"""Rule-based check of a description against the geometry of its change.

A description may be worded freely, but it must not contradict the
``ChangeFact`` it describes. The rules are deliberately small and explicit;
a sentence that trips one is replaced by the deterministic template, so a
false alarm costs style and a missed contradiction costs correctness.

Rules:
    * type: a sentence must not use words of another change type (for
      example "removed" for an added object), and a sentence for a moved,
      added, or removed object must name its own change with one of a few
      synonyms. A turn names a move only for a pure rotation. An uncertain
      change allows all words.
    * presence: a removed object must not be said to be still there, and an
      added object must not be said to have been there before.
    * direction: on each axis the fact words (left or right, closer or
      farther, up or down), the sentence must not state only the opposite
      side. Only motion phrases count ("to the left", "moved right",
      "upward", "toward the camera"); bare words such as "right next to"
      or "ended up" do not. Saying both sides, or an axis the fact
      leaves out, is allowed.
    * toward: a sentence must not say the object moved away from the
      landmark it moved toward.
    * distance and rotation: every distance and angle in the sentence must
      lie within ``DISTANCE_TOLERANCE_M`` or ``RELATIVE_TOLERANCE`` of a
      reference: the displacement or a nearby landmark distance for
      distances, the rotation for angles. A number with no reference is
      rejected. Numbers are read by ``spatial_change.semantics.quantities``
      (digits or words, ranges, mm to feet, quarter and half turns); every
      number of a range is checked.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from spatial_change.domain import ChangeType
from spatial_change.semantics.quantities import angles_deg, distances_m
from spatial_change.semantics.relations import ChangeFact
from spatial_change.semantics.template import template_sentence

DISTANCE_TOLERANCE_M = 0.10
"""Absolute distance tolerance in meters."""

ROTATION_TOLERANCE_DEG = 15.0
"""Absolute rotation tolerance in degrees."""

RELATIVE_TOLERANCE = 0.5
"""Relative tolerance of distances and angles, as a fraction."""

_EPS = 1e-9
_LOWERCASE_WORD = re.compile(r"[a-z]+")


def _words(
    words: Iterable[str], phrases: Iterable[str] = ()
) -> re.Pattern[str]:
    """Compiles whole-word alternatives; spaces match any whitespace run.

    Args:
        words: Single words.
        phrases: Multi-word phrases.

    Returns:
        The compiled pattern.
    """
    escaped = [r"\s+".join(map(re.escape, p.split())) for p in phrases]
    escaped += [re.escape(word) for word in words]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


class _TypeWords(NamedTuple):
    """The words that state one kind of change.

    Attributes:
        verbs: Verbs that name the change for their own type and contradict
            every other type.
        own_extra: Words that name the change but never contradict another
            type: "placed", "put", and "new" also appear in sentences about
            moved objects ("moved and placed near the table").
        phrases: Multi-word phrases that name the change.
    """

    verbs: tuple[str, ...]
    own_extra: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()


_TYPE_WORDS: dict[ChangeType, _TypeWords] = {
    ChangeType.ADDED: _TypeWords(
        verbs=("added", "adds", "adding", "appeared", "appears", "appearing"),
        own_extra=(
            "new",
            "placed",
            "places",
            "placing",
            "put",
            "puts",
            "putting",
        ),
        phrases=("brought in", "brings in", "bringing in"),
    ),
    ChangeType.REMOVED: _TypeWords(
        verbs=(
            "removed",
            "removes",
            "removing",
            "disappeared",
            "disappears",
            "disappearing",
            "vanished",
            "vanishes",
            "vanishing",
            "missing",
            "gone",
        ),
        phrases=(
            "taken away",
            "takes away",
            "taking away",
            "took away",
            "no longer",
        ),
    ),
    ChangeType.MOVED: _TypeWords(
        verbs=(
            "moved",
            "moves",
            "moving",
            "shifted",
            "shifts",
            "shifting",
            "relocated",
            "relocates",
            "relocating",
            "repositioned",
            "repositions",
            "repositioning",
            "pushed",
            "pushes",
            "pushing",
            "slid",
            "slides",
            "sliding",
            "dragged",
            "drags",
            "dragging",
        ),
    ),
}
_TURN_VERBS = ("turned", "turns", "turning", "rotated", "rotates", "rotating")
"""Turning verbs: they contradict added and removed objects, and name a move
only when the fact is a pure rotation."""

_OWN = {
    change_type: _words(words.verbs + words.own_extra, words.phrases)
    for change_type, words in _TYPE_WORDS.items()
}
_CONTRADICTING = {
    change_type: frozenset(
        verb
        for other, words in _TYPE_WORDS.items()
        if other is not change_type
        for verb in words.verbs
    )
    | (frozenset() if change_type is ChangeType.MOVED else set(_TURN_VERBS))
    for change_type in _TYPE_WORDS
}
_TURN = _words(_TURN_VERBS)

_PRESENCE: dict[ChangeType, re.Pattern[str]] = {
    ChangeType.REMOVED: _words(
        ("remains", "still"),
        ("is now", "now rests", "now resting", "now sits", "now stands"),
    ),
    ChangeType.ADDED: _words(
        (), ("was previously", "used to be", "no longer", "was already")
    ),
}
"""Per type, phrases that contradict the change."""

_MOTION = (
    r"(?:"
    + "|".join((*_TYPE_WORDS[ChangeType.MOVED].verbs, "went", "goes", "going"))
    + r")"
)
_VIEWER = r"(?:camera|viewer|viewpoint)"
_NOT_PLACE = r"(?!\s+(?:next|where|up|against|at|by|beside|near|in|on)\b)"


class _Side(NamedTuple):
    """One side of an axis.

    Attributes:
        fact: Pattern of the word the fact uses for this side.
        text: Pattern of the motion phrases that state this side.
    """

    fact: re.Pattern[str]
    text: re.Pattern[str]


def _side(fact: str, *text: str) -> _Side:
    return _Side(re.compile(fact), re.compile("|".join(text)))


_AXES: dict[str, tuple[_Side, _Side]] = {
    "lateral": (
        _side(
            r"\bleft\b",
            r"\bto the left\b",
            r"\bleftwards?\b",
            rf"\b{_MOTION} left\b{_NOT_PLACE}",
        ),
        _side(
            r"\bright\b",
            r"\bto the right\b",
            r"\brightwards?\b",
            rf"\b{_MOTION} right\b{_NOT_PLACE}",
        ),
    ),
    "depth": (
        _side(
            r"\bcloser\b",
            r"\b(?:closer|nearer)\b(?!\s+to\b)",
            rf"\b(?:closer|nearer) to the {_VIEWER}\b",
            rf"\btowards? the {_VIEWER}\b",
        ),
        _side(
            r"\bfarther\b",
            r"\b(?:farther|further)\b(?!\s+(?:from|to)\b)",
            rf"\b(?:farther|further) from the {_VIEWER}\b",
            rf"\baway from the {_VIEWER}\b",
        ),
    ),
    "vertical": (
        _side(
            r"\bup\b",
            r"\bupwards?\b",
            rf"\b{_MOTION} up\b",
            r"\b(?:raised|lifted)\b",
        ),
        _side(
            r"\bdown\b",
            r"\bdownwards?\b",
            rf"\b{_MOTION} down\b",
            r"\blowered\b",
        ),
    ),
}
"""Per axis, the fact word and the motion phrases of each of two sides.

Bare words such as "right" or "up" are not motion phrases ("right next to",
"ended up"), so only phrases that state a motion count.
"""


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of ``validate_description``.

    Attributes:
        ok: True when no rule was violated.
        reasons: One short sentence per violation, each starting with the
            rule (``"type"``, ``"presence"``, ``"direction"``, ``"toward"``,
            ``"distance"``, or ``"rotation"``); empty when ``ok``.
    """

    ok: bool
    reasons: tuple[str, ...]


def _is_pure_rotation(fact: ChangeFact) -> bool:
    return (
        fact.rotation_text is not None
        and fact.displacement_text is None
        and fact.direction is None
        and fact.toward is None
    )


def _type_reasons(text: str, fact: ChangeFact) -> list[str]:
    if fact.type is ChangeType.UNCERTAIN:
        return []
    words = set(_LOWERCASE_WORD.findall(text))
    reasons = [
        f"type contradicts the fact: '{word}' for a {fact.type.value} change"
        for word in sorted(words & _CONTRADICTING[fact.type])
    ]
    names_change = _OWN[fact.type].search(text) is not None or (
        fact.type is ChangeType.MOVED
        and _is_pure_rotation(fact)
        and _TURN.search(text) is not None
    )
    if not names_change:
        reasons.append(
            f"type not stated: the sentence does not say the object "
            f"was {fact.type.value}"
        )
    return reasons


def _scrub(text: str, fact: ChangeFact) -> str:
    """Blanks the labels of the fact, so a label is not read as a word.

    A "sliding door" next to an added chair, a "missing person poster", or
    a "3 m cable" hold change words and quantities that say nothing about
    the change.
    """
    labels = {fact.label, fact.support_before, fact.support_after}
    labels.update(relation.label for relation in fact.near_before)
    labels.update(relation.label for relation in fact.near_after)
    ordered = sorted(
        (label.lower() for label in labels if label), key=len, reverse=True
    )
    for label in ordered:
        text = re.sub(rf"\b{re.escape(label)}\b", " ", text)
    return text


def _presence_reasons(text: str, fact: ChangeFact) -> list[str]:
    pattern = _PRESENCE.get(fact.type)
    if pattern is None:
        return []
    return [
        f"presence contradicts the fact: '{match.group()}' for a "
        f"{fact.type.value} change"
        for match in pattern.finditer(text)
    ]


def _axis_reasons(text: str, fact: ChangeFact) -> list[str]:
    reasons = []
    stated = (fact.direction or "").lower()
    for axis, (first, second) in _AXES.items():
        for side, opposite in ((first, second), (second, first)):
            if (
                side.fact.search(stated)
                and opposite.text.search(text)
                and not side.text.search(text)
            ):
                reasons.append(
                    f"direction contradicts the fact: {axis} axis, "
                    f"the fact says '{fact.direction}'"
                )
    return reasons


def _toward_reasons(text: str, fact: ChangeFact) -> list[str]:
    if fact.toward is None:
        return []
    if not re.search(
        rf"\baway from the {re.escape(fact.toward.lower())}\b", text
    ):
        return []
    return [f"toward contradicts the fact: moved toward the {fact.toward}"]


def _within(value: float, reference: float, absolute: float) -> bool:
    limit = max(absolute, RELATIVE_TOLERANCE * reference)
    return abs(value - reference) <= limit + _EPS


def _number_reasons(text: str, fact: ChangeFact) -> list[str]:
    references_m = [relation.distance_m for relation in fact.near_before]
    references_m += [relation.distance_m for relation in fact.near_after]
    if fact.displacement_m is not None:
        references_m.append(fact.displacement_m)
    reasons = []
    for meters in distances_m(text):
        if not any(
            _within(meters, ref, DISTANCE_TOLERANCE_M) for ref in references_m
        ):
            reasons.append(
                f"distance far from the fact: {meters * 100:g} cm "
                f"(allowed within 10 cm or 50 percent of a stated distance)"
            )
    for degrees in angles_deg(text):
        if fact.rotation_deg is None or not _within(
            degrees, fact.rotation_deg, ROTATION_TOLERANCE_DEG
        ):
            reasons.append(
                f"rotation far from the fact: {degrees:g} degrees "
                f"(allowed within 15 degrees or 50 percent of the rotation)"
            )
    return reasons


def validate_description(text: str, fact: ChangeFact) -> ValidationResult:
    """Checks a description against the fact it describes.

    Args:
        text: The description, in English.
        fact: The structured facts of the change.

    Returns:
        The result; ``reasons`` lists every violated rule.
    """
    if not text.strip():
        return ValidationResult(ok=False, reasons=("empty description",))
    lowered = text.lower()
    scrubbed = _scrub(lowered, fact)
    reasons = (
        _type_reasons(scrubbed, fact)
        + _presence_reasons(scrubbed, fact)
        + _axis_reasons(scrubbed, fact)
        + _toward_reasons(lowered, fact)
        + _number_reasons(scrubbed, fact)
    )
    return ValidationResult(ok=not reasons, reasons=tuple(reasons))


def repair_description(text: str, fact: ChangeFact) -> str:
    """Keeps a valid description and replaces an invalid one.

    Args:
        text: The description.
        fact: The structured facts of the change.

    Returns:
        ``text`` if it passes ``validate_description``, otherwise the
        template sentence of the fact.
    """
    if validate_description(text, fact).ok:
        return text
    return template_sentence(fact)
