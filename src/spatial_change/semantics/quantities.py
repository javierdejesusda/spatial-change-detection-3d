"""Distances and angles quoted in a sentence.

The readers are deliberately small: digits and spelled-out numbers up to
ninety-nine, ranges, hyphenated units, and a few fractions. Anything else is
not read, so it is neither checked nor rejected.
"""

import re

from spatial_change.semantics.wording import NUMBER_WORDS

_WORD_VALUES = {word: value for value, word in enumerate(NUMBER_WORDS)}
_WORD = "|".join(
    re.escape(word).replace(r"\-", "[- ]")
    for word in sorted(NUMBER_WORDS, key=len, reverse=True)
)
_NUMBER = rf"(?:(?<![\d.])(?:\d+\.\d+|\.\d+|\d+)|\b(?:{_WORD})\b)"
_RANGE = r"(?:\s*[-–]\s*|\s+to\s+)"
_QUANTITY = rf"(?P<first>{_NUMBER})(?:{_RANGE}(?P<second>{_NUMBER}))?\s*-?\s*"

_UNIT_SCALE: dict[str, float] = {
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
    "mm": 0.001,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "cm": 0.01,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "m": 1.0,
    "foot": 0.3048,
    "feet": 0.3048,
    "ft": 0.3048,
    "inch": 0.0254,
    "inches": 0.0254,
    "in": 0.0254,
}
"""Length units and their size in meters.

"in" is also a preposition, so it counts only after digits and not before
"the", "a", "an", "this", "that", "front", or "back".
"""

_UNIT_PATTERN = "|".join(
    re.escape(unit)
    + (
        r"(?!\w)(?!\s+(?:the|a|an|this|that|front|back)\b)"
        if unit == "in"
        else ""
    )
    for unit in sorted(_UNIT_SCALE, key=len, reverse=True)
)
_LENGTH = re.compile(rf"{_QUANTITY}(?P<unit>{_UNIT_PATTERN})\b", re.IGNORECASE)
_HALF_METER = re.compile(
    rf"(?:(?P<whole>{_NUMBER})\s+and\s+a\s+half|\b(?:a|one)\s+half"
    r"|\bhalf(?:\s+an?)?)\s*-?\s*(?:met(?:er|re)s?|m)\b",
    re.IGNORECASE,
)
_ANGLE = re.compile(rf"{_QUANTITY}(?:°|degrees?\b|deg\b)", re.IGNORECASE)
_QUARTER_TURN = re.compile(
    r"(?P<three>\bthree[\s-]+)?\bquarters?(?:\s+of\s+an?)?[\s-]+turn\b",
    re.IGNORECASE,
)
_HALF_TURN = re.compile(r"\bhalf(?:\s+an?)?[\s-]+turn\b", re.IGNORECASE)


def _value(token: str) -> float:
    """Reads a digit or spelled-out number such as ``"twenty-five"``."""
    if token[0].isdigit() or token[0] == ".":
        return float(token)
    return float(_WORD_VALUES[token.replace(" ", "-")])


def _numbers(match: re.Match[str]) -> list[float]:
    values = [_value(match.group("first").lower())]
    if match.group("second") is not None:
        values.append(_value(match.group("second").lower()))
    return values


def distances_m(text: str) -> list[float]:
    """Lists the distances a sentence quotes, in meters.

    Args:
        text: The sentence.

    Returns:
        Every number of every quoted distance, so both ends of a range are
        listed. Units: mm, cm, m, feet, and inches (``"in"`` only after
        digits); a half meter in several wordings counts as 0.5.
    """
    found: list[float] = []
    for match in _LENGTH.finditer(text):
        unit = match.group("unit").lower()
        if unit == "in" and not match.group("first")[0].isdigit():
            continue
        found.extend(value * _UNIT_SCALE[unit] for value in _numbers(match))
    for match in _HALF_METER.finditer(text):
        whole = match.group("whole")
        found.append(0.5 + (_value(whole.lower()) if whole else 0.0))
    return found


def angles_deg(text: str) -> list[float]:
    """Lists the angles a sentence quotes, in degrees.

    Args:
        text: The sentence.

    Returns:
        Every number of every quoted angle, so both ends of a range are
        listed. A quarter turn counts as 90, a half turn as 180, and a
        three-quarter turn as 270.
    """
    found: list[float] = []
    for match in _ANGLE.finditer(text):
        found.extend(_numbers(match))
    for match in _QUARTER_TURN.finditer(text):
        found.append(270.0 if match.group("three") else 90.0)
    found.extend(180.0 for _ in _HALF_TURN.finditer(text))
    return found
