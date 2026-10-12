"""Describers: from a change fact and its crops to one sentence."""

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from spatial_change.domain import ChangeType
from spatial_change.ports import ChangeCrops, Describer, VisionLanguageModel
from spatial_change.semantics.relations import ChangeFact, LandmarkRelation
from spatial_change.semantics.template import template_sentence
from spatial_change.semantics.validator import validate_description
from spatial_change.semantics.wording import count_phrase

_MAX_DESCRIPTION_CHARS = 300
"""Longest accepted description; a longer reply is unusable."""
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

_INSTRUCTION = (
    "Write one short sentence in plain English that describes this change. "
    "Use only the facts above: do not state a change type, direction, "
    "distance, or angle that differs from them. "
    'Reply with strict JSON only, in the form {"description": '
    '"<one sentence>"}.'
)

_CORRECTION = (
    "\n\nYour previous reply was not valid. Reply again with only the JSON "
    'object {"description": "<one sentence>"}, with no other text.'
)


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    description: str

    @field_validator("description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must not be blank")
        return value


@dataclass(frozen=True)
class LocalVlmConfig:
    """Settings of the model-based describer.

    Attributes:
        max_crops_per_side: How many before crops and how many after crops
            are shown to the model, taken in order.

    Raises:
        ValueError: If ``max_crops_per_side`` is below 1.
    """

    max_crops_per_side: int = 1

    def __post_init__(self) -> None:
        """Validates the settings."""
        if self.max_crops_per_side < 1:
            raise ValueError("max_crops_per_side must be >= 1")


class TemplateDescriber:
    """Describes a change with the deterministic template.

    It needs no model and no crops, so it also serves as the fallback of
    model-based describers.
    """

    def describe(self, fact: ChangeFact, crops: ChangeCrops) -> str:
        """Words the fact with the template.

        Args:
            fact: The structured facts of the change.
            crops: Ignored.

        Returns:
            The template sentence of the fact.
        """
        return template_sentence(fact)

    def unload(self) -> None:
        """Does nothing; the template holds no resources."""


def _relations(relations: list[LandmarkRelation]) -> str:
    return ", ".join(f"{r.relation} the {r.label}" for r in relations)


def _fact_lines(fact: ChangeFact) -> list[str]:
    """States the fact as plain text, one line per part it has."""
    candidates = [
        ("Change type", fact.type.value),
        ("Object", fact.label),
        ("Rested on before", fact.support_before),
        ("Rests on after", fact.support_after),
        ("Nearby before", _relations(fact.near_before)),
        ("Nearby after", _relations(fact.near_after)),
        ("Direction", fact.direction),
        ("Moved toward", fact.toward),
        ("Distance", fact.displacement_text),
        ("Rotation", fact.rotation_text),
    ]
    return [f"{name}: {value}" for name, value in candidates if value]


def _image_line(before: int, after: int) -> str | None:
    """Tells the model which images show which capture."""
    if not before and not after:
        return None
    if not after:
        return "The images show the object before the change."
    if not before:
        return "The images show the object after the change."
    return (
        f"The first {_images_show(before)} the object before the change; "
        f"the next {_images_show(after)} it after the change."
    )


def _images_show(count: int) -> str:
    verb = "shows" if count == 1 else "show"
    return f"{count_phrase(count, 'image', indefinite=False)} {verb}"


_ABBREVIATIONS = frozenset(
    {"approx", "e.g", "i.e", "etc", "vs", "fig", "ca", "cf", "est"}
)
_SENTENCE_END = re.compile(r"[.!?]+\s+(?=[A-Z])")


def _first_sentence(text: str) -> str:
    """Cuts a text after its first sentence.

    A sentence ends at ``.``, ``!``, or ``?`` followed by whitespace and a
    capital letter, unless the word before the mark is a common
    abbreviation such as "approx." or "e.g.".

    Args:
        text: Text with its whitespace already collapsed.

    Returns:
        The first sentence, or the whole text if it holds one.
    """
    for match in _SENTENCE_END.finditer(text):
        word = text[: match.start()].split(" ")[-1].lower()
        if word not in _ABBREVIATIONS:
            return text[: match.end()].rstrip()
    return text


def _extract_reply(text: str) -> str | None:
    """Reads the description from a model reply, if it holds a usable one.

    Args:
        text: The raw reply; a reasoning block or text around the JSON
            object is ignored.

    Returns:
        The first sentence of the description with its whitespace collapsed,
        or ``None`` if there is no JSON object of
        the required form.
    """
    text = _THINK_BLOCK.sub("", text)
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
        reply = _Reply.model_validate(payload)
    except (ValueError, ValidationError):
        return None
    sentence = _first_sentence(" ".join(reply.description.split()))
    if len(sentence) > _MAX_DESCRIPTION_CHARS:
        return None
    return sentence


class LocalVlmDescriber:
    """Describes a change with a vision-language model.

    The model sees the fact as text, plus up to ``max_crops_per_side``
    before and after crops, and must answer with a JSON object. An unusable
    reply is retried once with a corrective prompt. A description that
    contradicts the fact (``validate_description``), or a second unusable
    reply, is replaced by the fallback describer, so the result does not
    contradict the type, direction, or numbers the validator checks. The
    validator is rule-based and loose in places: a wrong displacement that
    equals a nearby landmark distance still passes.
    """

    def __init__(
        self,
        vlm: VisionLanguageModel,
        fallback: Describer,
        config: LocalVlmConfig | None = None,
    ) -> None:
        """Creates the describer.

        Args:
            vlm: The model that writes the descriptions.
            fallback: Describer used when the model gives nothing usable;
                its lifetime belongs to the caller.
            config: Settings; the defaults if None.
        """
        self._vlm = vlm
        self._fallback = fallback
        self._config = config or LocalVlmConfig()

    def describe(self, fact: ChangeFact, crops: ChangeCrops) -> str:
        """Describes one change.

        Args:
            fact: Structured spatial facts about the change.
            crops: Before and after crops of the object, possibly empty.

        Returns:
            The model's sentence if it parses and passes the validator,
            otherwise the fallback's. An uncertain change always goes to
            the fallback: images cannot express doubt about a change, so
            the model is not asked.
        """
        if fact.type is ChangeType.UNCERTAIN:
            return self._fallback.describe(fact, crops)
        limit = self._config.max_crops_per_side
        before, after = crops.before[:limit], crops.after[:limit]
        lines = _fact_lines(fact)
        image_line = _image_line(len(before), len(after))
        if image_line is not None:
            lines.append(image_line)
        prompt = "\n".join(lines) + "\n\n" + _INSTRUCTION
        images = [*before, *after]
        for attempt_prompt in (prompt, prompt + _CORRECTION):
            reply = self._vlm.generate(images, attempt_prompt)
            text = _extract_reply(reply)
            if text is None:
                continue
            if validate_description(text, fact).ok:
                return text
            break
        return self._fallback.describe(fact, crops)

    def unload(self) -> None:
        """Unloads the model."""
        self._vlm.unload()
