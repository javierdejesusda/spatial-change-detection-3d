"""The deterministic sentence for a change fact.

The wording uses only what the fact states, so the sentence cannot
contradict the geometry. It is the describer of last resort: the validator
and the model-based describer fall back to it.
"""

from spatial_change.domain import ChangeType
from spatial_change.semantics.relations import ChangeFact, LandmarkRelation
from spatial_change.semantics.wording import with_article


def _capitalize(text: str) -> str:
    """Upper-cases the first character and leaves the rest as it is."""
    return text[:1].upper() + text[1:]


def _location(
    preposition: str, surface: str | None, near: list[LandmarkRelation]
) -> str:
    """Phrases where an object is, or was, from its surface and landmarks.

    Args:
        preposition: Word that introduces the surface, such as ``"on"``.
        surface: What the object rests on, if known.
        near: Nearby landmarks, nearest first.

    Returns:
        The phrase with a leading space, or an empty string if there is
        nothing to say. The nearest landmark that is not the surface is
        named.
    """
    parts = []
    if surface is not None:
        parts.append(f" {preposition} the {surface}")
    for landmark in near:
        if landmark.label != surface:
            parts.append(f" {landmark.relation} the {landmark.label}")
            break
    return "".join(parts)


def _moved(fact: ChangeFact) -> str:
    """Words a moved object: distance, direction, target, surface, turn."""
    motion = " ".join(
        part for part in (fact.displacement_text, fact.direction) if part
    )
    targets = []
    if fact.toward is not None:
        targets.append(f"toward the {fact.toward}")
    if fact.support_after not in (None, fact.support_before):
        targets.append(f"onto the {fact.support_after}")
    if fact.rotation_text is not None and not (motion or targets):
        return f"turned {fact.rotation_text}"
    text = "moved" + (f" {motion}" if motion else "")
    for index, target in enumerate(targets):
        text += (", " if motion or index else " ") + target
    if fact.rotation_text is not None:
        text += f"{',' if targets else ''} and turned {fact.rotation_text}"
    return text


def template_sentence(fact: ChangeFact) -> str:
    """Words a change fact as one sentence.

    Args:
        fact: The structured facts of the change.

    Returns:
        A capitalized sentence ending in a period. A moved object gets
        the distance, direction, target landmark, new surface, and turn the
        fact has; an added or removed object gets its surface and nearest
        landmark; an uncertain change says the views do not settle it.
    """
    label = fact.label
    match fact.type:
        case ChangeType.MOVED:
            return f"The {label} {_moved(fact)}."
        case ChangeType.ADDED:
            where = _location("on", fact.support_after, fact.near_after)
            return f"{_capitalize(with_article(label))} was added{where}."
        case ChangeType.REMOVED:
            where = _location("from", fact.support_before, fact.near_before)
            return f"{_capitalize(with_article(label))} was removed{where}."
        case ChangeType.UNCERTAIN:
            return f"The {label} may have changed; the views do not settle it."
