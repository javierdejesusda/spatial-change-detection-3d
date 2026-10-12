"""A deterministic one-paragraph summary of the changes of a scene."""

from collections import Counter
from collections.abc import Sequence

from spatial_change.domain import Change, ChangeReport, ChangeType
from spatial_change.semantics.relations import ChangeFact
from spatial_change.semantics.wording import count_phrase, join_and

_ORDER = (
    ChangeType.MOVED,
    ChangeType.REMOVED,
    ChangeType.ADDED,
    ChangeType.UNCERTAIN,
)
"""The order in which change types are listed."""


def _clause(change_type: ChangeType, labels: list[str]) -> str:
    """Words all changes of one type, such as ``"a crate was removed"``.

    Equal labels are grouped in order of first occurrence.
    """
    counts = Counter(labels)
    names = join_and([count_phrase(n, label) for label, n in counts.items()])
    plural = len(labels) > 1
    match change_type:
        case ChangeType.MOVED:
            return f"{names} moved"
        case ChangeType.UNCERTAIN:
            return f"{names} may have changed"
        case ChangeType.REMOVED | ChangeType.ADDED:
            verb = "were" if plural else "was"
            return f"{names} {verb} {change_type.value}"


def _unobserved(count: int) -> str:
    if count == 1:
        return "1 region could not be judged because only one capture saw it."
    return (
        f"{count} regions could not be judged because only one capture "
        "saw them."
    )


def summarize_changes(
    changes: Sequence[Change | ChangeFact], unobserved_regions: int = 0
) -> str:
    """Summarizes the changes of a scene.

    Args:
        changes: The changes, as report changes or facts.
        unobserved_regions: How many regions only one capture observed;
            they are mentioned but never counted as changes.

    Returns:
        ``"N changes: ..."`` with the labels grouped by type in the order
        moved, removed, added, uncertain (input order within a type), or
        ``"No changes were found."``, followed by a sentence about the
        unobserved regions if there are any.

    Raises:
        ValueError: If ``unobserved_regions`` is negative.
    """
    if unobserved_regions < 0:
        raise ValueError("unobserved_regions must be >= 0")
    if changes:
        clauses = []
        for change_type in _ORDER:
            labels = [c.label for c in changes if c.type is change_type]
            if labels:
                clauses.append(_clause(change_type, labels))
        noun = "change" if len(changes) == 1 else "changes"
        text = f"{len(changes)} {noun}: {', '.join(clauses)}."
    else:
        text = "No changes were found."
    if unobserved_regions:
        text += " " + _unobserved(unobserved_regions)
    return text


def summarize_report(report: ChangeReport) -> str:
    """Summarizes a change report.

    Args:
        report: The report.

    Returns:
        The summary of its changes and unobserved regions, as
        ``summarize_changes`` words it.
    """
    return summarize_changes(report.changes, len(report.unobserved_regions))
