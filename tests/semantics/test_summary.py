import numpy as np
import pytest
from facts import make_fact

from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeReport,
    ChangeType,
    EvidenceCounts,
    OrientedBox,
    UnobservedRegion,
)
from spatial_change.domain.camera import Epoch
from spatial_change.semantics.summary import (
    summarize_changes,
    summarize_report,
)


def _box() -> BoxModel:
    return BoxModel.from_box(
        OrientedBox(center=np.zeros(3), extent=np.ones(3), rotation=np.eye(3))
    )


def _change(change_type: ChangeType, label: str) -> Change:
    boxes = {
        ChangeType.ADDED: (None, _box()),
        ChangeType.REMOVED: (_box(), None),
    }
    before, after = boxes.get(change_type, (_box(), _box()))
    moved = change_type is ChangeType.MOVED
    return Change(
        change_id=f"{change_type.value}-{label}",
        type=change_type,
        label=label,
        description="",
        confidence=0.5,
        before_box=before,
        after_box=after,
        displacement_m=0.3 if moved else None,
        rotation_deg=10.0 if moved else None,
        evidence=EvidenceCounts(consistent=1, free=0, occluded=0, unobserved=0),
    )


def _region(region_id: str) -> UnobservedRegion:
    return UnobservedRegion(
        region_id=region_id, box=_box(), epoch_seen=Epoch.BEFORE
    )


def test_the_example_sentence() -> None:
    facts = [
        make_fact(ChangeType.ADDED, label="drum"),
        make_fact(ChangeType.MOVED, label="chair"),
        make_fact(ChangeType.REMOVED, label="crate"),
    ]

    assert summarize_changes(facts) == (
        "3 changes: a chair moved, a crate was removed, a drum was added."
    )


def test_no_changes() -> None:
    assert summarize_changes([]) == "No changes were found."


def test_one_change_is_singular() -> None:
    summary = summarize_changes([make_fact(ChangeType.ADDED, label="lamp")])

    assert summary == "1 change: a lamp was added."


def test_the_order_is_moved_removed_added_uncertain() -> None:
    facts = [
        make_fact(ChangeType.UNCERTAIN, label="stool"),
        make_fact(ChangeType.ADDED, label="drum"),
        make_fact(ChangeType.REMOVED, label="crate"),
        make_fact(ChangeType.MOVED, label="chair"),
    ]

    assert summarize_changes(facts) == (
        "4 changes: a chair moved, a crate was removed, "
        "a drum was added, a stool may have changed."
    )


def test_labels_of_one_type_are_joined() -> None:
    facts = [
        make_fact(ChangeType.MOVED, label="chair"),
        make_fact(ChangeType.MOVED, label="table"),
        make_fact(ChangeType.MOVED, label="lamp"),
        make_fact(ChangeType.REMOVED, label="crate"),
        make_fact(ChangeType.REMOVED, label="armchair"),
    ]

    assert summarize_changes(facts) == (
        "5 changes: a chair, a table, and a lamp moved, "
        "a crate and an armchair were removed."
    )


def test_the_input_order_does_not_change_the_summary_of_types() -> None:
    first = [
        make_fact(ChangeType.REMOVED, label="crate"),
        make_fact(ChangeType.MOVED, label="chair"),
    ]

    assert summarize_changes(first) == summarize_changes(first[::-1])


def test_unobserved_regions_are_mentioned_but_not_counted() -> None:
    summary = summarize_changes(
        [make_fact(ChangeType.MOVED, label="chair")], unobserved_regions=2
    )

    assert summary == (
        "1 change: a chair moved. "
        "2 regions could not be judged because only one capture saw them."
    )


def test_one_unobserved_region_is_singular() -> None:
    assert summarize_changes([], unobserved_regions=1) == (
        "No changes were found. "
        "1 region could not be judged because only one capture saw it."
    )


def test_a_negative_unobserved_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="unobserved_regions"):
        summarize_changes([], unobserved_regions=-1)


def test_changes_of_a_report_are_summarized() -> None:
    changes = [_change(ChangeType.MOVED, "chair")]

    assert summarize_changes(changes) == "1 change: a chair moved."


def test_a_report_counts_its_unobserved_regions() -> None:
    report = ChangeReport(
        changes=[
            _change(ChangeType.ADDED, "drum"),
            _change(ChangeType.REMOVED, "crate"),
        ],
        unobserved_regions=[_region("r1")],
        metadata={},
    )

    assert summarize_report(report) == (
        "2 changes: a crate was removed, a drum was added. "
        "1 region could not be judged because only one capture saw it."
    )


def test_an_empty_report() -> None:
    report = ChangeReport(changes=[], unobserved_regions=[], metadata={})

    assert summarize_report(report) == "No changes were found."


def _facts(change_type: ChangeType, *labels: str) -> list:
    return [make_fact(change_type, label=label) for label in labels]


def test_equal_labels_are_grouped_with_a_count() -> None:
    facts = _facts(ChangeType.REMOVED, "chair", "chair")

    assert summarize_changes(facts) == "2 changes: two chairs were removed."


def test_one_label_is_not_counted() -> None:
    facts = _facts(ChangeType.MOVED, "chair")

    assert summarize_changes(facts) == "1 change: a chair moved."


def test_groups_keep_first_occurrence_order() -> None:
    facts = _facts(ChangeType.ADDED, "lamp", "drum", "lamp", "armchair", "lamp")

    assert summarize_changes(facts) == (
        "5 changes: three lamps, a drum, and an armchair were added."
    )


def test_a_moved_group_keeps_the_verb() -> None:
    facts = _facts(ChangeType.MOVED, "chair", "chair", "table")

    assert summarize_changes(facts) == (
        "3 changes: two chairs and a table moved."
    )


def test_uncertain_groups_say_may_have_changed() -> None:
    facts = _facts(ChangeType.UNCERTAIN, "stool", "stool")

    assert summarize_changes(facts) == (
        "2 changes: two stools may have changed."
    )


@pytest.mark.parametrize(
    ("count", "word"),
    [
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
        (7, "seven"),
        (8, "eight"),
        (9, "nine"),
        (10, "ten"),
        (11, "11"),
        (12, "12"),
    ],
)
def test_counts_up_to_ten_are_words_then_digits(count: int, word: str) -> None:
    facts = _facts(ChangeType.ADDED, *["drum"] * count)

    assert summarize_changes(facts) == (
        f"{count} changes: {word} drums were added."
    )


def test_irregular_plurals_are_used() -> None:
    facts = _facts(ChangeType.REMOVED, "shelf", "shelf", "person", "person")

    assert summarize_changes(facts) == (
        "4 changes: two shelves and two people were removed."
    )
