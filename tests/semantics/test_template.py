import pytest
from facts import make_fact, moved_fact, near
from hypothesis import given
from hypothesis import strategies as st

from spatial_change.domain import ChangeType
from spatial_change.semantics.template import template_sentence
from spatial_change.semantics.validator import validate_description


def test_a_full_move_names_distance_direction_target_and_turn() -> None:
    assert template_sentence(moved_fact()) == (
        "The chair moved about 30 cm to the left, toward the table, "
        "and turned about 70 degrees."
    )


def test_a_move_without_a_rotation_has_no_turn() -> None:
    fact = moved_fact(rotation_deg=None, rotation_text=None)

    assert template_sentence(fact) == (
        "The chair moved about 30 cm to the left, toward the table."
    )


def test_a_move_without_toward_has_no_target() -> None:
    fact = moved_fact(toward=None)

    assert template_sentence(fact) == (
        "The chair moved about 30 cm to the left and turned about 70 degrees."
    )


def test_a_move_without_a_direction_keeps_the_distance() -> None:
    fact = moved_fact(direction=None, toward=None, rotation_text=None)

    assert template_sentence(fact) == "The chair moved about 30 cm."


def test_a_move_without_a_distance_keeps_the_direction() -> None:
    fact = moved_fact(displacement_text=None, toward=None, rotation_text=None)

    assert template_sentence(fact) == "The chair moved to the left."


def test_a_pure_rotation_is_worded_as_a_turn() -> None:
    fact = moved_fact(
        direction=None,
        toward=None,
        displacement_m=0.01,
        displacement_text=None,
    )

    assert template_sentence(fact) == ("The chair turned about 70 degrees.")


def test_a_move_with_no_words_is_still_a_move() -> None:
    fact = moved_fact(
        direction=None,
        toward=None,
        displacement_text=None,
        rotation_text=None,
    )

    assert template_sentence(fact) == "The chair moved."


def test_a_move_onto_a_new_surface_names_it() -> None:
    fact = moved_fact(
        toward=None,
        rotation_text=None,
        support_before="floor",
        support_after="table",
    )

    assert template_sentence(fact) == (
        "The chair moved about 30 cm to the left, onto the table."
    )


def test_an_unchanged_surface_is_not_repeated() -> None:
    fact = moved_fact(
        toward=None,
        rotation_text=None,
        support_before="floor",
        support_after="floor",
    )

    assert template_sentence(fact) == (
        "The chair moved about 30 cm to the left."
    )


def test_a_vertical_direction_reads_naturally() -> None:
    fact = moved_fact(direction="up", toward=None, rotation_text=None)

    assert template_sentence(fact) == "The chair moved about 30 cm up."


def test_an_added_object_names_its_surface() -> None:
    fact = make_fact(ChangeType.ADDED, label="drum", support_after="table")

    assert template_sentence(fact) == "A drum was added on the table."


def test_an_added_object_names_the_nearest_landmark() -> None:
    fact = make_fact(
        ChangeType.ADDED,
        label="crate",
        support_after="floor",
        near_after=[near("shelf", "next to"), near("lamp")],
    )

    assert template_sentence(fact) == (
        "A crate was added on the floor next to the shelf."
    )


def test_an_added_object_without_context_is_bare() -> None:
    fact = make_fact(ChangeType.ADDED, label="armchair")

    assert template_sentence(fact) == "An armchair was added."


def test_a_removed_object_uses_the_before_context() -> None:
    fact = make_fact(
        ChangeType.REMOVED,
        label="crate",
        support_before="floor",
        near_before=[near("shelf")],
        support_after="table",
        near_after=[near("lamp")],
    )

    assert template_sentence(fact) == (
        "A crate was removed from the floor near the shelf."
    )


def test_a_removed_object_without_a_surface_names_the_landmark() -> None:
    fact = make_fact(
        ChangeType.REMOVED, label="crate", near_before=[near("shelf")]
    )

    assert template_sentence(fact) == "A crate was removed near the shelf."


def test_a_landmark_equal_to_the_surface_is_not_repeated() -> None:
    fact = make_fact(
        ChangeType.ADDED,
        label="drum",
        support_after="table",
        near_after=[near("table")],
    )

    assert template_sentence(fact) == "A drum was added on the table."


def test_an_uncertain_change_says_the_views_do_not_settle_it() -> None:
    fact = make_fact(ChangeType.UNCERTAIN, label="stool")

    assert template_sentence(fact) == (
        "The stool may have changed; the views do not settle it."
    )


def test_the_sentence_is_deterministic() -> None:
    fact = moved_fact()

    assert template_sentence(fact) == template_sentence(fact)


@pytest.mark.parametrize("change_type", list(ChangeType))
def test_every_type_gives_one_sentence_ending_in_a_period(
    change_type: ChangeType,
) -> None:
    text = template_sentence(make_fact(change_type, label="lamp"))

    assert text.endswith(".")
    assert text[0].isupper()
    assert text.count(".") == 1


def test_a_turn_after_a_surface_is_set_off_by_a_comma() -> None:
    fact = moved_fact(
        toward=None, support_before="floor", support_after="table"
    )

    assert template_sentence(fact) == (
        "The chair moved about 30 cm to the left, onto the table, "
        "and turned about 70 degrees."
    )


def test_only_the_first_character_of_the_label_is_capitalized() -> None:
    assert template_sentence(make_fact(ChangeType.ADDED, label="TV")) == (
        "A TV was added."
    )
    assert template_sentence(
        make_fact(ChangeType.REMOVED, label="OLED panel")
    ) == ("An OLED panel was removed.")
    assert template_sentence(make_fact(ChangeType.ADDED, label="iPad")) == (
        "An iPad was added."
    )


LABELS = st.sampled_from(
    [
        "chair",
        "TV",
        "sliding door",
        "moving box",
        "left cabinet",
        "removed items bin",
        "turned table",
        "missing person poster",
        "gone fishing sign",
        "new lamp",
        "pushing cart",
        "up stairs",
        "right shelf",
        "still life",
    ]
)


@given(
    change_type=st.sampled_from(list(ChangeType)),
    label=LABELS,
    landmark=LABELS,
    surface=st.one_of(st.none(), st.just("floor"), LABELS),
    move=st.sampled_from(["none", "full", "rotation"]),
)
def test_the_template_always_passes_the_validator(
    change_type: ChangeType,
    label: str,
    landmark: str,
    surface: str | None,
    move: str,
) -> None:
    moving = {}
    if move == "full":
        moving = {
            "direction": "to the left",
            "toward": landmark,
            "displacement_m": 0.3,
            "displacement_text": "about 30 cm",
            "rotation_deg": 40.0,
            "rotation_text": "about 40 degrees",
        }
    elif move == "rotation":
        moving = {
            "direction": None,
            "toward": None,
            "displacement_m": 0.01,
            "displacement_text": None,
            "rotation_deg": 40.0,
            "rotation_text": "about 40 degrees",
        }
    fact = make_fact(
        change_type,
        label=label,
        support_before=surface,
        support_after=surface,
        near_before=[near(landmark)],
        near_after=[near(landmark)],
        **moving,
    )

    text = template_sentence(fact)

    assert validate_description(text, fact).ok, text
