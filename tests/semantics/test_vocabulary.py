from spatial_change.semantics.vocabulary import (
    FURNITURE,
    INDOOR_OBJECTS,
    STRUCTURE,
)
from spatial_change.synthetic.objects import FACTORIES


def test_the_indoor_vocabulary_is_a_sorted_list_of_unique_names() -> None:
    assert 40 <= len(INDOOR_OBJECTS) <= 60
    assert list(INDOOR_OBJECTS) == sorted(set(INDOOR_OBJECTS))


def test_the_indoor_vocabulary_covers_every_synthetic_class() -> None:
    assert set(FACTORIES) <= set(INDOOR_OBJECTS)


def test_furniture_is_static_and_part_of_the_indoor_vocabulary() -> None:
    assert set(FURNITURE) <= set(INDOOR_OBJECTS)
    assert {"table", "shelf", "cabinet", "sofa", "bed"} <= set(FURNITURE)
    assert "chair" not in FURNITURE


def test_structure_is_disjoint_from_objects_and_furniture() -> None:
    names = FURNITURE + STRUCTURE

    assert set(STRUCTURE) == {"wall", "floor", "ceiling", "door", "window"}
    assert not set(STRUCTURE) & set(INDOOR_OBJECTS)
    assert len(set(names)) == len(names)
