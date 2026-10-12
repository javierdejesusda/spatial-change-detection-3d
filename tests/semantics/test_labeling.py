import dataclasses

import numpy as np
import pytest
from fakes import ScriptedLabeler
from lifting_scenes import Cuboid, HandScene, hand_scene, ring_poses
from matching_scenes import instance

from spatial_change.domain import Epoch, EpochReconstruction, ObjectInstance
from spatial_change.geometry.views import ViewConfig, usable_frames
from spatial_change.perception import FakeLabeler
from spatial_change.semantics.labeling import (
    LabelConfig,
    label_instances,
    label_point_sets,
)
from spatial_change.semantics.vocabulary import INDOOR_OBJECTS

RED_BOX = Cuboid((-1.2, -0.4, 0.0), (-0.6, 0.2, 0.6))
BLUE_BOX = Cuboid((0.6, -0.3, 0.0), (1.2, 0.3, 0.9))
COLORS = {"red": (200, 20, 20), "blue": (20, 20, 200)}
VOCABULARY = ["blue", "red"]
SMALL = LabelConfig(min_crop_px=1)
TWO_VIEWS = LabelConfig(views=ViewConfig(max_views=2), min_crop_px=1)


def _scene(epoch: Epoch = Epoch.BEFORE) -> HandScene:
    scene = hand_scene(
        [RED_BOX, BLUE_BOX], ring_poses(8, radius=1.7, height=1.4), epoch
    )
    frames = []
    for frame, instances in zip(scene.frames, scene.instance_maps, strict=True):
        assert frame.image is not None
        image = frame.image.copy()
        image[instances == 0] = COLORS["red"]
        image[instances == 1] = COLORS["blue"]
        frames.append(dataclasses.replace(frame, image=image))
    return HandScene(frames, scene.instance_maps, epoch)


def _reconstruction(scene: HandScene) -> EpochReconstruction:
    points = scene.object_points(0)
    count = len(points)
    return EpochReconstruction(
        epoch=scene.epoch,
        frames=scene.frames,
        points=points,
        colors=np.zeros((count, 3), np.uint8),
        confidence=np.ones(count, np.float32),
    )


def _red(scene: HandScene) -> ObjectInstance:
    return instance(scene.object_points(0), "red", scene.epoch)


def _blue(scene: HandScene) -> ObjectInstance:
    return instance(scene.object_points(1), "blue", scene.epoch)


def _label_red(
    labeler: ScriptedLabeler,
    config: LabelConfig,
    scene: HandScene | None = None,
) -> ObjectInstance:
    scene = scene or _scene()
    (labeled,) = label_instances(
        _reconstruction(scene), (_red(scene),), labeler, VOCABULARY, config
    )
    return labeled


@pytest.mark.parametrize(
    "overrides",
    [
        {"margin": -0.1},
        {"margin": float("inf")},
        {"min_margin_px": -1.0},
        {"min_crop_px": 0},
        {"min_score": -0.1},
        {"min_score": 1.1},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LabelConfig(**overrides)  # type: ignore[arg-type]


def test_objects_get_the_label_of_their_color() -> None:
    scene = _scene()

    labeled = label_instances(
        _reconstruction(scene),
        (_red(scene), _blue(scene)),
        FakeLabeler(COLORS),
        VOCABULARY,
        SMALL,
    )

    assert [i.label for i in labeled] == ["red", "blue"]
    assert [i.object_id for i in labeled] == ["red", "blue"]
    for item in labeled:
        assert item.label_score is not None
        assert item.label_score > 0.5


def test_the_inputs_are_not_changed() -> None:
    scene = _scene()
    item = _red(scene)

    (labeled,) = label_instances(
        _reconstruction(scene), (item,), FakeLabeler(COLORS), VOCABULARY, SMALL
    )

    assert item.label is None
    assert labeled is not item
    assert labeled.points is item.points


def test_one_labeler_call_serves_all_instances_and_views() -> None:
    scene = _scene()
    labeler = ScriptedLabeler([("red", 0.9)] * 100)

    label_instances(
        _reconstruction(scene),
        (_red(scene), _blue(scene)),
        labeler,
        VOCABULARY,
        LabelConfig(views=ViewConfig(max_views=3), min_crop_px=1),
    )

    ((crops, vocabulary),) = labeler.calls
    assert len(crops) == 6
    assert vocabulary == VOCABULARY


def test_crops_come_from_the_frames_that_see_the_object() -> None:
    labeler = ScriptedLabeler([("red", 0.9)] * 100)

    _label_red(labeler, TWO_VIEWS)

    ((crops, _),) = labeler.calls
    assert len(crops) == 2
    for crop in crops:
        assert crop.dtype == np.uint8
        assert (crop == COLORS["red"]).all(axis=2).any()
        assert not (crop == COLORS["blue"]).all(axis=2).any()


def test_a_crop_is_the_object_with_a_margin_around_it() -> None:
    labeler = ScriptedLabeler([("red", 0.9)])
    config = LabelConfig(
        views=ViewConfig(max_views=1),
        min_crop_px=1,
        margin=0.1,
        min_margin_px=4,
    )

    _label_red(labeler, config)

    ((crops, _),) = labeler.calls
    (crop,) = crops
    rows, cols = np.nonzero((crop == COLORS["red"]).all(axis=2))
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    pad_y = max(0.1 * height, 4)
    pad_x = max(0.1 * width, 4)
    assert abs(crop.shape[0] - (height + 2 * pad_y)) <= 2
    assert abs(crop.shape[1] - (width + 2 * pad_x)) <= 2
    assert abs(rows.min() - pad_y) <= 1
    assert abs(cols.min() - pad_x) <= 1


def test_the_margin_is_at_least_the_minimum_in_pixels() -> None:
    shapes = []
    for margin_px in (0.0, 6.0):
        labeler = ScriptedLabeler([("red", 0.9)])
        config = LabelConfig(
            views=ViewConfig(max_views=1),
            min_crop_px=1,
            margin=0.0,
            min_margin_px=margin_px,
        )
        _label_red(labeler, config)
        shapes.append(labeler.calls[0][0][0].shape)

    assert shapes[1][0] >= shapes[0][0] + 11
    assert shapes[1][1] >= shapes[0][1] + 11


def test_crops_stay_inside_the_image() -> None:
    scene = _scene()
    labeler = ScriptedLabeler([("red", 0.9)] * 100)

    _label_red(labeler, LabelConfig(min_crop_px=1, min_margin_px=500.0), scene)

    ((crops, _),) = labeler.calls
    image = scene.frames[0].image
    assert image is not None
    assert crops
    assert all(c.shape == image.shape for c in crops)


def test_views_with_a_small_crop_are_skipped() -> None:
    labeler = ScriptedLabeler([("red", 0.9)] * 100)

    labeled = _label_red(labeler, LabelConfig(min_crop_px=10**6))

    assert labeler.calls == []
    assert labeled.label is None
    assert labeled.label_score is None


@pytest.mark.parametrize(
    ("results", "score"),
    [
        ([("red", 0.9), ("blue", 0.5)], 0.45),
        ([("red", 0.8), ("red", 0.6)], 0.7),
    ],
)
def test_crop_scores_are_summed_over_the_number_of_crops(
    results: list[tuple[str, float]], score: float
) -> None:
    config = LabelConfig(
        views=ViewConfig(max_views=2), min_crop_px=1, min_score=0.0
    )

    labeled = _label_red(ScriptedLabeler(results), config)

    assert labeled.label == "red"
    assert labeled.label_score == pytest.approx(score)


@pytest.mark.parametrize(
    "results",
    [
        [("red", 0.6), ("blue", 0.6)],
        [("blue", 0.6), ("red", 0.6)],
    ],
)
def test_a_fusion_tie_goes_to_the_earlier_vocabulary_entry(
    results: list[tuple[str, float]],
) -> None:
    config = LabelConfig(
        views=ViewConfig(max_views=2), min_crop_px=1, min_score=0.0
    )

    labeled = _label_red(ScriptedLabeler(results), config)

    assert labeled.label == "blue"


def test_a_fused_score_below_the_minimum_gives_neither_label_nor_score() -> (
    None
):
    labeler = ScriptedLabeler([("red", 0.9), ("blue", 0.5)])
    config = LabelConfig(
        views=ViewConfig(max_views=2), min_crop_px=1, min_score=0.5
    )

    labeled = _label_red(labeler, config)

    assert labeled.label is None
    assert labeled.label_score is None


def test_a_zero_score_never_gives_a_label() -> None:
    config = LabelConfig(
        views=ViewConfig(max_views=1), min_crop_px=1, min_score=0.0
    )

    labeled = _label_red(ScriptedLabeler([("red", 0.0)]), config)

    assert labeled.label is None
    assert labeled.label_score is None


@pytest.mark.parametrize("vocabulary", [[], ["red", "red"], ["red", ""]])
def test_an_invalid_vocabulary_is_rejected_even_without_crops(
    vocabulary: list[str],
) -> None:
    scene = _scene()
    far = instance(
        scene.object_points(0) + [50.0, 0.0, 0.0], "far", Epoch.BEFORE
    )

    for items in ((far,), ()):
        with pytest.raises(ValueError, match="vocabulary"):
            label_instances(
                _reconstruction(scene),
                items,
                ScriptedLabeler(),
                vocabulary,
                SMALL,
            )


def test_a_label_outside_the_vocabulary_is_an_error() -> None:
    with pytest.raises(ValueError, match="nope"):
        _label_red(ScriptedLabeler([("nope", 0.5)] * 100), SMALL)


def test_a_fused_score_at_the_minimum_keeps_the_label() -> None:
    config = LabelConfig(
        views=ViewConfig(max_views=1), min_crop_px=1, min_score=0.5
    )

    labeled = _label_red(ScriptedLabeler([("red", 0.5)]), config)

    assert labeled.label == "red"


def test_an_object_no_frame_sees_gets_neither_label_nor_score() -> None:
    scene = _scene()
    far = instance(
        scene.object_points(0) + [50.0, 0.0, 0.0], "far", Epoch.BEFORE
    )
    labeler = ScriptedLabeler([("red", 0.9)] * 100)

    labeled = label_instances(
        _reconstruction(scene),
        (far, _red(scene)),
        labeler,
        VOCABULARY,
        LabelConfig(views=ViewConfig(max_views=1), min_crop_px=1),
    )

    assert labeled[0].label is None
    assert labeled[0].label_score is None
    assert labeled[1].label == "red"
    assert len(labeler.calls[0][0]) == 1


def test_few_visible_points_give_no_view() -> None:
    labeler = ScriptedLabeler([("red", 0.9)] * 100)

    labeled = _label_red(
        labeler,
        LabelConfig(views=ViewConfig(min_visible_points=10**6), min_crop_px=1),
    )

    assert labeled.label is None
    assert labeler.calls == []


def test_no_instances_give_no_labels_and_no_call() -> None:
    labeler = ScriptedLabeler()

    assert label_instances(_reconstruction(_scene()), (), labeler) == ()
    assert labeler.calls == []


def test_the_default_vocabulary_is_the_indoor_objects() -> None:
    scene = _scene()
    labeler = ScriptedLabeler([("table", 0.9)] * 100)

    label_instances(
        _reconstruction(scene), (_red(scene),), labeler, config=SMALL
    )

    assert labeler.calls[0][1] == list(INDOOR_OBJECTS)


def test_an_instance_of_another_epoch_is_rejected() -> None:
    scene = _scene()
    other = instance(scene.object_points(0), "late", Epoch.AFTER)

    with pytest.raises(ValueError, match="epoch"):
        label_instances(_reconstruction(scene), (other,), ScriptedLabeler())


def test_the_epoch_of_the_instance_is_kept() -> None:
    after = _scene(Epoch.AFTER)

    labeled = _label_red(ScriptedLabeler([("red", 0.9)] * 100), SMALL, after)

    assert labeled.epoch == Epoch.AFTER
    assert labeled.label == "red"


@pytest.mark.parametrize(("min_score", "label"), [(0.5, None), (0.0, "red")])
def test_label_point_sets_applies_the_given_minimum_score(
    min_score: float, label: str | None
) -> None:
    scene = _scene()
    config = LabelConfig(
        views=ViewConfig(max_views=1), min_crop_px=1, min_score=0.9
    )

    ((found, _),) = label_point_sets(
        usable_frames(_reconstruction(scene), 0.1),
        [scene.object_points(0)],
        ScriptedLabeler([("red", 0.4)]),
        VOCABULARY,
        config,
        min_score,
    )

    assert found == label
