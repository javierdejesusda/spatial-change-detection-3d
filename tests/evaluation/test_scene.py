import numpy as np
import pytest
from eval_pairs import (
    own_segments,
    with_after_window,
    with_metadata,
    with_uncertain_move,
)
from rle_helpers import encode_mask

from spatial_change.change import ChangeSources
from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeType,
    Epoch,
    EvidenceCounts,
    ObjectInstance,
    OrientedBox,
)
from spatial_change.evaluation import (
    PairAnnotations,
    eval_size,
    evaluate,
    official_eval_hw,
    parse_pair_annotations,
)
from spatial_change.evaluation.metrics import PredItem
from spatial_change.evaluation.predictions import Predictions
from spatial_change.evaluation.scene import (
    PairEvalConfig,
    build_scene,
    change_objects,
    eval_grid,
    project_run,
    run_pred_items,
    scored_items,
    uncertain_ids,
)
from spatial_change.pipeline import PipelineResult
from spatial_change.pipeline.videos import ImageWindow

EVIDENCE = EvidenceCounts(consistent=1, free=2, occluded=0, unobserved=0)


def box_at(x: float) -> OrientedBox:
    return OrientedBox(np.array([x, 0.0, 0.0]), np.ones(3), np.eye(3))


def instance(epoch: Epoch, x: float) -> ObjectInstance:
    points = np.array([[x, 0.0, 0.0], [x + 0.1, 0.0, 0.0]])
    return ObjectInstance(f"{epoch.value}-{x}", epoch, points, box_at(x))


def change(kind: ChangeType, confidence: float = 0.5) -> Change:
    moved = kind in (ChangeType.MOVED, ChangeType.UNCERTAIN)
    has_before = kind is not ChangeType.ADDED
    has_after = kind is not ChangeType.REMOVED
    return Change(
        change_id=f"change-{kind.value}",
        type=kind,
        label="object",
        description="",
        confidence=confidence,
        before_box=BoxModel.from_box(box_at(0.0)) if has_before else None,
        after_box=BoxModel.from_box(box_at(1.0)) if has_after else None,
        displacement_m=1.0 if moved else None,
        rotation_deg=0.0 if kind is ChangeType.MOVED else None,
        evidence=EVIDENCE,
    )


def sources(kind: ChangeType) -> ChangeSources:
    return ChangeSources(
        instance(Epoch.BEFORE, 0.0) if kind is not ChangeType.ADDED else None,
        instance(Epoch.AFTER, 1.0) if kind is not ChangeType.REMOVED else None,
    )


def sources_of(changes: list[Change]) -> dict[str, ChangeSources]:
    return {c.change_id: sources(c.type) for c in changes}


def test_each_type_is_scored_in_the_epochs_it_is_present_in() -> None:
    kinds = [ChangeType.ADDED, ChangeType.REMOVED, ChangeType.MOVED]
    changes = [change(k, 0.2 + i / 10) for i, k in enumerate(kinds)]
    found = sources_of(changes)

    objects = change_objects(changes, found)

    assert [o.change_type for o in objects] == kinds
    assert [set(o.points) for o in objects] == [
        {Epoch.AFTER},
        {Epoch.BEFORE},
        {Epoch.BEFORE, Epoch.AFTER},
    ]
    assert [o.obj_id for o in objects] == [0, 1, 2]
    assert [o.score for o in objects] == pytest.approx([0.2, 0.3, 0.4])
    moved = objects[2]
    before, after = found["change-moved"].before, found["change-moved"].after
    assert before is not None and after is not None
    np.testing.assert_array_equal(moved.points[Epoch.BEFORE], before.points)
    np.testing.assert_array_equal(moved.points[Epoch.AFTER], after.points)


def test_an_uncertain_change_is_a_moved_object_at_its_own_confidence() -> None:
    changes = [change(ChangeType.UNCERTAIN, 0.2), change(ChangeType.ADDED, 0.9)]
    found = sources_of(changes)

    objects = change_objects(changes, found)

    assert [(o.obj_id, o.change_type, o.score) for o in objects] == [
        (0, ChangeType.MOVED, 0.2),
        (1, ChangeType.ADDED, 0.9),
    ]
    before = found["change-uncertain"].before
    after = found["change-uncertain"].after
    assert before is not None and after is not None
    np.testing.assert_array_equal(
        objects[0].points[Epoch.BEFORE], before.points
    )
    np.testing.assert_array_equal(objects[0].points[Epoch.AFTER], after.points)


def test_a_change_without_sources_is_an_error() -> None:
    with pytest.raises(ValueError, match="change-added"):
        change_objects([change(ChangeType.ADDED)], {})


@pytest.mark.parametrize(
    ("kind", "wrong"),
    [
        (ChangeType.ADDED, sources(ChangeType.REMOVED)),
        (ChangeType.ADDED, sources(ChangeType.MOVED)),
        (ChangeType.REMOVED, sources(ChangeType.ADDED)),
        (ChangeType.REMOVED, sources(ChangeType.MOVED)),
        (ChangeType.MOVED, sources(ChangeType.ADDED)),
        (ChangeType.MOVED, sources(ChangeType.REMOVED)),
        (ChangeType.UNCERTAIN, sources(ChangeType.ADDED)),
    ],
)
def test_sources_that_disagree_with_the_type_are_an_error(
    kind: ChangeType, wrong: ChangeSources
) -> None:
    with pytest.raises(ValueError, match="change-"):
        change_objects([change(kind)], {f"change-{kind.value}": wrong})


def test_the_pair_eval_config_validates_its_numbers() -> None:
    with pytest.raises(ValueError, match="stride"):
        PairEvalConfig(stride=0)
    with pytest.raises(ValueError, match="max_length"):
        PairEvalConfig(max_length=0)


def test_uncertain_changes_are_dropped_unless_asked_otherwise() -> None:
    assert PairEvalConfig().uncertain_as == "drop"
    assert PairEvalConfig(uncertain_as="moved").uncertain_as == "moved"


@pytest.mark.parametrize(
    "value", ["added", "uncertain", "Moved", "", "none", None, 1]
)
def test_uncertain_can_only_be_dropped_or_read_as_moved(value: object) -> None:
    with pytest.raises(ValueError, match="uncertain_as"):
        PairEvalConfig(uncertain_as=value)  # type: ignore[arg-type]


def annotations_with_a_mask(height: int, width: int) -> PairAnnotations:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    return parse_pair_annotations(
        "p",
        {
            "objects": [
                {"original_obj_idx": 0, "in_video1": True, "in_video2": False}
            ],
            "video1_objects": {0: {"0": encode_mask(mask)}},
            "video2_objects": {},
        },
    )


def test_the_grid_comes_from_the_first_ground_truth_mask() -> None:
    annotations = annotations_with_a_mask(1280, 720)

    grid = eval_grid(annotations, {}, 1024)

    assert grid == eval_size(annotations, 1024) == (576, 1024)


def test_a_pair_without_masks_gets_the_grid_of_its_video() -> None:
    metadata = {"video_before_width": 720, "video_before_height": 1280}

    grid = eval_grid(PairAnnotations("p", ()), metadata, 1024)

    assert grid == official_eval_hw(1280, 720, 1024) == (576, 1024)


def test_a_pair_without_masks_or_video_sizes_has_no_grid() -> None:
    with pytest.raises(ValueError, match="video"):
        eval_grid(PairAnnotations("p", ()), {}, 1024)


SQUARE = PairEvalConfig(stride=1, max_length=32)
MOVED_SQUARE = PairEvalConfig(stride=1, max_length=32, uncertain_as="moved")


def test_only_the_changes_the_run_could_not_classify_are_uncertain() -> None:
    kinds = [ChangeType.ADDED, ChangeType.UNCERTAIN, ChangeType.MOVED]

    assert uncertain_ids([change(k) for k in kinds]) == {1}


def test_an_uncertain_object_is_scored_only_when_read_as_moved() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, 0] = True
    items = [
        PredItem(obj_id, {(1, 0): mask}, {(1, 0): 0.5}) for obj_id in (0, 1, 2)
    ]
    predictions = Predictions(32, 32, items, frozenset({1}))
    annotations = PairAnnotations("p", ())

    dropped = build_scene("p", annotations, predictions, SQUARE)
    moved = build_scene("p", annotations, predictions, MOVED_SQUARE)

    assert [i.obj_id for i in scored_items(items, frozenset({1}), "drop")] == [
        0,
        2,
    ]
    assert [i.obj_id for i in dropped.pred] == [0, 2]
    assert [i.obj_id for i in moved.pred] == [0, 1, 2]
    assert (moved.name, moved.height, moved.width) == ("p", 32, 32)
    assert moved.gt == dropped.gt == []


def test_a_run_scored_against_its_own_masks_is_perfect(
    template_run: PipelineResult,
) -> None:
    annotations = parse_pair_annotations(
        "p", own_segments(template_run, SQUARE)
    )

    scene = build_scene(
        "p", annotations, project_run(template_run, annotations, SQUARE), SQUARE
    )
    metrics = evaluate([scene])

    assert len(scene.gt) == len(scene.pred) > 0
    assert (scene.height, scene.width) == (32, 32)
    assert metrics.ap_per_scene == 1.0
    assert metrics.ap_per_scene_typed == 1.0
    assert metrics.pixel_iou == 1.0
    assert (metrics.tp, metrics.fp, metrics.fn) == (len(scene.gt), 0, 0)


def test_a_run_projects_its_uncertain_changes_flagged_as_moved(
    template_run: PipelineResult,
) -> None:
    uncertain = with_uncertain_move(template_run, confidence=0.2)
    annotations = parse_pair_annotations(
        "p", own_segments(template_run, SQUARE)
    )
    index = next(
        i
        for i, c in enumerate(uncertain.report.changes)
        if c.type is ChangeType.UNCERTAIN
    )

    predictions = project_run(uncertain, annotations, SQUARE)

    assert (predictions.height, predictions.width) == (32, 32)
    assert predictions.uncertain == {index}
    (item,) = [i for i in predictions.items if i.obj_id == index]
    assert list(item.scores.values()) == pytest.approx([0.2] * len(item.scores))
    assert {i.obj_id for i in predictions.items} == {
        i.obj_id for i in run_pred_items(uncertain, SQUARE, 32, 32)
    }
    dropped = build_scene("p", annotations, predictions, SQUARE)
    moved = build_scene("p", annotations, predictions, MOVED_SQUARE)
    assert len(moved.pred) == len(dropped.pred) + 1
    assert evaluate([moved]).tp == len(moved.gt)
    assert evaluate([dropped]).fn == 1


def test_a_cropped_after_window_is_placed_and_still_scores_perfectly(
    template_run: PipelineResult,
) -> None:
    result = with_after_window(
        template_run, ImageWindow(0.0, 0.125, 1.0, 0.875)
    )
    annotations = parse_pair_annotations("p", own_segments(result, SQUARE))

    scene = build_scene(
        "p", annotations, project_run(result, annotations, SQUARE), SQUARE
    )
    metrics = evaluate([scene])

    after = [
        m for item in scene.pred for (v, _), m in item.masks.items() if v == 2
    ]
    assert after
    for mask in after:
        assert not mask[:, :4].any()
        assert not mask[:, 28:].any()
    assert metrics.ap_per_scene == 1.0
    assert metrics.pixel_iou == 1.0
    assert (metrics.fp, metrics.fn) == (0, 0)


def test_a_pair_without_masks_takes_the_grid_of_the_video_of_the_run(
    template_run: PipelineResult,
) -> None:
    result = with_metadata(
        template_run, video_before_width=64, video_before_height=48
    )
    annotations = PairAnnotations("p", ())

    predictions = project_run(result, annotations, SQUARE)

    assert (predictions.height, predictions.width) == official_eval_hw(
        48, 64, 32
    )
