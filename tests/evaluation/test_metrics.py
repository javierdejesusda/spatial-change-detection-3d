from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.evaluation.metrics import (
    EvalConfig,
    GtItem,
    PredItem,
    SceneEval,
    evaluate,
    voc_ap,
)

SIZE = 12


def box(top: int, left: int, h: int = 4, w: int = 4) -> NDArray[np.bool_]:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top : top + h, left : left + w] = True
    return mask


def gt(obj_id: int, **frames: NDArray[np.bool_]) -> GtItem:
    masks = {_key(k): m for k, m in frames.items()}
    videos = {video for video, _ in masks}
    return GtItem(obj_id, masks, moved=videos == {1, 2})


def pred(
    obj_id: int, scored: dict[str, tuple[NDArray[np.bool_], float]]
) -> PredItem:
    return PredItem(
        obj_id,
        masks={_key(k): m for k, (m, _) in scored.items()},
        scores={_key(k): s for k, (_, s) in scored.items()},
    )


def _key(name: str) -> tuple[int, int]:
    video, frame = name.removeprefix("v").split("f")
    return int(video), int(frame)


def scene(
    gts: list[GtItem], preds: list[PredItem], name: str = "s"
) -> SceneEval:
    return SceneEval(name, SIZE, SIZE, gts, preds)


def test_voc_ap_hand_values() -> None:
    assert voc_ap(np.array([]), np.array([])) == 0.0
    assert voc_ap(np.array([1.0]), np.array([1.0])) == 1.0
    ap = voc_ap(np.array([0.0, 1.0]), np.array([0.0, 0.5]))
    assert ap == pytest.approx(0.5)


def test_perfect_prediction_scores_one_everywhere() -> None:
    a, b = box(0, 0), box(6, 6)
    result = evaluate(
        [
            scene(
                [gt(0, v1f0=a, v2f0=b)],
                [pred(0, {"v1f0": (a, 0.9), "v2f0": (b, 0.7)})],
            )
        ]
    )

    assert result.pixel_iou == 1.0
    assert result.ap_per_image == 1.0
    assert result.ap_per_scene == 1.0
    assert result.ap_per_scene_typed == 1.0
    assert (result.tp, result.fp, result.fn) == (1, 0, 0)


def test_empty_prediction_scores_zero() -> None:
    result = evaluate([scene([gt(0, v1f0=box(0, 0))], [])])

    assert result.pixel_iou == 0.0
    assert result.ap_per_image == 0.0
    assert result.ap_per_scene == 0.0
    assert result.ap_per_scene_typed == 0.0
    assert (result.tp, result.fp, result.fn) == (0, 0, 1)


def test_empty_ground_truth_with_a_prediction_scores_zero() -> None:
    result = evaluate([scene([], [pred(0, {"v1f0": (box(0, 0), 0.9)})])])

    assert result.pixel_iou == 0.0
    assert result.ap_per_image == 0.0
    assert result.ap_per_scene == 0.0
    assert (result.tp, result.fp, result.fn) == (0, 1, 0)


def test_nothing_anywhere_scores_zero_without_dividing_by_zero() -> None:
    result = evaluate([scene([], [])])

    assert result.pixel_iou == 0.0
    assert result.ap_per_image == 0.0
    assert result.ap_per_scene == 0.0


def test_false_positive_ranked_below_the_hit_keeps_ap_at_one() -> None:
    a = box(0, 0)
    result = evaluate(
        [
            scene(
                [gt(0, v1f0=a)],
                [
                    pred(0, {"v1f0": (a, 0.9)}),
                    pred(1, {"v1f0": (box(7, 7), 0.8)}),
                ],
            )
        ]
    )

    assert result.ap_per_image == pytest.approx(1.0)
    assert result.ap_per_scene == pytest.approx(1.0)
    assert (result.tp, result.fp, result.fn) == (1, 1, 0)


def test_false_positive_ranked_above_the_hit_halves_ap() -> None:
    a = box(0, 0)
    result = evaluate(
        [
            scene(
                [gt(0, v1f0=a)],
                [
                    pred(0, {"v1f0": (a, 0.9)}),
                    pred(1, {"v1f0": (box(7, 7), 0.95)}),
                ],
            )
        ]
    )

    assert result.ap_per_image == pytest.approx(0.5)
    assert result.ap_per_scene == pytest.approx(0.5)


def test_iou_exactly_at_the_threshold_is_not_a_match() -> None:
    result = evaluate(
        [
            scene(
                [gt(0, v1f0=box(0, 0, 2, 2))],
                [pred(0, {"v1f0": (box(0, 0, 1, 2), 0.9)})],
            )
        ]
    )

    assert result.ap_per_image == 0.0
    assert result.ap_per_scene == 0.0


def test_duplicate_detection_is_a_false_positive_by_default() -> None:
    a, b = box(0, 0), box(7, 7)
    scenes = [
        scene(
            [gt(0, v1f0=a), gt(1, v1f0=b)],
            [
                pred(0, {"v1f0": (a, 0.9)}),
                pred(1, {"v1f0": (a, 0.8)}),
                pred(2, {"v1f0": (b, 0.7)}),
            ],
        )
    ]

    result = evaluate(scenes)

    expected = 0.5 * 1.0 + 0.5 * (2 / 3)
    assert result.ap_per_image == pytest.approx(expected)
    assert result.ap_per_scene == pytest.approx(expected)


def test_duplicate_thresholds_forgive_repeat_matches() -> None:
    a, b = box(0, 0), box(7, 7)
    scenes = [
        scene(
            [gt(0, v1f0=a), gt(1, v1f0=b)],
            [
                pred(0, {"v1f0": (a, 0.9)}),
                pred(1, {"v1f0": (a, 0.8)}),
                pred(2, {"v1f0": (b, 0.7)}),
            ],
        )
    ]

    result = evaluate(
        scenes,
        EvalConfig(
            duplicate_match_threshold=2, per_frame_duplicate_match_threshold=2
        ),
    )

    assert result.ap_per_image == pytest.approx(1.0)
    assert result.ap_per_scene == pytest.approx(1.0)


def test_wrong_type_only_hurts_typed_ap() -> None:
    a = box(0, 0)
    far = box(8, 8, 1, 1)
    scenes = [
        scene(
            [gt(0, v1f0=a)],
            [pred(0, {"v1f0": (a, 0.9), "v2f0": (far, 0.9)})],
        )
    ]

    result = evaluate(scenes)

    assert result.ap_per_scene == pytest.approx(1.0)
    assert result.ap_per_scene_typed == 0.0


def test_moved_prediction_matches_moved_ground_truth_when_typed() -> None:
    a, b = box(0, 0), box(6, 6)
    scenes = [
        scene(
            [gt(0, v1f0=a, v2f0=b)],
            [pred(0, {"v1f0": (a, 0.9), "v2f0": (b, 0.9)})],
        )
    ]

    assert evaluate(scenes).ap_per_scene_typed == pytest.approx(1.0)


def test_added_and_removed_are_one_type_for_typed_ap() -> None:
    a, b = box(0, 0), box(6, 6)
    scenes = [
        scene(
            [gt(0, v1f0=a), gt(1, v2f0=b)],
            [
                pred(0, {"v1f0": (a, 0.9)}),
                pred(1, {"v2f0": (b, 0.8)}),
            ],
        )
    ]

    assert evaluate(scenes).ap_per_scene_typed == pytest.approx(1.0)


def test_stacked_object_iou_hand_value() -> None:
    a, b = box(0, 0), box(6, 6)
    half = box(6, 6, 2, 4)
    scenes = [
        scene(
            [gt(0, v1f0=a, v2f0=b)],
            [pred(0, {"v1f0": (a, 0.9), "v2f0": (half, 0.9)})],
        )
    ]

    # (16 + 8) / (16 + 16) = 0.75 > 0.5 matches at object level.
    assert evaluate(scenes).ap_per_scene == pytest.approx(1.0)


def test_stacked_object_iou_below_threshold_does_not_match() -> None:
    a, b = box(0, 0), box(6, 6)
    scenes = [
        scene(
            [gt(0, v1f0=a, v2f0=b)],
            [pred(0, {"v1f0": (a, 0.9)})],
        )
    ]

    # 16 / 32 = 0.5 is not above the threshold.
    assert evaluate(scenes).ap_per_scene == 0.0


def test_object_confidence_is_the_mean_of_frame_scores() -> None:
    a, c = box(0, 0), box(7, 7)
    low_high = pred(0, {"v1f0": (a, 0.2), "v1f1": (a, 0.8)})
    steady = pred(1, {"v1f0": (c, 0.6)})

    assert low_high.confidence == pytest.approx(0.5)
    scenes = [
        scene(
            [gt(0, v1f0=a, v1f1=a)],
            [low_high, steady],
        )
    ]

    # The 0.6 false positive outranks the 0.5 hit at object level.
    assert evaluate(scenes).ap_per_scene == pytest.approx(0.5)


def test_pixel_iou_hand_value() -> None:
    result = evaluate(
        [
            scene(
                [gt(0, v1f0=box(0, 0))],
                [pred(0, {"v1f0": (box(0, 2), 0.9)})],
            )
        ]
    )

    # intersection 8, union 24.
    assert result.pixel_iou == pytest.approx(1 / 3)


def test_pixel_iou_merges_objects_and_pools_frames() -> None:
    a, b = box(0, 0), box(6, 6)
    result = evaluate(
        [
            scene(
                [gt(0, v1f0=a), gt(1, v1f0=b), gt(2, v2f3=a)],
                [
                    pred(0, {"v1f0": (a, 0.9)}),
                    pred(1, {"v1f0": (box(6, 8), 0.9)}),
                ],
            )
        ]
    )

    # frame v1f0: tp 16 + 8, fp 8, fn 8; frame v2f3: fn 16.
    assert result.pixel_iou == pytest.approx(24 / (24 + 8 + 24))


def test_scenes_pool_into_one_ranking() -> None:
    a = box(0, 0)
    hit = scene([gt(0, v1f0=a)], [pred(0, {"v1f0": (a, 0.9)})], "hit")
    empty = scene([], [pred(0, {"v1f0": (a, 0.95)})], "empty")

    result = evaluate([hit, empty])

    assert result.ap_per_image == pytest.approx(0.5)
    assert result.ap_per_scene == pytest.approx(0.5)


def test_prediction_without_masks_is_ignored() -> None:
    a = box(0, 0)
    scenes = [
        scene(
            [gt(0, v1f0=a)],
            [PredItem(5, {}, {}), pred(0, {"v1f0": (a, 0.9)})],
        )
    ]

    result = evaluate(scenes)

    assert result.ap_per_scene == pytest.approx(1.0)
    assert result.fp == 0


def test_ground_truth_without_frames_counts_but_never_matches() -> None:
    a = box(0, 0)
    scenes = [
        scene(
            [gt(0, v1f0=a), GtItem(1, {})],
            [pred(0, {"v1f0": (a, 0.9)})],
        )
    ]

    result = evaluate(scenes)

    assert result.ap_per_scene == pytest.approx(0.5)
    assert result.fn == 1


def test_mask_of_the_wrong_size_is_rejected() -> None:
    bad = np.zeros((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="shape"):
        evaluate([scene([], [pred(0, {"v1f0": (bad, 0.9)})])])


def test_scores_must_match_masks() -> None:
    with pytest.raises(ValueError, match="score"):
        PredItem(0, {(1, 0): box(0, 0)}, {})


def test_video_index_must_be_one_or_two() -> None:
    with pytest.raises(ValueError, match="video"):
        GtItem(0, {(3, 0): box(0, 0)})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iou_threshold": 1.0},
        {"iou_threshold": -0.1},
        {"duplicate_match_threshold": 0},
        {"per_frame_duplicate_match_threshold": 0},
    ],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        EvalConfig(**kwargs)  # type: ignore[arg-type]


def test_scenes_can_be_streamed_one_at_a_time() -> None:
    a, b = box(0, 0), box(6, 6)
    scenes = [
        scene(
            [gt(0, v1f0=a, v2f0=b)],
            [pred(0, {"v1f0": (a, 0.9), "v2f0": (b, 0.7)})],
            name="first",
        ),
        scene([gt(1, v1f0=a)], [pred(1, {"v1f0": (b, 0.4)})], name="second"),
    ]

    def stream() -> Iterator[SceneEval]:
        yield from scenes

    assert evaluate(stream()) == evaluate(scenes)
