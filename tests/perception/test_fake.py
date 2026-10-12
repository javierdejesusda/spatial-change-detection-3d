import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.perception.fake import (
    FAKE_ENCODER_ID,
    FakeFeatureEncoder,
    FakeLabeler,
    FakeMaskSegmenter,
    FakeVlm,
)
from spatial_change.ports import (
    FeatureEncoder,
    Labeler,
    MaskSegmenter,
    VisionLanguageModel,
)


def _image(height: int, width: int, rgb: tuple[int, int, int]) -> NDArray:
    return np.full((height, width, 3), rgb, np.uint8)


def test_fake_encoder_satisfies_the_port() -> None:
    assert isinstance(FakeFeatureEncoder(), FeatureEncoder)


def test_default_patch_size_is_14() -> None:
    assert FakeFeatureEncoder().patch_size == 14


@pytest.mark.parametrize("patch_size", [0, -1])
def test_invalid_patch_sizes_are_rejected(patch_size: int) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        FakeFeatureEncoder(patch_size)


@pytest.mark.parametrize(
    ("shape", "patch", "grid"),
    [
        ((294, 518), 16, (19, 33)),
        ((294, 518), 14, (21, 37)),
        ((8, 8), 4, (2, 2)),
    ],
)
def test_grid_is_the_padded_image_over_the_patch_size(
    shape: tuple[int, int], patch: int, grid: tuple[int, int]
) -> None:
    features = FakeFeatureEncoder(patch).encode(_image(*shape, (9, 9, 9)))

    assert features.shape == (*grid, 3)
    assert features.dtype == np.float32


def test_feature_is_the_unit_direction_of_the_shifted_mean_color() -> None:
    image = np.zeros((4, 8, 3), np.uint8)
    image[:, :4] = (0, 0, 0)
    image[:, 4:] = (254, 0, 0)

    features = FakeFeatureEncoder(4).encode(image)

    np.testing.assert_allclose(np.linalg.norm(features, axis=2), 1.0, atol=1e-6)
    expected_black = np.ones(3) / np.sqrt(3)
    np.testing.assert_allclose(features[0, 0], expected_black, atol=1e-6)
    red = np.array([255.0, 1.0, 1.0])
    np.testing.assert_allclose(
        features[0, 1], red / np.linalg.norm(red), atol=1e-6
    )


def test_patch_feature_averages_the_colors_of_its_pixels() -> None:
    image = np.zeros((2, 2, 3), np.uint8)
    image[0] = (100, 0, 0)
    image[1] = (0, 100, 0)

    feature = FakeFeatureEncoder(2).encode(image)[0, 0]

    mean = np.array([51.0, 51.0, 1.0])
    np.testing.assert_allclose(feature, mean / np.linalg.norm(mean), atol=1e-6)


def test_edge_patches_are_padded_by_edge_replication() -> None:
    image = np.zeros((3, 3, 3), np.uint8)
    image[:, :] = (200, 10, 10)

    features = FakeFeatureEncoder(2).encode(image)

    np.testing.assert_allclose(features[1, 1], features[0, 0], atol=1e-6)


def test_cosine_ignores_brightness_but_not_hue() -> None:
    encoder = FakeFeatureEncoder(4)
    red = encoder.encode(_image(4, 4, (200, 40, 40)))[0, 0]
    dark_red = encoder.encode(_image(4, 4, (99, 19, 19)))[0, 0]
    blue = encoder.encode(_image(4, 4, (40, 40, 200)))[0, 0]

    assert float(red @ dark_red) > 0.999
    assert float(red @ blue) < 0.5


def test_calls_are_counted() -> None:
    encoder = FakeFeatureEncoder(4)

    encoder.encode(_image(4, 4, (1, 2, 3)))
    encoder.encode(_image(4, 4, (1, 2, 3)))
    encoder.unload()

    assert encoder.encode_count == 2
    assert encoder.unload_count == 1


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), np.uint8),
        np.zeros((4, 4, 4), np.uint8),
        np.zeros((4, 4, 3), np.float32),
        np.zeros((0, 4, 3), np.uint8),
    ],
)
def test_invalid_images_are_rejected(image: NDArray) -> None:
    with pytest.raises(ValueError, match="image"):
        FakeFeatureEncoder(4).encode(image)


def _segment(
    shape: tuple[int, int], boxes: list[list[float]]
) -> NDArray[np.bool_]:
    image = np.zeros((*shape, 3), np.uint8)
    array = np.array(boxes, np.float64).reshape(-1, 4)
    return FakeMaskSegmenter().segment(image, array, None)


def test_fake_segmenter_satisfies_the_port() -> None:
    assert isinstance(FakeMaskSegmenter(), MaskSegmenter)


def test_a_mask_is_the_pixels_whose_centers_lie_in_the_box() -> None:
    masks = _segment((6, 8), [[2.0, 1.0, 4.0, 3.0]])

    assert masks.shape == (1, 6, 8)
    assert masks.dtype == np.bool_
    expected = np.zeros((6, 8), bool)
    expected[1:4, 2:5] = True
    np.testing.assert_array_equal(masks[0], expected)


def test_fractional_boxes_include_only_the_centers_inside() -> None:
    masks = _segment((6, 8), [[1.5, 0.5, 3.4, 2.6]])

    expected = np.zeros((6, 8), bool)
    expected[1:3, 2:4] = True
    np.testing.assert_array_equal(masks[0], expected)


def test_boxes_are_clipped_at_the_image_border() -> None:
    masks = _segment((4, 5), [[-3.0, -2.0, 1.0, 1.0], [3.0, 2.0, 99.0, 99.0]])

    first = np.zeros((4, 5), bool)
    first[:2, :2] = True
    second = np.zeros((4, 5), bool)
    second[2:, 3:] = True
    np.testing.assert_array_equal(masks[0], first)
    np.testing.assert_array_equal(masks[1], second)


def test_a_box_between_pixel_centers_gives_an_empty_mask() -> None:
    assert not _segment((4, 4), [[1.2, 1.2, 1.8, 1.8]]).any()


def test_a_box_outside_the_image_gives_an_empty_mask() -> None:
    assert not _segment((4, 4), [[10.0, 10.0, 12.0, 12.0]]).any()


def test_masks_follow_the_prompt_order() -> None:
    masks = _segment((4, 4), [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]])

    assert masks[0, 0, 0] and masks[0].sum() == 1
    assert masks[1, 3, 3] and masks[1].sum() == 1


def test_no_boxes_give_an_empty_stack() -> None:
    assert _segment((4, 5), []).shape == (0, 4, 5)


def test_segment_and_unload_calls_are_counted() -> None:
    segmenter = FakeMaskSegmenter()
    image = np.zeros((4, 4, 3), np.uint8)
    boxes = np.zeros((1, 4), np.float64)

    segmenter.segment(image, boxes, None)
    segmenter.segment(image, boxes, None)
    segmenter.unload()

    assert len(segmenter.calls) == 2
    assert segmenter.unload_count == 1


def test_the_prompts_of_each_call_are_recorded() -> None:
    segmenter = FakeMaskSegmenter()
    image = np.zeros((4, 4, 3), np.uint8)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
    points = np.array([[0.5, 0.5]])

    segmenter.segment(image, boxes, points)

    ((got_image, got_boxes, got_points),) = segmenter.calls
    assert got_image is image
    np.testing.assert_array_equal(got_boxes, boxes)
    np.testing.assert_array_equal(got_points, points)


def test_the_segmenter_rejects_an_invalid_prompt_and_records_nothing() -> None:
    segmenter = FakeMaskSegmenter()
    image = np.zeros((4, 4, 3), np.uint8)

    with pytest.raises(ValueError, match="boxes"):
        segmenter.segment(image, np.array([[3.0, 0.0, 1.0, 1.0]]), None)

    assert segmenter.calls == []


_COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}


def test_fake_labeler_satisfies_the_port() -> None:
    assert isinstance(FakeLabeler(_COLORS), Labeler)


def test_the_label_is_the_entry_with_the_nearest_known_hue() -> None:
    crops = [_image(4, 6, (230, 20, 10)), _image(3, 3, (10, 20, 240))]

    labels = FakeLabeler(_COLORS).label(crops, ["green", "red", "blue"])

    assert [label for label, _ in labels] == ["red", "blue"]
    assert [score for _, score in labels] == [1.0, 1.0]


def test_brightness_does_not_change_the_label() -> None:
    labeler = FakeLabeler({"red": (200, 40, 40), "blue": (40, 40, 200)})

    dim = labeler.label([_image(2, 2, (50, 10, 10))], ["red", "blue"])
    bright = labeler.label([_image(2, 2, (255, 51, 51))], ["red", "blue"])

    assert dim == [("red", 1.0)]
    assert bright == [("red", 1.0)]


def test_the_score_is_the_share_of_the_matching_pixels_for_the_label() -> None:
    crop = _image(4, 4, (255, 0, 0))
    crop[:1] = (0, 0, 255)
    crop[1:2, :2] = (90, 90, 90)

    ((label, score),) = FakeLabeler(_COLORS).label([crop], ["red", "blue"])

    assert label == "red"
    assert score == pytest.approx(10 / 14)


def test_pixels_of_another_hue_do_not_vote() -> None:
    crop = _image(2, 2, (255, 0, 0))
    crop[0, 0] = (255, 80, 0)

    ((_, score),) = FakeLabeler(_COLORS, min_cosine=0.99).label(
        [crop], ["red", "green"]
    )
    ((_, loose),) = FakeLabeler(_COLORS, min_cosine=0.9).label(
        [crop], ["red", "green"]
    )

    assert score == 1.0
    assert loose == 1.0
    ((_, split),) = FakeLabeler(
        {"red": (255, 0, 0), "orange": (255, 80, 0)}, min_cosine=0.9
    ).label([crop], ["red", "orange"])
    assert split == pytest.approx(0.75)


def test_entries_without_a_known_color_are_not_candidates() -> None:
    labeler = FakeLabeler(_COLORS)

    ((label, score),) = labeler.label(
        [_image(2, 2, (255, 0, 0))], ["table", "red"]
    )

    assert (label, score) == ("red", 1.0)


def test_a_crop_that_matches_nothing_gives_the_first_entry_with_zero() -> None:
    results = FakeLabeler(_COLORS).label(
        [_image(2, 2, (90, 90, 90)), _image(2, 2, (0, 0, 0))],
        ["table", "red"],
    )

    assert results == [("table", 0.0), ("table", 0.0)]


def test_a_vocabulary_without_known_colors_gives_the_first_with_zero() -> None:
    results = FakeLabeler(_COLORS).label(
        [_image(2, 2, (255, 0, 0))] * 2, ["table", "chair"]
    )

    assert results == [("table", 0.0), ("table", 0.0)]


def test_a_hue_tie_goes_to_the_earlier_entry() -> None:
    labeler = FakeLabeler({"a": (10, 0, 0), "b": (20, 0, 0)})

    ((label, _),) = labeler.label([_image(2, 2, (10, 0, 0))], ["b", "a"])

    assert label == "b"


def test_no_crops_give_no_labels() -> None:
    assert FakeLabeler(_COLORS).label([], ["red"]) == []


@pytest.mark.parametrize("min_cosine", [0.0, -0.5, 1.5])
def test_the_minimum_cosine_must_be_in_the_open_unit_interval(
    min_cosine: float,
) -> None:
    with pytest.raises(ValueError, match="min_cosine"):
        FakeLabeler(_COLORS, min_cosine=min_cosine)


def test_labeler_inputs_are_validated() -> None:
    labeler = FakeLabeler(_COLORS)

    with pytest.raises(ValueError, match="image"):
        labeler.label([np.zeros((2, 2), np.uint8)], ["red"])
    with pytest.raises(ValueError, match="vocabulary"):
        labeler.label([_image(2, 2, (1, 1, 1))], [])


def test_labeler_unloads_are_counted() -> None:
    labeler = FakeLabeler(_COLORS)

    labeler.unload()
    labeler.unload()

    assert labeler.unload_count == 2


def test_fake_vlm_satisfies_the_port() -> None:
    assert isinstance(FakeVlm(["x"]), VisionLanguageModel)


def test_fake_vlm_replies_in_order_and_records_the_calls() -> None:
    vlm = FakeVlm(["first", "second"])
    image = _image(4, 5, (1, 2, 3))

    assert vlm.generate([image], "p1") == "first"
    assert vlm.generate([], "p2") == "second"

    assert [prompt for _, prompt in vlm.calls] == ["p1", "p2"]
    assert vlm.calls[0][0][0] is image
    assert vlm.calls[1][0] == ()


def test_fake_vlm_with_no_replies_left_raises() -> None:
    vlm = FakeVlm(["only"])
    vlm.generate([], "p")

    with pytest.raises(RuntimeError, match="no scripted reply"):
        vlm.generate([], "p")


def test_fake_vlm_counts_unloads() -> None:
    vlm = FakeVlm([])

    vlm.unload()
    vlm.unload()

    assert vlm.unload_count == 2


def test_fake_encoder_reports_the_id_it_loaded() -> None:
    assert FakeFeatureEncoder().loaded_model_id == FAKE_ENCODER_ID
