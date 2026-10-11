import dataclasses
import math

import numpy as np
import pytest
from appearance_scenes import reconstruction
from numpy.typing import NDArray

from spatial_change.change import (
    AppearanceConfig,
    CrossAppearance,
    PointAppearance,
    compare_appearance,
    fuse_appearance,
)
from spatial_change.domain import (
    SE3,
    Epoch,
    EpochReconstruction,
    Frame,
    Intrinsics,
)
from spatial_change.geometry.projection import DEFAULT_MIN_CONFIDENCE
from spatial_change.perception import FakeFeatureEncoder
from spatial_change.visibility import CrossEvidence, PointEvidence, PointState

WIDTH, HEIGHT = 64, 48
FOCAL = 64.0
INTRINSICS = Intrinsics(
    FOCAL, FOCAL, (WIDTH - 1) / 2, (HEIGHT - 1) / 2, WIDTH, HEIGHT
)
WALL_Z = 3.0
GRAY = (120, 120, 120)
RED = (255, 0, 0)
SQUARE_HALF = 0.45
CAMERA_X = (0.0, 0.05, 0.1, 0.15)
CONFIG = AppearanceConfig(changed_below=0.8)
PATCH = 4
EDGE_POINT = (1.62, 0.025, WALL_Z)


def render(camera_x: float, red_square: bool) -> NDArray[np.uint8]:
    """Paints the wall in world coordinates as seen from ``camera_x``."""
    x = (np.arange(WIDTH) - INTRINSICS.cx) * WALL_Z / FOCAL + camera_x
    y = (np.arange(HEIGHT) - INTRINSICS.cy) * WALL_Z / FOCAL
    xx, yy = np.meshgrid(x, y)
    image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = GRAY
    if red_square:
        image[(abs(xx) <= SQUARE_HALF) & (abs(yy) <= SQUARE_HALF)] = RED
    return image


def make_frame(
    epoch: Epoch,
    index: int,
    camera_x: float,
    red_square: bool,
    image: bool = True,
    depth: bool = True,
) -> Frame:
    return Frame(
        epoch=epoch,
        index=index,
        timestamp_s=0.0,
        intrinsics=INTRINSICS,
        world_from_camera=SE3(np.eye(3), np.array([camera_x, 0.0, 0.0])),
        image=render(camera_x, red_square) if image else None,
        depth=np.full((HEIGHT, WIDTH), WALL_Z, np.float32) if depth else None,
        confidence=np.ones((HEIGHT, WIDTH), np.float32),
    )


def wall_points() -> NDArray[np.float64]:
    """Wall points on a 5 cm grid plus one point seen by a single frame."""
    xs = (np.arange(-20, 20) + 0.5) * 0.05
    ys = (np.arange(-12, 12) + 0.5) * 0.05
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    wall = np.column_stack([grid, np.full(len(grid), WALL_Z)])
    return np.vstack([wall, np.array([EDGE_POINT])])


def scene(
    before_frames: list[Frame] | None = None,
    after_frames: list[Frame] | None = None,
) -> tuple[EpochReconstruction, EpochReconstruction]:
    """A wall with a red square in BEFORE that is gone in AFTER."""
    before_frames = before_frames or [
        make_frame(Epoch.BEFORE, i, x, True) for i, x in enumerate(CAMERA_X)
    ]
    after_frames = after_frames or plain_after_frames()
    points = wall_points()
    return (
        reconstruction(Epoch.BEFORE, before_frames, points),
        reconstruction(Epoch.AFTER, after_frames, points),
    )


def plain_after_frames() -> list[Frame]:
    """AFTER frames of the bare wall, one per camera position."""
    return [
        make_frame(Epoch.AFTER, i, x, False) for i, x in enumerate(CAMERA_X)
    ]


def row_of(x: float) -> int:
    """Index of the wall point at ``(x, 0.025)``."""
    near = np.isclose(wall_points()[:, :2], (x, 0.025)).all(axis=1)
    return int(np.flatnonzero(near)[0])


def point_masks() -> dict[str, NDArray[np.bool_]]:
    points = wall_points()
    x, y = np.abs(points[:, 0]), np.abs(points[:, 1])
    edge = np.zeros(len(points), bool)
    edge[-1] = True
    return {
        "inside": (x <= 0.15) & (y <= 0.15) & ~edge,
        "outside": (x >= 0.7) & ~edge,
        "edge": edge,
    }


def test_config_defaults() -> None:
    config = AppearanceConfig()

    assert config.voxel_size == 0.05
    assert config.abs_tolerance == 0.05
    assert config.rel_tolerance == 0.02
    assert config.min_confidence == DEFAULT_MIN_CONFIDENCE
    assert config.min_views == 2
    assert config.changed_below == 0.4


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        AppearanceConfig().min_views = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"voxel_size": 0.0},
        {"voxel_size": math.inf},
        {"abs_tolerance": 0.0},
        {"abs_tolerance": math.nan},
        {"rel_tolerance": -0.1},
        {"rel_tolerance": math.inf},
        {"min_confidence": -0.1},
        {"min_confidence": 1.1},
        {"min_views": 0},
        {"changed_below": -1.5},
        {"changed_below": 1.5},
        {"changed_below": math.nan},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AppearanceConfig(**overrides)  # type: ignore[arg-type]


def test_a_vanished_patch_has_low_similarity_on_both_sides() -> None:
    before, after = scene()

    appearance = compare_appearance(
        before, after, FakeFeatureEncoder(PATCH), CONFIG
    )

    masks = point_masks()
    for side in (appearance.before, appearance.after):
        assert side.similarity.dtype == np.float32
        assert side.own_views.dtype == np.int32
        assert side.other_views.dtype == np.int32
        assert side.similarity[masks["inside"]].max() < 0.7
        assert side.similarity[masks["outside"]].min() > 0.99
        np.testing.assert_array_equal(side.own_views[masks["inside"]], 4)
        np.testing.assert_array_equal(side.other_views[masks["inside"]], 4)


def test_points_under_min_views_get_nan() -> None:
    before, after = scene()

    appearance = compare_appearance(
        before, after, FakeFeatureEncoder(PATCH), CONFIG
    )

    edge = point_masks()["edge"]
    for side in (appearance.before, appearance.after):
        assert np.isnan(side.similarity[edge]).all()
        np.testing.assert_array_equal(side.own_views[edge], 1)
        np.testing.assert_array_equal(side.other_views[edge], 1)
        assert not np.isnan(side.similarity[~edge]).any()


def test_a_lower_min_views_lets_a_single_view_decide() -> None:
    before, after = scene()

    appearance = compare_appearance(
        before,
        after,
        FakeFeatureEncoder(PATCH),
        dataclasses.replace(CONFIG, min_views=1),
    )

    assert appearance.before.similarity[-1] > 0.99


def test_a_point_behind_a_nearer_surface_gets_no_view_from_that_frame() -> None:
    after_frames = plain_after_frames()
    assert after_frames[0].depth is not None
    after_frames[0].depth[:, : WIDTH // 2] = 1.5
    before, after = scene(after_frames=after_frames)

    appearance = compare_appearance(
        before, after, FakeFeatureEncoder(PATCH), CONFIG
    )

    left, right = row_of(-0.475), row_of(0.525)
    assert appearance.before.other_views[left] == 3
    assert appearance.before.other_views[right] == 4
    assert appearance.after.own_views[left] == 3
    assert appearance.before.own_views[left] == 4


def test_low_confidence_pixels_give_no_view() -> None:
    after_frames = plain_after_frames()
    assert after_frames[1].confidence is not None
    after_frames[1].confidence[:, WIDTH // 2 :] = 0.0
    before, after = scene(after_frames=after_frames)

    appearance = compare_appearance(
        before, after, FakeFeatureEncoder(PATCH), CONFIG
    )

    assert appearance.before.other_views[row_of(0.525)] == 3


def test_every_frame_is_encoded_once_and_frames_without_data_are_skipped() -> (
    None
):
    before_frames = [
        make_frame(Epoch.BEFORE, i, x, True) for i, x in enumerate(CAMERA_X)
    ]
    before_frames.append(make_frame(Epoch.BEFORE, 4, 0.0, True, image=False))
    before_frames.append(make_frame(Epoch.BEFORE, 5, 0.0, True, depth=False))
    before, after = scene(before_frames=before_frames)
    encoder = FakeFeatureEncoder(PATCH)

    compare_appearance(before, after, encoder, CONFIG)

    assert encoder.encode_count == 8


def test_swapped_epochs_are_rejected() -> None:
    before, after = scene()
    encoder = FakeFeatureEncoder(PATCH)

    with pytest.raises(ValueError, match="before"):
        compare_appearance(after, before, encoder, CONFIG)
    with pytest.raises(ValueError, match="after"):
        compare_appearance(before, before, encoder, CONFIG)


def test_empty_point_sets_give_empty_appearance() -> None:
    empty = {
        "points": np.zeros((0, 3)),
        "colors": np.zeros((0, 3), np.uint8),
        "confidence": np.zeros(0, np.float32),
    }
    before = EpochReconstruction(Epoch.BEFORE, [], **empty)
    after = EpochReconstruction(Epoch.AFTER, [], **empty)

    appearance = compare_appearance(
        before, after, FakeFeatureEncoder(PATCH), CONFIG
    )

    for side in (appearance.before, appearance.after):
        assert side.similarity.shape == (0,)
        assert side.similarity.dtype == np.float32
        assert side.own_views.shape == side.other_views.shape == (0,)


def evidence(states: list[PointState]) -> PointEvidence:
    count = len(states)
    return PointEvidence(
        counts=np.arange(count * 4, dtype=np.int32).reshape(count, 4),
        states=np.array(states, np.int8),
    )


def cross(states: list[PointState]) -> CrossEvidence:
    return CrossEvidence(evidence(states), evidence(states))


def appearance_of(similarity: list[float]) -> PointAppearance:
    count = len(similarity)
    return PointAppearance(
        similarity=np.array(similarity, np.float32),
        own_views=np.full(count, 3, np.int32),
        other_views=np.full(count, 3, np.int32),
    )


def test_fusion_changes_only_present_points_with_low_similarity() -> None:
    states = [
        PointState.PRESENT,
        PointState.PRESENT,
        PointState.PRESENT,
        PointState.PRESENT,
        PointState.CHANGED,
        PointState.UNKNOWN,
    ]
    similarity = [0.1, math.nan, 0.4, 0.9, 0.1, 0.1]
    votes = cross(states)
    appearance = CrossAppearance(
        appearance_of(similarity), appearance_of(similarity)
    )

    fused = fuse_appearance(votes, appearance, AppearanceConfig())

    expected = np.array(
        [
            PointState.CHANGED,
            PointState.PRESENT,
            PointState.PRESENT,
            PointState.PRESENT,
            PointState.CHANGED,
            PointState.UNKNOWN,
        ],
        np.int8,
    )
    for side in (fused.before, fused.after):
        assert side.states.dtype == np.int8
        np.testing.assert_array_equal(side.states, expected)
        np.testing.assert_array_equal(side.counts, votes.before.counts)
    np.testing.assert_array_equal(
        votes.before.states, np.array(states, np.int8)
    )


def test_fusion_rejects_mismatched_lengths() -> None:
    states = [PointState.PRESENT, PointState.PRESENT]
    votes = cross(states)
    good = appearance_of([0.9, 0.9])
    short = appearance_of([0.9])

    with pytest.raises(ValueError, match="before"):
        fuse_appearance(votes, CrossAppearance(short, good), CONFIG)
    with pytest.raises(ValueError, match="after"):
        fuse_appearance(votes, CrossAppearance(good, short), CONFIG)


def test_fusion_of_the_wall_scene_marks_the_square_changed() -> None:
    before, after = scene()
    appearance = compare_appearance(
        before, after, FakeFeatureEncoder(PATCH), CONFIG
    )
    count = len(before.points)

    fused = fuse_appearance(
        cross([PointState.PRESENT] * count), appearance, CONFIG
    )

    masks = point_masks()
    for side in (fused.before, fused.after):
        np.testing.assert_array_equal(
            side.states[masks["inside"]], PointState.CHANGED
        )
        np.testing.assert_array_equal(
            side.states[masks["outside"]], PointState.PRESENT
        )
        np.testing.assert_array_equal(
            side.states[masks["edge"]], PointState.PRESENT
        )
