import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from lifting_scenes import (
    INTRINSICS,
    Cuboid,
    HandScene,
    OracleSegmenter,
    candidate_of,
    hand_scene,
    ring_poses,
)
from numpy.typing import NDArray

from spatial_change.change import (
    ChangeCandidates,
    LiftConfig,
    LiftedObjects,
    lift_candidates,
    lift_change_candidates,
)
from spatial_change.domain import Epoch, Frame
from spatial_change.geometry import (
    fit_upright_box,
    usable_pixels,
    visible_points,
)
from spatial_change.geometry.views import ViewConfig

TALL = Cuboid((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0))
LOW = Cuboid((1.2, -0.3, 0.0), (1.6, 0.3, 0.4))
WIDTH, HEIGHT = INTRINSICS.width, INTRINSICS.height
NEAR = 1.0


@functools.cache
def ring_scene() -> HandScene:
    """Six cameras around a tall box and a low box that it can hide."""
    return hand_scene([TALL, LOW], ring_poses(6))


@functools.cache
def twin_scene() -> HandScene:
    """Four cameras whose first two share a pose."""
    first, second, third = ring_poses(3)
    return hand_scene([TALL], [first, first, second, third])


def visible_counts(
    scene: HandScene, points: NDArray[np.float64], config: LiftConfig
) -> list[int]:
    counts = []
    for frame in scene.frames:
        usable = usable_pixels(frame, config.views.min_confidence)
        seen = visible_points(
            points,
            frame,
            usable,
            config.views.abs_tolerance,
            config.views.rel_tolerance,
        )
        counts.append(len(seen.indices))
    return counts


def called_frames(scene: HandScene, segmenter: OracleSegmenter) -> list[int]:
    """Lists the positions of the frames that were segmented, in call order."""
    by_image = {id(f.image): i for i, f in enumerate(scene.frames)}
    return [by_image[id(image)] for image, _, _ in segmenter.calls]


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        LiftConfig().box_margin_px = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"box_margin_px": -1.0},
        {"box_margin_px": float("nan")},
        {"voxel_size": 0.0},
        {"voxel_size": float("inf")},
        {"min_agreement": -0.1},
        {"min_agreement": 1.1},
        {"min_points": 0},
        {"merge_overlap": 0.0},
        {"merge_overlap": 1.1},
        {"link_radius": 0},
        {"floor_clearance": -0.01},
        {"floor_clearance": float("nan")},
        {"floor_clearance": float("inf")},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LiftConfig(**overrides)


def test_views_are_the_frames_that_see_most_candidate_points() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    config = LiftConfig(views=ViewConfig(max_views=3))
    counts = visible_counts(scene, points, config)
    expected = sorted(np.argsort(counts, kind="stable")[::-1][:3].tolist())
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(), (candidate_of(points),), segmenter, config
    )

    assert sorted(called_frames(scene, segmenter)) == expected
    assert 0 in expected


def test_ties_between_frames_go_to_the_earlier_frame() -> None:
    scene = twin_scene()
    points = scene.object_points(0, [0])
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        segmenter,
        LiftConfig(views=ViewConfig(max_views=1)),
    )

    assert called_frames(scene, segmenter) == [0]


def test_frames_seeing_too_few_candidate_points_are_skipped() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    base = LiftConfig(views=ViewConfig(max_views=6))
    counts = visible_counts(scene, points, base)
    threshold = sorted(counts)[-2]
    expected = {i for i, count in enumerate(counts) if count >= threshold}
    assert 0 < len(expected) < 6
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        segmenter,
        dataclasses.replace(
            base,
            views=dataclasses.replace(base.views, min_visible_points=threshold),
        ),
    )

    assert set(called_frames(scene, segmenter)) == expected


def test_a_frame_that_cannot_see_the_candidate_is_not_selected() -> None:
    scene = ring_scene()
    points = scene.object_points(1, [0])
    config = LiftConfig(views=ViewConfig(max_views=6))
    counts = visible_counts(scene, points, config)
    assert counts[3] == 0 and counts[0] > 0
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(), (candidate_of(points),), segmenter, config
    )

    assert 3 not in called_frames(scene, segmenter)
    assert 0 in called_frames(scene, segmenter)


def _prompt_pixels(
    scene: HandScene, points: NDArray[np.float64], config: LiftConfig
) -> NDArray[np.intp]:
    frame = scene.frames[0]
    usable = usable_pixels(frame, config.views.min_confidence)
    seen = visible_points(
        points,
        frame,
        usable,
        config.views.abs_tolerance,
        config.views.rel_tolerance,
    )
    return seen.pixels


@pytest.mark.parametrize("margin", [0.0, 4.0, 2.5])
def test_the_box_is_the_visible_pixels_grown_by_the_margin(
    margin: float,
) -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    config = LiftConfig(views=ViewConfig(max_views=1), box_margin_px=margin)
    pixels = _prompt_pixels(scene, points, config)
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(), (candidate_of(points),), segmenter, config
    )

    (_, boxes, _) = segmenter.calls[0]
    expected = [
        pixels[:, 0].min() - margin,
        pixels[:, 1].min() - margin,
        pixels[:, 0].max() + margin,
        pixels[:, 1].max() + margin,
    ]
    np.testing.assert_allclose(boxes, [expected])
    assert boxes.dtype == np.float64


def test_the_box_is_clipped_to_the_image() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        segmenter,
        LiftConfig(views=ViewConfig(max_views=1), box_margin_px=1000.0),
    )

    (_, boxes, _) = segmenter.calls[0]
    np.testing.assert_array_equal(boxes, [[0, 0, WIDTH - 1, HEIGHT - 1]])


def test_the_point_is_the_visible_pixel_nearest_to_their_median() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    config = LiftConfig(views=ViewConfig(max_views=1))
    pixels = _prompt_pixels(scene, points, config)
    median = np.median(pixels, axis=0)
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(), (candidate_of(points),), segmenter, config
    )

    (_, _, prompt) = segmenter.calls[0]
    nearest = np.min(np.linalg.norm(pixels - median, axis=1))
    assert prompt.shape == (1, 2)
    assert prompt.dtype == np.float64
    assert any(np.array_equal(prompt[0], p) for p in pixels)
    assert np.linalg.norm(prompt[0] - median) == pytest.approx(nearest)
    x, y = prompt[0]
    assert scene.instance_maps[0][int(y), int(x)] == 0


def test_each_frame_is_segmented_once_with_all_its_candidates() -> None:
    scene = ring_scene()
    tall = candidate_of(scene.object_points(0, [0]), "before-0")
    low = candidate_of(scene.object_points(1, [0]), "before-1")
    segmenter = scene.segmenter()

    lift_candidates(
        scene.reconstruction(),
        (tall, low),
        segmenter,
        LiftConfig(views=ViewConfig(max_views=6)),
    )

    frames = called_frames(scene, segmenter)
    assert len(frames) == len(set(frames))
    first = segmenter.calls[frames.index(0)]
    assert first[1].shape == (2, 4)
    assert first[2].shape == (2, 2)
    inside_low = scene.instance_maps[0][
        int(first[2][1, 1]), int(first[2][1, 0])
    ]
    assert inside_low == 1


def test_a_candidate_without_views_keeps_its_own_points() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    segmenter = scene.segmenter()

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points, "before-7"),),
        segmenter,
        LiftConfig(views=ViewConfig(min_visible_points=10**9)),
    )

    assert segmenter.calls == []
    assert instance.object_id == "before-7"
    assert instance.epoch == Epoch.BEFORE
    np.testing.assert_array_equal(instance.points, points)
    expected = fit_upright_box(points, 0.05)
    np.testing.assert_array_equal(instance.box.center, expected.center)
    np.testing.assert_array_equal(instance.box.extent, expected.extent)
    np.testing.assert_array_equal(instance.box.rotation, expected.rotation)
    assert instance.label is None


def test_no_candidates_give_no_instances_and_no_segmentation() -> None:
    scene = ring_scene()
    segmenter = scene.segmenter()

    assert lift_candidates(scene.reconstruction(), (), segmenter) == ()
    assert segmenter.calls == []


def test_candidates_of_another_epoch_are_rejected() -> None:
    scene = ring_scene()
    foreign = candidate_of(
        scene.object_points(0, [0]), "after-0", epoch=Epoch.AFTER
    )

    with pytest.raises(ValueError, match="epoch"):
        lift_candidates(scene.reconstruction(), (foreign,), scene.segmenter())


def test_lifted_objects_hold_both_directions() -> None:
    before = ring_scene()
    after = hand_scene([LOW], ring_poses(6), Epoch.AFTER)
    disappeared = candidate_of(before.object_points(0, [0]), "before-0")
    appeared = candidate_of(
        after.object_points(0, [0]), "after-0", epoch=Epoch.AFTER
    )
    segmenter = OracleSegmenter(
        {**before.segmenter().maps, **after.segmenter().maps}
    )

    lifted = lift_change_candidates(
        before.reconstruction(),
        after.reconstruction(),
        ChangeCandidates((disappeared,), (appeared,)),
        segmenter,
        LiftConfig(views=ViewConfig(max_views=1)),
    )

    assert isinstance(lifted, LiftedObjects)
    assert [o.object_id for o in lifted.disappeared] == ["before-0"]
    assert [o.object_id for o in lifted.appeared] == ["after-0"]
    assert lifted.disappeared[0].epoch == Epoch.BEFORE
    assert lifted.appeared[0].epoch == Epoch.AFTER
    images = {id(f.image) for f in before.frames}
    assert {id(c[0]) in images for c in segmenter.calls} == {True, False}


def test_lifted_objects_check_the_epochs_of_the_reconstructions() -> None:
    before = ring_scene()
    after = hand_scene([LOW], ring_poses(6), Epoch.AFTER)
    none = ChangeCandidates((), ())

    with pytest.raises(ValueError, match="before"):
        lift_change_candidates(
            after.reconstruction(),
            after.reconstruction(),
            none,
            before.segmenter(),
        )
    with pytest.raises(ValueError, match="after"):
        lift_change_candidates(
            before.reconstruction(),
            before.reconstruction(),
            none,
            before.segmenter(),
        )


class BleedSegmenter:
    """Grows the masks of one frame by a few pixels in every direction."""

    def __init__(self, inner: OracleSegmenter, image: Any, pixels: int) -> None:
        self._inner = inner
        self._image = image
        self._pixels = pixels

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        masks = self._inner.segment(image, boxes_xyxy, points_xy)
        if image is not self._image:
            return masks
        for _ in range(self._pixels):
            masks[:, 1:] |= masks[:, :-1].copy()
            masks[:, :-1] |= masks[:, 1:].copy()
            masks[:, :, 1:] |= masks[:, :, :-1].copy()
            masks[:, :, :-1] |= masks[:, :, 1:].copy()
        return masks

    def unload(self) -> None:
        pass


class StripeSegmenter:
    """Keeps only the columns within a few pixels of each point prompt."""

    def __init__(self, inner: OracleSegmenter, half_width: int) -> None:
        self._inner = inner
        self._half_width = half_width

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        assert points_xy is not None
        masks = self._inner.segment(image, boxes_xyxy, points_xy)
        columns = np.arange(masks.shape[2])
        for k, (x, _) in enumerate(points_xy):
            masks[k] &= (np.abs(columns - x) <= self._half_width)[None, :]
        return masks

    def unload(self) -> None:
        pass


class EmptySegmenter:
    """Finds nothing."""

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        return np.zeros((len(boxes_xyxy), *image.shape[:2]), np.bool_)

    def unload(self) -> None:
        pass


def outside(points: NDArray[np.float64], box: Cuboid, margin: float) -> int:
    """Counts the points further than a margin from an axis-aligned box."""
    low = np.asarray(box.low) - margin
    high = np.asarray(box.high) + margin
    return int((~np.all((points >= low) & (points <= high), axis=1)).sum())


def stray(
    points: NDArray[np.float64], box: Cuboid, margin: float, reach: float
) -> int:
    """Counts the points between a margin and a reach from a box."""
    gap = np.maximum(
        np.maximum(np.asarray(box.low) - points, points - np.asarray(box.high)),
        0.0,
    ).max(axis=1)
    return int(((gap > margin) & (gap <= reach)).sum())


def test_masks_lift_to_more_of_the_object_than_the_candidate_has() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    config = LiftConfig(views=ViewConfig(max_views=5))

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        scene.segmenter(),
        config,
    )

    assert len(instance.points) > 1.5 * len(points)
    assert outside(instance.points, TALL, 1e-3) == 0
    expected = fit_upright_box(instance.points, config.voxel_size)
    np.testing.assert_array_equal(instance.box.center, expected.center)
    np.testing.assert_array_equal(instance.box.extent, expected.extent)


def test_consensus_drops_voxels_that_only_one_mask_includes() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    reconstruction = scene.reconstruction()
    bleeding = BleedSegmenter(scene.segmenter(), scene.frames[0].image, 6)
    config = LiftConfig(views=ViewConfig(max_views=5), floor_clearance=0.0)
    unchecked = dataclasses.replace(config, min_agreement=0.0)

    (agreed,) = lift_candidates(
        reconstruction, (candidate_of(points),), bleeding, config
    )
    (blind,) = lift_candidates(
        reconstruction, (candidate_of(points),), bleeding, unchecked
    )

    margin = config.voxel_size + 1e-3
    assert stray(blind.points, TALL, margin, NEAR) > 50
    assert stray(agreed.points, TALL, margin, NEAR) == 0
    assert len(agreed.points) > len(points)


def floor_pixels(
    frame: Frame, inside: Callable[[NDArray[np.float64]], NDArray[np.bool_]]
) -> NDArray[np.bool_]:
    """Marks the pixels that see a floor point where ``inside`` holds.

    Args:
        frame: A frame with depth.
        inside: Maps world points of shape ``(H, W, 3)`` to a mask.

    Returns:
        The ``(H, W)`` mask.
    """
    k = frame.intrinsics
    assert frame.depth is not None
    rows, cols = np.mgrid[0 : k.height, 0 : k.width]
    rays = np.stack(
        [(cols - k.cx) / k.fx, (rows - k.cy) / k.fy, np.ones(cols.shape)],
        axis=-1,
    )
    pose = frame.world_from_camera
    with np.errstate(invalid="ignore"):
        world = (rays * frame.depth[..., None]) @ pose.rotation.T
    world += pose.translation
    floor = np.isfinite(frame.depth) & (np.abs(world[..., 2]) < 1e-3)
    with np.errstate(invalid="ignore"):
        return floor & inside(world)


class SpillSegmenter:
    """Adds patches of pixels to the masks of some frames."""

    def __init__(
        self, inner: OracleSegmenter, patches: dict[int, NDArray[np.bool_]]
    ) -> None:
        self._inner = inner
        self._patches = patches

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        masks = self._inner.segment(image, boxes_xyxy, points_xy)
        patch = self._patches.get(id(image))
        if patch is not None:
            masks |= patch
        return masks

    def unload(self) -> None:
        pass


def _far_floor(world: NDArray[np.float64]) -> NDArray[np.bool_]:
    distance = np.linalg.norm(world[..., :2], axis=-1)
    return (distance >= 2.0) & (distance <= 3.0)


def test_spill_far_from_the_object_is_dropped() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    first = scene.frames[0]
    patch = floor_pixels(first, _far_floor)
    assert patch.sum() > 100
    spill = SpillSegmenter(scene.segmenter(), {id(first.image): patch})

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        spill,
        LiftConfig(views=ViewConfig(max_views=1)),
    )

    assert outside(instance.points, TALL, 0.2) == 0
    assert instance.box.extent.max() < 1.5


def test_masks_that_miss_the_candidate_fall_back_to_its_points() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    first = scene.frames[0]
    patch = floor_pixels(first, _far_floor)
    only_spill = SpillSegmenter(
        EmptySegmenter(),  # type: ignore[arg-type]
        {id(first.image): patch},
    )

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        only_spill,
        LiftConfig(views=ViewConfig(max_views=1)),
    )

    np.testing.assert_array_equal(instance.points, points)


def test_voxels_that_straddle_an_edge_are_judged_at_their_points() -> None:
    scene = ring_scene()
    points = scene.object_points(0)
    tight = LiftConfig(
        views=ViewConfig(max_views=6, abs_tolerance=0.002, rel_tolerance=0.0)
    )

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        scene.segmenter(),
        tight,
    )

    assert len(instance.points) >= 0.99 * len(scene.object_points(0))


def _beside_the_tall_box(world: NDArray[np.float64]) -> NDArray[np.bool_]:
    return (
        (world[..., 0] >= 0.5)
        & (world[..., 0] <= 0.8)
        & (np.abs(world[..., 1]) <= 0.5)
    )


def test_a_view_does_not_confirm_its_own_bleed() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0, 1])
    first = scene.frames[0]
    patch = floor_pixels(first, _beside_the_tall_box)
    assert patch.sum() > 50
    oracle = scene.segmenter()
    bleeding = SpillSegmenter(oracle, {id(first.image): patch})

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        bleeding,
        LiftConfig(views=ViewConfig(max_views=2)),
    )

    assert sorted(called_frames(scene, oracle)) == [0, 1]
    assert outside(instance.points, TALL, 0.1) == 0


def test_a_candidate_whose_masks_are_empty_keeps_its_own_points() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])

    (instance,) = lift_candidates(
        scene.reconstruction(), (candidate_of(points),), EmptySegmenter()
    )

    np.testing.assert_array_equal(instance.points, points)


def test_too_few_lifted_points_fall_back_to_the_candidate_points() -> None:
    scene = ring_scene()
    points = scene.object_points(1, [0])
    config = LiftConfig(min_points=10**6)

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        scene.segmenter(),
        config,
    )

    np.testing.assert_array_equal(instance.points, points)


def _halves() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    points = ring_scene().object_points(0, [0])
    return points[points[:, 0] < 0.0], points[points[:, 0] >= 0.0]


def test_fragments_of_one_object_merge_under_the_first_candidate() -> None:
    scene = ring_scene()
    left, right = _halves()
    assert len(left) > 20 and len(right) > 20
    fragments = (
        candidate_of(right, "before-4"),
        candidate_of(left, "before-9"),
    )

    instances = lift_candidates(
        scene.reconstruction(), fragments, scene.segmenter(), LiftConfig()
    )
    (first,) = lift_candidates(
        scene.reconstruction(),
        fragments[:1],
        scene.segmenter(),
        LiftConfig(),
    )

    (merged,) = instances
    assert merged.object_id == "before-4"
    assert outside(merged.points, TALL, 1e-3) == 0
    assert len(np.unique(merged.points, axis=0)) == len(merged.points)
    assert len(merged.points) >= len(first.points)
    np.testing.assert_array_equal(
        merged.points[: len(first.points)], first.points
    )


def test_the_fallback_points_are_a_copy_of_the_candidate_points() -> None:
    scene = ring_scene()
    points = scene.object_points(0, [0])
    candidate = candidate_of(points)

    (instance,) = lift_candidates(
        scene.reconstruction(), (candidate,), EmptySegmenter()
    )

    assert not np.shares_memory(instance.points, candidate.points)
    np.testing.assert_array_equal(instance.points, candidate.points)


def test_separate_objects_stay_separate() -> None:
    scene = ring_scene()
    candidates = (
        candidate_of(scene.object_points(0, [0]), "before-0"),
        candidate_of(scene.object_points(1, [0]), "before-1"),
    )

    instances = lift_candidates(
        scene.reconstruction(), candidates, scene.segmenter()
    )

    assert [i.object_id for i in instances] == ["before-0", "before-1"]
    assert outside(instances[0].points, TALL, 1e-3) == 0
    assert outside(instances[1].points, LOW, 1e-3) == 0


def test_instances_merge_only_above_the_overlap_threshold() -> None:
    scene = ring_scene()
    left, right = _halves()
    fragments = (
        candidate_of(left, "before-0"),
        candidate_of(right, "before-1"),
    )
    stripes = StripeSegmenter(scene.segmenter(), 8)

    low = lift_candidates(
        scene.reconstruction(),
        fragments,
        stripes,
        LiftConfig(merge_overlap=0.3),
    )
    high = lift_candidates(
        scene.reconstruction(),
        fragments,
        stripes,
        LiftConfig(merge_overlap=0.8),
    )

    assert [i.object_id for i in low] == ["before-0"]
    assert [i.object_id for i in high] == ["before-0", "before-1"]


LEFT = Cuboid((-1.6, -0.2, 0.0), (-1.2, 0.2, 0.4))
RIGHT = Cuboid((1.2, -0.2, 0.0), (1.6, 0.2, 0.4))


def _tabletop(world: NDArray[np.float64]) -> NDArray[np.bool_]:
    return (np.abs(world[..., 0]) <= 1.3) & (np.abs(world[..., 1]) <= 0.6)


def test_objects_that_share_leaked_background_stay_separate() -> None:
    scene = hand_scene([LEFT, RIGHT], ring_poses(6))
    leaks = SpillSegmenter(
        scene.segmenter(),
        {id(f.image): floor_pixels(f, _tabletop) for f in scene.frames},
    )
    candidates = (
        candidate_of(scene.object_points(0), "before-0"),
        candidate_of(scene.object_points(1), "before-1"),
    )

    instances = lift_candidates(scene.reconstruction(), candidates, leaks)

    assert [i.object_id for i in instances] == ["before-0", "before-1"]


def in_own_voxels(
    points: NDArray[np.float64],
    own_points: NDArray[np.float64],
    voxel_size: float,
) -> NDArray[np.bool_]:
    """Tells which points lie in a voxel of the candidate's own points."""
    own = np.unique(np.floor(own_points / voxel_size), axis=0)
    cells = np.floor(points / voxel_size)
    _, ids = np.unique(
        np.concatenate([own, cells]), axis=0, return_inverse=True
    )
    ids = ids.reshape(-1)
    return np.isin(ids[len(own) :], ids[: len(own)])


def floor_strays(
    points: NDArray[np.float64],
    own_points: NDArray[np.float64],
    config: LiftConfig,
) -> int:
    """Counts low points outside the voxels of the candidate's own points."""
    own = in_own_voxels(points, own_points, config.voxel_size)
    return int(((points[:, 2] < config.floor_clearance) & ~own).sum())


def test_floor_spill_beside_the_object_is_dropped() -> None:
    scene = ring_scene()
    points = scene.object_points(0)
    spill = SpillSegmenter(
        scene.segmenter(),
        {
            id(f.image): floor_pixels(f, _beside_the_tall_box)
            for f in scene.frames
        },
    )
    config = LiftConfig(views=ViewConfig(max_views=6))
    unclipped = dataclasses.replace(config, floor_clearance=0.0)

    (instance,) = lift_candidates(
        scene.reconstruction(), (candidate_of(points),), spill, config
    )
    (raw,) = lift_candidates(
        scene.reconstruction(), (candidate_of(points),), spill, unclipped
    )

    assert floor_strays(raw.points, points, config) > 50
    assert floor_strays(instance.points, points, config) == 0
    assert outside(instance.points, TALL, config.voxel_size + 1e-3) == 0
    np.testing.assert_allclose(
        instance.box.extent, [1.0, 1.0, 1.0], atol=2 * config.voxel_size
    )


def test_the_floor_contact_of_the_candidate_is_kept() -> None:
    scene = ring_scene()
    points = scene.object_points(0)
    config = LiftConfig(views=ViewConfig(max_views=6), floor_clearance=0.05)
    unclipped = dataclasses.replace(config, floor_clearance=0.0)

    (instance,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        scene.segmenter(),
        config,
    )
    (raw,) = lift_candidates(
        scene.reconstruction(),
        (candidate_of(points),),
        scene.segmenter(),
        unclipped,
    )

    low = instance.points[:, 2] < config.floor_clearance
    assert low.sum() > 20
    assert low.sum() == (raw.points[:, 2] < config.floor_clearance).sum()
    assert len(instance.points) == len(raw.points)


def test_objects_that_share_floor_spill_stay_separate() -> None:
    scene = hand_scene([LEFT, RIGHT], ring_poses(6))
    leaks = SpillSegmenter(
        scene.segmenter(),
        {id(f.image): floor_pixels(f, _tabletop) for f in scene.frames},
    )
    own = [scene.object_points(0), scene.object_points(1)]
    config = LiftConfig()

    instances = lift_candidates(
        scene.reconstruction(),
        (candidate_of(own[0], "before-0"), candidate_of(own[1], "before-1")),
        leaks,
        config,
    )

    assert [i.object_id for i in instances] == ["before-0", "before-1"]
    for instance, mine in zip(instances, own, strict=True):
        assert floor_strays(instance.points, mine, config) == 0
