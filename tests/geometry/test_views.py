import dataclasses

import numpy as np
import pytest
from lifting_scenes import INTRINSICS, Cuboid, hand_scene, ring_poses

from spatial_change.domain import Epoch, EpochReconstruction
from spatial_change.geometry.projection import (
    DEFAULT_MIN_CONFIDENCE,
    visible_points,
)
from spatial_change.geometry.views import (
    ViewConfig,
    pixel_box,
    select_views,
    usable_frames,
)
from spatial_change.synthetic.trajectory import look_at

BOX = Cuboid((-0.4, -0.4, 0.0), (0.4, 0.4, 0.8))


def _scene(poses: list):  # type: ignore[type-arg]
    scene = hand_scene([BOX], poses)
    return scene, scene.object_points(0)


def test_usable_frames_keeps_frames_with_image_and_depth() -> None:
    scene, _ = _scene(ring_poses(4))
    frames = list(scene.frames)
    frames[1] = dataclasses.replace(frames[1], image=None)
    frames[3] = dataclasses.replace(frames[3], depth=None)
    reconstruction = EpochReconstruction(
        epoch=Epoch.BEFORE,
        frames=frames,
        points=np.zeros((0, 3)),
        colors=np.zeros((0, 3), np.uint8),
        confidence=np.zeros(0, np.float32),
    )

    found = usable_frames(reconstruction, 0.1)

    assert list(found) == [0, 2]
    assert found[2].frame is frames[2]
    assert found[2].image is frames[2].image
    assert [f.name for f in dataclasses.fields(found[2])] == ["frame", "usable"]
    assert found[2].usable.shape == (INTRINSICS.height, INTRINSICS.width)
    assert found[2].usable.all()


def test_views_are_the_frames_that_see_the_most_points_in_frame_order() -> None:
    poses = [
        look_at([6.0, 0.0, 3.0], [0.0, 0.0, 0.4]),
        look_at([1.6, 0.0, 1.0], [0.0, 0.0, 0.4]),
        look_at([0.0, 6.0, 3.0], [0.0, 0.0, 0.4]),
        look_at([4.0, 0.0, 3.0], [8.0, 0.0, 3.0]),
    ]
    scene, points = _scene(poses)
    frames = usable_frames(scene.reconstruction(), 0.1)
    counts = [
        len(visible_points(points, f.frame, f.usable, 0.05, 0.02).indices)
        for f in frames.values()
    ]
    best_two = sorted(np.argsort(-np.array(counts), kind="stable")[:2])

    views = select_views(points, frames, ViewConfig(max_views=2))

    assert counts[3] == 0
    assert [v.position for v in views] == best_two
    assert [len(v.seen.indices) for v in views] == [counts[i] for i in best_two]


def test_a_tie_goes_to_the_earlier_frame() -> None:
    pose = look_at([4.0, 0.0, 3.0], [0.0, 0.0, 0.4])
    scene, points = _scene([pose, pose, pose])
    frames = usable_frames(scene.reconstruction(), 0.1)

    views = select_views(points, frames, ViewConfig(max_views=2))

    assert [v.position for v in views] == [0, 1]


def test_frames_that_see_too_few_points_give_no_view() -> None:
    away = look_at([4.0, 0.0, 3.0], [8.0, 0.0, 3.0])
    scene, points = _scene([away, *ring_poses(2)])
    frames = usable_frames(scene.reconstruction(), 0.1)

    views = select_views(points, frames, ViewConfig(max_views=5))

    assert [v.position for v in views] == [1, 2]
    assert all(len(v.seen.indices) >= 10 for v in views)


def test_a_minimum_above_every_count_gives_no_views() -> None:
    scene, points = _scene(ring_poses(3))
    frames = usable_frames(scene.reconstruction(), 0.1)

    assert (
        select_views(points, frames, ViewConfig(min_visible_points=10**9)) == []
    )


def test_views_carry_the_visible_pixels_of_the_points() -> None:
    scene, points = _scene(ring_poses(2))
    frames = usable_frames(scene.reconstruction(), 0.1)

    (first, _) = select_views(points, frames, ViewConfig(max_views=2))

    mask = scene.instance_maps[first.position] == 0
    seen = first.seen.pixels
    assert mask[seen[:, 1], seen[:, 0]].all()


def test_the_box_of_pixels_is_grown_by_the_margin() -> None:
    pixels = np.array([[10, 20], [14, 26]], np.intp)

    box = pixel_box(pixels, 3.0, 100, 80)

    np.testing.assert_array_equal(box, [7.0, 17.0, 17.0, 29.0])
    assert box.dtype == np.float64


def test_the_box_is_clipped_to_the_image() -> None:
    pixels = np.array([[1, 2], [98, 78]], np.intp)

    box = pixel_box(pixels, 5.0, 100, 80)

    np.testing.assert_array_equal(box, [0.0, 0.0, 99.0, 79.0])


def test_the_margin_may_differ_per_axis() -> None:
    pixels = np.array([[30, 30], [40, 50]], np.intp)

    box = pixel_box(pixels, np.array([2.0, 6.0]), 100, 100)

    np.testing.assert_array_equal(box, [28.0, 24.0, 42.0, 56.0])


@pytest.mark.parametrize("margin", [0.0, 4.0])
def test_one_pixel_gives_a_box_of_the_margin(margin: float) -> None:
    box = pixel_box(np.array([[50, 40]], np.intp), margin, 100, 100)

    np.testing.assert_array_equal(
        box, [50 - margin, 40 - margin, 50 + margin, 40 + margin]
    )


def test_view_config_defaults() -> None:
    config = ViewConfig()

    assert config.max_views == 5
    assert config.min_visible_points == 10
    assert config.abs_tolerance == 0.05
    assert config.rel_tolerance == 0.02
    assert config.min_confidence == DEFAULT_MIN_CONFIDENCE


def test_view_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        ViewConfig().max_views = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_views": 0},
        {"min_visible_points": 0},
        {"abs_tolerance": 0.0},
        {"abs_tolerance": float("inf")},
        {"rel_tolerance": -0.1},
        {"rel_tolerance": float("nan")},
        {"min_confidence": 1.5},
        {"min_confidence": -0.1},
    ],
)
def test_view_config_rejects_invalid_values(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        ViewConfig(**overrides)  # type: ignore[arg-type]
