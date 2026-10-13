import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.domain import SE3, ChangeType, Epoch, Frame, Intrinsics
from spatial_change.evaluation.projection import (
    ChangeObject3D,
    ProjectionConfig,
    project_to_pred_items,
)
from spatial_change.pipeline.videos import ImageWindow

SIZE = 20
INTRINSICS = Intrinsics(
    fx=10.0, fy=10.0, cx=10.0, cy=10.0, width=SIZE, height=SIZE
)
EXACT = ProjectionConfig(splat_radius=0)


def frame(
    epoch: Epoch = Epoch.BEFORE,
    index: int = 0,
    depth_value: float = 5.0,
    pose: SE3 | None = None,
    depth: NDArray[np.float32] | None = None,
) -> Frame:
    if depth is None:
        depth = np.full((SIZE, SIZE), depth_value, dtype=np.float32)
    return Frame(
        epoch=epoch,
        index=index,
        timestamp_s=0.0,
        intrinsics=INTRINSICS,
        world_from_camera=pose or SE3.identity(),
        depth=depth,
    )


def render(
    obj: ChangeObject3D, cam: Frame, config: ProjectionConfig | None = None
) -> NDArray[np.bool_] | None:
    before = [cam] if cam.epoch is Epoch.BEFORE else []
    after = [cam] if cam.epoch is Epoch.AFTER else []
    items = project_to_pred_items(
        [obj], before, after, SIZE, SIZE, config=config, stride=1
    )
    return next(iter(items[0].masks.values())) if items else None


def face(z: float = 5.0, half: float = 0.5, n: int = 5) -> NDArray[np.float64]:
    xs = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


def block(lo: int, hi: int) -> NDArray[np.bool_]:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[lo : hi + 1, lo : hi + 1] = True
    return mask


def test_cube_face_projects_to_the_expected_pixels() -> None:
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})

    mask = render(obj, frame(), EXACT)

    assert mask is not None
    np.testing.assert_array_equal(mask, block(9, 11))


def test_pose_moves_the_projection() -> None:
    # Camera moved 1 m right: a point at x = 0 now sits at x = -1 in the
    # camera frame, so u = 10 * -1 / 5 + 10 = 8.
    pose = SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
    obj = ChangeObject3D(
        0,
        ChangeType.REMOVED,
        0.8,
        {Epoch.BEFORE: np.array([[0.0, 0.0, 5.0]])},
    )

    mask = render(obj, frame(pose=pose), EXACT)

    assert mask is not None
    expected = np.zeros((SIZE, SIZE), dtype=bool)
    expected[10, 8] = True
    np.testing.assert_array_equal(mask, expected)


def test_splat_radius_draws_a_disk_around_each_point() -> None:
    obj = ChangeObject3D(
        0,
        ChangeType.REMOVED,
        0.8,
        {Epoch.BEFORE: np.array([[0.0, 0.0, 5.0]])},
    )

    mask = render(obj, frame(), ProjectionConfig(splat_radius=2))

    assert mask is not None
    rows, cols = np.mgrid[:SIZE, :SIZE]
    expected = (rows - 10) ** 2 + (cols - 10) ** 2 <= 4
    np.testing.assert_array_equal(mask, expected)


def test_splat_is_clipped_at_the_image_border() -> None:
    obj = ChangeObject3D(
        0,
        ChangeType.REMOVED,
        0.8,
        {Epoch.BEFORE: np.array([[-4.9, 0.0, 5.0]])},
    )

    mask = render(obj, frame(), ProjectionConfig(splat_radius=3))

    assert mask is not None
    assert mask[:, :4].any()
    assert mask.shape == (SIZE, SIZE)


def test_points_behind_a_nearer_surface_are_not_drawn() -> None:
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})

    assert render(obj, frame(depth_value=3.0), EXACT) is None


def test_occlusion_is_decided_per_pixel() -> None:
    depth = np.full((SIZE, SIZE), 5.0, dtype=np.float32)
    depth[:, :10] = 2.0
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})

    mask = render(obj, frame(depth=depth), EXACT)

    assert mask is not None
    assert not mask[:, :10].any()
    assert mask[:, 10:].any()


def test_points_within_the_tolerance_are_drawn() -> None:
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})
    config = ProjectionConfig(
        splat_radius=0, abs_tolerance=0.0, rel_tolerance=0.05
    )

    assert render(obj, frame(depth_value=4.8), config) is not None
    assert render(obj, frame(depth_value=4.6), config) is None


def test_points_in_front_of_the_observed_depth_are_drawn() -> None:
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})

    assert render(obj, frame(depth_value=9.0), EXACT) is not None


def test_invalid_depth_is_not_drawn() -> None:
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})

    assert render(obj, frame(depth_value=0.0), EXACT) is None


def test_low_confidence_pixels_are_not_drawn() -> None:
    cam = frame()
    cam.confidence = np.zeros((SIZE, SIZE), dtype=np.float32)
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})

    assert render(obj, cam, EXACT) is None


def test_points_behind_the_camera_or_outside_the_image_are_dropped() -> None:
    points = np.array(
        [
            [0.0, 0.0, -5.0],
            [50.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ]
    )
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: points})

    mask = render(obj, frame(), EXACT)

    assert mask is not None
    assert mask.sum() == 1
    assert mask[10, 10]


def test_object_seen_nowhere_has_no_mask() -> None:
    points = np.array([[0.0, 0.0, -5.0]])
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: points})

    assert render(obj, frame(), EXACT) is None


def test_object_absent_in_the_frame_epoch_has_no_mask() -> None:
    obj = ChangeObject3D(0, ChangeType.ADDED, 0.8, {Epoch.AFTER: face()})

    assert render(obj, frame(epoch=Epoch.BEFORE), EXACT) is None


def test_frame_without_depth_is_rejected() -> None:
    cam = frame()
    cam.depth = None
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: face()})

    with pytest.raises(ValueError, match="depth"):
        render(obj, cam, EXACT)


def test_change_type_must_agree_with_the_epochs_of_the_points() -> None:
    with pytest.raises(ValueError, match="added"):
        ChangeObject3D(0, ChangeType.ADDED, 0.8, {Epoch.BEFORE: face()})
    with pytest.raises(ValueError, match="moved"):
        ChangeObject3D(0, ChangeType.MOVED, 0.8, {Epoch.AFTER: face()})
    with pytest.raises(ValueError, match="removed"):
        ChangeObject3D(
            0,
            ChangeType.REMOVED,
            0.8,
            {Epoch.BEFORE: face(), Epoch.AFTER: face()},
        )


def test_uncertain_objects_are_rejected() -> None:
    with pytest.raises(ValueError, match="uncertain"):
        ChangeObject3D(0, ChangeType.UNCERTAIN, 0.8, {Epoch.BEFORE: face()})


def test_points_must_be_an_n_by_3_array() -> None:
    with pytest.raises(ValueError, match="points"):
        ChangeObject3D(
            0, ChangeType.REMOVED, 0.8, {Epoch.BEFORE: np.zeros((4, 2))}
        )


def test_pred_items_use_video_and_stride_indexed_keys() -> None:
    moved = ChangeObject3D(
        7,
        ChangeType.MOVED,
        0.6,
        {Epoch.BEFORE: face(), Epoch.AFTER: face(half=1.0)},
    )
    before = [frame(Epoch.BEFORE, 0), frame(Epoch.BEFORE, 30)]
    after = [frame(Epoch.AFTER, 60)]

    items = project_to_pred_items(
        [moved], before, after, SIZE, SIZE, config=EXACT, stride=30
    )

    assert len(items) == 1
    item = items[0]
    assert item.obj_id == 7
    assert sorted(item.masks) == [(1, 0), (1, 1), (2, 2)]
    assert item.moved
    assert set(item.scores.values()) == {0.6}
    np.testing.assert_array_equal(item.masks[(1, 0)], block(9, 11))
    assert item.masks[(2, 2)].sum() > item.masks[(1, 0)].sum()


def test_frames_off_the_stride_are_skipped() -> None:
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.5, {Epoch.BEFORE: face()})

    items = project_to_pred_items(
        [obj],
        [frame(Epoch.BEFORE, 7), frame(Epoch.BEFORE, 30)],
        [],
        SIZE,
        SIZE,
        config=EXACT,
        stride=30,
    )

    assert sorted(items[0].masks) == [(1, 1)]


def test_objects_seen_in_no_frame_are_left_out() -> None:
    hidden = ChangeObject3D(
        1, ChangeType.REMOVED, 0.5, {Epoch.BEFORE: face(z=-5.0)}
    )

    assert (
        project_to_pred_items(
            [hidden], [frame(Epoch.BEFORE, 0)], [], SIZE, SIZE, config=EXACT
        )
        == []
    )


def test_masks_are_resized_to_the_evaluation_grid() -> None:
    obj = ChangeObject3D(0, ChangeType.REMOVED, 0.5, {Epoch.BEFORE: face()})

    items = project_to_pred_items(
        [obj], [frame(Epoch.BEFORE, 0)], [], 40, 10, config=EXACT
    )

    assert items[0].masks[(1, 0)].shape == (40, 10)


def test_added_object_appears_only_in_video_two() -> None:
    obj = ChangeObject3D(0, ChangeType.ADDED, 0.5, {Epoch.AFTER: face()})

    items = project_to_pred_items(
        [obj],
        [frame(Epoch.BEFORE, 0)],
        [frame(Epoch.AFTER, 0)],
        SIZE,
        SIZE,
        config=EXACT,
    )

    assert sorted(items[0].masks) == [(2, 0)]
    assert not items[0].moved


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="splat"):
        ProjectionConfig(splat_radius=-1)
    with pytest.raises(ValueError, match="tolerance"):
        ProjectionConfig(abs_tolerance=-0.1)
    with pytest.raises(ValueError, match="tolerance"):
        ProjectionConfig(rel_tolerance=-0.1)
    with pytest.raises(ValueError, match="stride"):
        project_to_pred_items([], [], [], SIZE, SIZE, stride=0)


def test_before_and_after_frames_are_validated() -> None:
    with pytest.raises(ValueError, match="before"):
        project_to_pred_items([], [frame(Epoch.AFTER)], [], SIZE, SIZE)
    with pytest.raises(ValueError, match="after"):
        project_to_pred_items([], [], [frame(Epoch.BEFORE)], SIZE, SIZE)


def test_usable_pixels_are_computed_once_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spatial_change.evaluation import projection

    calls = []
    real = projection.usable_pixels
    monkeypatch.setattr(
        projection,
        "usable_pixels",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    objs = [
        ChangeObject3D(i, ChangeType.REMOVED, 0.5, {Epoch.BEFORE: face()})
        for i in range(3)
    ]

    project_to_pred_items(
        objs, [frame(Epoch.BEFORE, 0)], [], SIZE, SIZE, config=EXACT
    )

    assert len(calls) == 1


def test_splat_larger_than_the_image_is_clipped() -> None:
    obj = ChangeObject3D(
        0,
        ChangeType.REMOVED,
        0.8,
        {Epoch.BEFORE: np.array([[0.0, 0.0, 5.0]])},
    )

    mask = render(obj, frame(), ProjectionConfig(splat_radius=50))

    assert mask is not None
    assert mask.all()


def test_splat_at_the_border_is_the_clipped_disk() -> None:
    obj = ChangeObject3D(
        0,
        ChangeType.REMOVED,
        0.8,
        {Epoch.BEFORE: np.array([[-4.9, 0.0, 5.0]])},
    )

    mask = render(obj, frame(), ProjectionConfig(splat_radius=3))

    rows, cols = np.mgrid[:SIZE, :SIZE]
    expected = (rows - 10) ** 2 + (cols - 0) ** 2 <= 9
    assert mask is not None
    np.testing.assert_array_equal(mask, expected)


def every_pixel(z: float = 5.0) -> NDArray[np.float64]:
    rows, cols = np.mgrid[:SIZE, :SIZE]
    x = (cols.ravel() - INTRINSICS.cx) * z / INTRINSICS.fx
    y = (rows.ravel() - INTRINSICS.cy) * z / INTRINSICS.fy
    return np.stack([x, y, np.full(x.size, z)], axis=1)


def test_a_whole_frame_mask_fills_only_the_window_of_the_grid() -> None:
    moved = ChangeObject3D(
        0,
        ChangeType.MOVED,
        0.5,
        {Epoch.BEFORE: every_pixel(), Epoch.AFTER: every_pixel()},
    )
    before = [frame(Epoch.BEFORE, 0)]
    after = [frame(Epoch.AFTER, 0)]

    (item,) = project_to_pred_items(
        [moved],
        before,
        after,
        20,
        40,
        config=EXACT,
        stride=1,
        after_window=ImageWindow(0.0, 0.25, 1.0, 0.75),
    )

    assert item.masks[(1, 0)].all()
    inside = np.zeros((20, 40), dtype=bool)
    inside[:, 10:30] = True
    np.testing.assert_array_equal(item.masks[(2, 0)], inside)


def test_without_a_window_the_projection_is_what_it_was() -> None:
    moved = ChangeObject3D(
        0,
        ChangeType.MOVED,
        0.5,
        {Epoch.BEFORE: face(), Epoch.AFTER: face(half=1.0)},
    )
    frames = ([frame(Epoch.BEFORE, 0)], [frame(Epoch.AFTER, 0)])

    plain = project_to_pred_items(
        [moved], *frames, 30, 50, config=EXACT, stride=1
    )
    whole = project_to_pred_items(
        [moved],
        *frames,
        30,
        50,
        config=EXACT,
        stride=1,
        after_window=ImageWindow(),
    )

    assert [sorted(i.masks) for i in whole] == [sorted(plain[0].masks)]
    for key, mask in plain[0].masks.items():
        np.testing.assert_array_equal(whole[0].masks[key], mask)


def test_a_window_smaller_than_a_pixel_still_holds_one() -> None:
    obj = ChangeObject3D(0, ChangeType.ADDED, 0.5, {Epoch.AFTER: every_pixel()})

    (item,) = project_to_pred_items(
        [obj],
        [],
        [frame(Epoch.AFTER, 0)],
        10,
        10,
        config=EXACT,
        stride=1,
        after_window=ImageWindow(0.99, 0.99, 1.0, 1.0),
    )

    mask = item.masks[(2, 0)]
    assert mask.sum() == 1
    assert mask[9, 9]


def camera(intrinsics: Intrinsics, epoch: Epoch) -> Frame:
    return Frame(
        epoch=epoch,
        index=0,
        timestamp_s=0.0,
        intrinsics=intrinsics,
        world_from_camera=SE3.identity(),
        depth=np.full((intrinsics.height, intrinsics.width), 5.0, np.float32),
    )


@pytest.mark.parametrize("scale", [1, 2])
def test_a_cropped_camera_with_its_window_matches_the_full_camera(
    scale: int,
) -> None:
    full = Intrinsics(fx=10.0, fy=10.0, cx=20.0, cy=10.0, width=40, height=20)
    cropped = Intrinsics(
        fx=10.0, fy=10.0, cx=10.0, cy=10.0, width=20, height=20
    )
    window = ImageWindow(0.0, 0.25, 1.0, 0.75)
    obj = ChangeObject3D(
        0, ChangeType.ADDED, 0.5, {Epoch.AFTER: face(half=2.0, n=9)}
    )
    grid = (20 * scale, 40 * scale)

    (reference,) = project_to_pred_items(
        [obj],
        [],
        [camera(full, Epoch.AFTER)],
        *grid,
        config=EXACT,
        stride=1,
    )
    (placed,) = project_to_pred_items(
        [obj],
        [],
        [camera(cropped, Epoch.AFTER)],
        *grid,
        config=EXACT,
        stride=1,
        after_window=window,
    )

    assert reference.masks[(2, 0)].any()
    np.testing.assert_array_equal(placed.masks[(2, 0)], reference.masks[(2, 0)])
