import itertools

import numpy as np
import pytest

from spatial_change.domain import SE3, ChangeType, Epoch, OrientedBox
from spatial_change.geometry.projection import (
    in_image,
    project,
    unproject_depth,
)
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticObject,
    SyntheticScenePair,
    generate_scene_pair,
)
from spatial_change.synthetic.trajectory import sees_points

CONFIG = SceneConfig(width=64, height=48, frames_per_epoch=6)
SEEDS = [0, 1, 2]


@pytest.fixture(scope="module", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def pair(request: pytest.FixtureRequest) -> SyntheticScenePair:
    return generate_scene_pair(request.param, CONFIG)


def by_status(
    pair: SyntheticScenePair, status: ObjectStatus
) -> list[SyntheticObject]:
    return [o for o in pair.objects if o.status == status]


def obbs_overlap(a: OrientedBox, b: OrientedBox, eps: float = 1e-6) -> bool:
    """Separating axis test; touching boxes do not overlap."""
    axes = [*a.rotation.T, *b.rotation.T]
    for u, v in itertools.product(a.rotation.T, b.rotation.T):
        cross = np.cross(u, v)
        if np.linalg.norm(cross) > 1e-9:
            axes.append(cross / np.linalg.norm(cross))
    for axis in axes:
        ra = np.sum(np.abs(a.rotation.T @ axis) * a.extent / 2)
        rb = np.sum(np.abs(b.rotation.T @ axis) * b.extent / 2)
        if abs((b.center - a.center) @ axis) >= ra + rb - eps:
            return False
    return True


def test_same_seed_is_byte_identical_and_other_seed_differs() -> None:
    a = generate_scene_pair(5, CONFIG)
    b = generate_scene_pair(5, CONFIG)
    c = generate_scene_pair(6, CONFIG)

    assert [o.object_id for o in a.objects] == [o.object_id for o in b.objects]
    for oa, ob in zip(a.objects, b.objects, strict=True):
        assert oa.status == ob.status
        for epoch in Epoch:
            if oa.exists_in(epoch):
                np.testing.assert_array_equal(
                    oa.box(epoch).center, ob.box(epoch).center
                )
                np.testing.assert_array_equal(
                    oa.pose(epoch).matrix, ob.pose(epoch).matrix
                )
    np.testing.assert_array_equal(
        a.unobserved_zone.center, b.unobserved_zone.center
    )
    assert a.occluded_object_ids == b.occluded_object_ids
    for epoch in Epoch:
        ea, eb = a.epoch_data(epoch), b.epoch_data(epoch)
        for fa, fb, ia, ib in zip(
            ea.frames, eb.frames, ea.instance_maps, eb.instance_maps,
            strict=True,
        ):  # fmt: skip
            assert fa.image is not None and fb.image is not None
            assert fa.depth is not None and fb.depth is not None
            assert fa.image.tobytes() == fb.image.tobytes()
            assert fa.depth.tobytes() == fb.depth.tobytes()
            assert ia.tobytes() == ib.tobytes()
            np.testing.assert_array_equal(
                fa.world_from_camera.matrix, fb.world_from_camera.matrix
            )
    centers_a = [o.box(Epoch.BEFORE).center for o in a.objects if o.box_before]
    centers_c = [o.box(Epoch.BEFORE).center for o in c.objects if o.box_before]
    assert not np.allclose(centers_a, centers_c)


def test_frames_carry_domain_metadata(pair: SyntheticScenePair) -> None:
    for epoch in Epoch:
        data = pair.epoch_data(epoch)
        assert len(data.frames) == len(data.instance_maps) == 6
        for i, (frame, instance) in enumerate(
            zip(data.frames, data.instance_maps, strict=True)
        ):
            assert frame.epoch == epoch
            assert frame.index == i
            assert frame.timestamp_s == pytest.approx(i / CONFIG.fps)
            assert (frame.intrinsics.width, frame.intrinsics.height) == (64, 48)
            assert frame.image is not None and frame.image.shape == (48, 64, 3)
            assert frame.depth is not None and frame.depth.dtype == np.float32
            assert instance.shape == (48, 64) and instance.dtype == np.int32
            height = frame.world_from_camera.translation[2]
            assert 1.3 < height < 1.7


def test_cameras_stay_inside_the_room_and_outside_objects(
    pair: SyntheticScenePair,
) -> None:
    room = np.array(pair.room_size)
    for epoch in Epoch:
        for frame in pair.epoch_data(epoch).frames:
            center = frame.world_from_camera.translation
            assert np.all(center > 0.2) and np.all(center < room - 0.2)
            for obj in pair.objects_in(epoch):
                assert not obj.box(epoch).contains(center[None, :])[0]


def test_every_change_type_is_present(pair: SyntheticScenePair) -> None:
    for status in ObjectStatus:
        assert by_status(pair, status), status
    assert len({o.object_id for o in pair.objects}) == len(pair.objects)
    for index, obj in enumerate(pair.objects):
        assert obj.index == index


def test_status_maps_to_domain_change_type() -> None:
    assert ObjectStatus.ADDED.change_type == ChangeType.ADDED
    assert ObjectStatus.REMOVED.change_type == ChangeType.REMOVED
    assert ObjectStatus.MOVED.change_type == ChangeType.MOVED
    assert ObjectStatus.UNCHANGED.change_type is None


def test_added_and_removed_exist_in_one_epoch_only(
    pair: SyntheticScenePair,
) -> None:
    for obj in by_status(pair, ObjectStatus.ADDED):
        assert obj.box_before is None and obj.box_after is not None
        assert obj.motion is None
    for obj in by_status(pair, ObjectStatus.REMOVED):
        assert obj.box_before is not None and obj.box_after is None
        assert obj.motion is None


def test_moved_boxes_differ_by_the_ground_truth_transform(
    pair: SyntheticScenePair,
) -> None:
    for obj in by_status(pair, ObjectStatus.MOVED):
        before, after = obj.box(Epoch.BEFORE), obj.box(Epoch.AFTER)
        assert obj.motion is not None
        np.testing.assert_allclose(
            obj.motion.apply(before.corners()), after.corners(), atol=1e-9
        )
        np.testing.assert_allclose(
            obj.motion.rotation @ before.rotation, after.rotation, atol=1e-9
        )
        np.testing.assert_allclose(
            obj.motion.compose(obj.pose(Epoch.BEFORE)).matrix,
            obj.pose(Epoch.AFTER).matrix,
            atol=1e-9,
        )
        move = np.linalg.norm(after.center - before.center)
        assert move >= CONFIG.min_move_m - 1e-9
        assert obj.displacement_m == pytest.approx(move)


def test_unchanged_objects_keep_their_pose(pair: SyntheticScenePair) -> None:
    for obj in by_status(pair, ObjectStatus.UNCHANGED):
        np.testing.assert_array_equal(
            obj.pose(Epoch.BEFORE).matrix, obj.pose(Epoch.AFTER).matrix
        )


def test_objects_rest_on_floor_or_support_and_do_not_interpenetrate(
    pair: SyntheticScenePair,
) -> None:
    room = np.array(pair.room_size)
    by_id = {o.object_id: o for o in pair.objects}
    for epoch in Epoch:
        present = pair.objects_in(epoch)
        for obj in present:
            box = obj.box(epoch)
            corners = box.corners()
            assert np.all(corners > -1e-9) and np.all(corners < room + 1e-9)
            bottom = box.center[2] - box.extent[2] / 2
            if obj.support_id is None:
                assert bottom == pytest.approx(0.0, abs=1e-9)
            else:
                support = by_id[obj.support_id].box(epoch)
                top = support.center[2] + support.extent[2] / 2
                assert bottom == pytest.approx(top, abs=1e-9)
        for a, b in itertools.combinations(present, 2):
            assert not obbs_overlap(a.box(epoch), b.box(epoch)), (
                a.object_id,
                b.object_id,
                epoch,
            )


def test_instance_maps_only_show_objects_of_their_epoch(
    pair: SyntheticScenePair,
) -> None:
    for epoch in Epoch:
        allowed = {-1} | {o.index for o in pair.objects_in(epoch)}
        for instance in pair.epoch_data(epoch).instance_maps:
            assert set(np.unique(instance).tolist()) <= allowed
    added = by_status(pair, ObjectStatus.ADDED)
    removed = by_status(pair, ObjectStatus.REMOVED)
    before = pair.epoch_data(Epoch.BEFORE).instance_maps
    after = pair.epoch_data(Epoch.AFTER).instance_maps
    assert any(np.any(m == o.index) for o in added for m in after)
    assert any(np.any(m == o.index) for o in removed for m in before)


def test_instance_pixels_unproject_inside_their_box(
    pair: SyntheticScenePair,
) -> None:
    for epoch in Epoch:
        data = pair.epoch_data(epoch)
        for obj in pair.objects_in(epoch):
            box = obj.box(epoch)
            grown = OrientedBox(box.center, box.extent + 0.02, box.rotation)
            seen = 0
            for frame, instance in zip(
                data.frames, data.instance_maps, strict=True
            ):
                assert frame.depth is not None
                mask = instance == obj.index
                points = unproject_depth(
                    frame.depth,
                    frame.intrinsics,
                    frame.world_from_camera,
                    mask,
                )
                assert np.all(grown.contains(points)), obj.object_id
                seen += len(points)
            if obj.status != ObjectStatus.UNCHANGED:
                assert seen > 0, (obj.object_id, epoch)


def test_occluded_object_is_seen_before_and_hidden_after(
    pair: SyntheticScenePair,
) -> None:
    assert len(pair.occluded_object_ids) >= 1
    occluder = pair.object_by_id(pair.occluder_id)
    assert occluder.status == ObjectStatus.UNCHANGED
    for object_id in pair.occluded_object_ids:
        hidden = pair.object_by_id(object_id)
        assert hidden.status == ObjectStatus.UNCHANGED
        before = pair.epoch_data(Epoch.BEFORE)
        assert any(np.any(m == hidden.index) for m in before.instance_maps)

        after = pair.epoch_data(Epoch.AFTER)
        projected_frames = 0
        for frame, instance in zip(
            after.frames, after.instance_maps, strict=True
        ):
            corners = hidden.box(Epoch.AFTER).corners()
            uv, depth = project(
                corners, frame.intrinsics, frame.world_from_camera
            )
            center_uv, center_depth = project(
                hidden.box(Epoch.AFTER).center[None, :],
                frame.intrinsics,
                frame.world_from_camera,
            )
            if not in_image(center_uv, center_depth, frame.intrinsics)[0]:
                continue
            projected_frames += 1
            assert not np.any(instance == hidden.index)
            lo = np.clip(np.floor(uv.min(axis=0)), 0, None).astype(int)
            hi = np.ceil(uv.max(axis=0)).astype(int) + 1
            patch = instance[lo[1] : hi[1], lo[0] : hi[0]]
            assert patch.size > 0 and np.all(patch == occluder.index)
        assert projected_frames == len(after.frames)


def test_unobserved_zone_is_unseen_after_and_seen_before(
    pair: SyntheticScenePair,
) -> None:
    zone = pair.unobserved_zone
    assert pair.zone_seen_in == Epoch.BEFORE
    inside = [
        o
        for o in pair.objects_in(Epoch.BEFORE)
        if np.all(zone.contains(o.box(Epoch.BEFORE).corners()))
    ]
    assert inside

    local = np.linspace(-0.5, 0.5, 7)
    grid = np.array(list(itertools.product(local, local, local)))
    samples = SE3(zone.rotation, zone.center).apply(grid * zone.extent)
    for frame in pair.epoch_data(Epoch.AFTER).frames:
        assert not np.any(
            sees_points(samples, frame.intrinsics, frame.world_from_camera)
        )
        assert frame.depth is not None
        points = unproject_depth(
            frame.depth, frame.intrinsics, frame.world_from_camera
        )
        assert not np.any(zone.contains(points))

    before_hits = 0
    for frame in pair.epoch_data(Epoch.BEFORE).frames:
        assert frame.depth is not None
        points = unproject_depth(
            frame.depth, frame.intrinsics, frame.world_from_camera
        )
        before_hits += int(np.sum(zone.contains(points)))
    assert before_hits > 0


def seen_through(
    pair: SyntheticScenePair, box: OrientedBox, epoch: Epoch
) -> tuple[bool, float]:
    """Tells where the other epoch's cameras saw through a box.

    A point is seen through when it projects into some frame of ``epoch``
    and the rendered depth there exceeds the point's depth by 5 cm.

    Returns:
        Whether the box center is seen through, and the fraction of the
        eight corners of the box shrunk to 80% that are.
    """
    shrunk = box.center + 0.8 * (box.corners() - box.center)
    points = np.vstack([box.center, shrunk])
    through = np.zeros(len(points), dtype=bool)
    for frame in pair.epoch_data(epoch).frames:
        assert frame.depth is not None
        uv, depth = project(points, frame.intrinsics, frame.world_from_camera)
        inside = in_image(uv, depth, frame.intrinsics)
        cols, rows = np.rint(uv[inside]).astype(int).T
        rendered = frame.depth[rows, cols]
        through[inside] |= rendered > depth[inside] + 0.05
    return bool(through[0]), float(through[1:].mean())


@pytest.mark.parametrize("seed", range(60))
def test_changed_objects_are_seen_through_in_the_other_epoch(seed: int) -> None:
    pair = generate_scene_pair(seed, CONFIG)

    checked = 0
    for obj in pair.objects:
        for epoch in Epoch:
            if obj.status == ObjectStatus.UNCHANGED or not obj.exists_in(epoch):
                continue
            other = Epoch.AFTER if epoch == Epoch.BEFORE else Epoch.BEFORE
            center_seen, corner_fraction = seen_through(
                pair, obj.box(epoch), other
            )
            assert center_seen, (obj.object_id, epoch)
            assert corner_fraction >= 0.5, (obj.object_id, epoch)
            checked += 1
    assert checked == 4


def test_unobserved_region_converts_to_the_report_type(
    pair: SyntheticScenePair,
) -> None:
    region = pair.unobserved_region("zone-1")

    assert region.region_id == "zone-1"
    assert region.epoch_seen == Epoch.BEFORE
    np.testing.assert_allclose(
        region.box.to_box().center, pair.unobserved_zone.center
    )


def test_min_move_is_respected_when_raised() -> None:
    config = SceneConfig(
        width=64, height=48, frames_per_epoch=6, min_move_m=0.8
    )

    moved = by_status(generate_scene_pair(3, config), ObjectStatus.MOVED)

    assert moved
    assert all(o.displacement_m >= 0.8 - 1e-9 for o in moved)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        SceneConfig(frames_per_epoch=3)
    with pytest.raises(ValueError):
        SceneConfig(room_size=(4.0, 3.0, 2.5))
    with pytest.raises(ValueError):
        SceneConfig(min_move_m=0.0)
