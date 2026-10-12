import dataclasses
import itertools

import numpy as np
import pytest
from rotations import rotation_z

from spatial_change.domain import (
    SE3,
    Epoch,
    EpochReconstruction,
    Frame,
    Intrinsics,
    ObjectInstance,
    OrientedBox,
    Sim3,
)


def test_axis_aligned_box_corners_and_volume() -> None:
    box = OrientedBox(
        center=np.array([1.0, 2.0, 3.0]),
        extent=np.array([2.0, 4.0, 6.0]),
        rotation=np.eye(3),
    )

    expected = {
        (1.0 + dx, 2.0 + dy, 3.0 + dz)
        for dx, dy, dz in itertools.product(
            (-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)
        )
    }
    corners = box.corners()
    assert corners.shape == (8, 3)
    assert {tuple(c) for c in corners.tolist()} == expected
    assert box.volume == pytest.approx(48.0)


def test_contains_rotated_box() -> None:
    # A long thin box along the x axis, rotated 45 degrees about z.
    box = OrientedBox(
        center=np.array([1.0, 1.0, 0.0]),
        extent=np.array([4.0, 0.2, 1.0]),
        rotation=rotation_z(45.0),
    )
    diagonal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    points = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0] + 1.9 * diagonal,
            [1.0, 1.0, 0.4],
            [3.0, 1.0, 0.0],
            [1.0, 1.0, 0.0] + 2.1 * diagonal,
            [1.0, 1.0, 0.6],
        ]
    )

    np.testing.assert_array_equal(
        box.contains(points), [True, True, True, False, False, False]
    )


def test_corners_of_rotated_box_are_inside() -> None:
    box = OrientedBox(
        center=np.array([0.5, -1.0, 2.0]),
        extent=np.array([1.0, 2.0, 3.0]),
        rotation=rotation_z(30.0),
    )
    shrunk = box.center + 0.999 * (box.corners() - box.center)

    assert box.contains(shrunk).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extent": np.array([1.0, 0.0, 1.0])},
        {"extent": np.array([1.0, 1.0])},
        {"center": np.array([0.0, np.nan, 0.0])},
        {"rotation": np.diag([1.0, 1.0, -1.0])},
    ],
    ids=["zero-extent", "extent-shape", "nan-center", "reflection"],
)
def test_invalid_box_raises(kwargs: dict[str, np.ndarray]) -> None:
    params = {
        "center": np.zeros(3),
        "extent": np.ones(3),
        "rotation": np.eye(3),
    }
    params.update(kwargs)
    with pytest.raises(ValueError):
        OrientedBox(**params)


def test_box_arrays_are_read_only() -> None:
    box = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))

    assert not box.center.flags.writeable
    assert not box.extent.flags.writeable
    assert not box.rotation.flags.writeable


def _frame(epoch: Epoch) -> Frame:
    return Frame(
        epoch=epoch,
        index=0,
        timestamp_s=0.0,
        intrinsics=Intrinsics(1.0, 1.0, 0.0, 0.0, 2, 2),
        world_from_camera=SE3.identity(),
    )


def test_epoch_reconstruction_accepts_matching_lengths() -> None:
    recon = EpochReconstruction(
        epoch=Epoch.BEFORE,
        frames=[_frame(Epoch.BEFORE)],
        points=np.zeros((5, 3)),
        colors=np.zeros((5, 3), np.uint8),
        confidence=np.ones(5, np.float32),
    )

    assert len(recon.points) == 5


@pytest.mark.parametrize(
    "arrays",
    [
        {"colors": np.zeros((4, 3), np.uint8)},
        {"confidence": np.ones(6, np.float32)},
        {"points": np.zeros((5, 2))},
        {"colors": np.zeros((5, 3), np.float32)},
    ],
    ids=["colors-length", "confidence-length", "points-shape", "colors-dtype"],
)
def test_epoch_reconstruction_mismatch_raises(
    arrays: dict[str, np.ndarray],
) -> None:
    params = {
        "points": np.zeros((5, 3)),
        "colors": np.zeros((5, 3), np.uint8),
        "confidence": np.ones(5, np.float32),
    }
    params.update(arrays)
    with pytest.raises(ValueError):
        EpochReconstruction(epoch=Epoch.BEFORE, frames=[], **params)


def test_epoch_reconstruction_rejects_frames_from_other_epoch() -> None:
    with pytest.raises(ValueError):
        EpochReconstruction(
            epoch=Epoch.BEFORE,
            frames=[_frame(Epoch.AFTER)],
            points=np.zeros((0, 3)),
            colors=np.zeros((0, 3), np.uint8),
            confidence=np.zeros(0, np.float32),
        )


def test_object_instance_defaults_and_validation() -> None:
    box = OrientedBox(np.zeros(3), np.ones(3), np.eye(3))
    obj = ObjectInstance(
        object_id="before-0",
        epoch=Epoch.BEFORE,
        points=np.zeros((3, 3)),
        box=box,
    )

    assert obj.label is None
    assert obj.label_score is None
    assert obj.feature is None
    with pytest.raises(ValueError):
        ObjectInstance("x", Epoch.AFTER, np.zeros((3, 2)), box)


def test_transforming_an_epoch_moves_points_poses_and_depth_together() -> None:
    depth = np.full((2, 2), 3.0, np.float32)
    frame = dataclasses.replace(_frame(Epoch.BEFORE), depth=depth)
    colors = np.zeros((2, 3), np.uint8)
    recon = EpochReconstruction(
        epoch=Epoch.BEFORE,
        frames=[frame],
        points=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]]),
        colors=colors,
        confidence=np.ones(2, np.float32),
    )
    world = Sim3(2.0, rotation_z(90.0), np.array([1.0, 2.0, 3.0]))

    moved = recon.transformed(world)

    np.testing.assert_allclose(moved.points, world.apply(recon.points))
    np.testing.assert_allclose(
        moved.frames[0].world_from_camera.translation,
        world.transform_pose(frame.world_from_camera).translation,
    )
    assert moved.frames[0].depth is not None
    np.testing.assert_allclose(moved.frames[0].depth, 2.0 * depth)
    assert moved.colors is colors
    assert moved.epoch is Epoch.BEFORE
    assert recon.frames[0] is frame
