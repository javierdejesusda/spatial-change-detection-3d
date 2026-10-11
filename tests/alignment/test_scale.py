import numpy as np
import pytest
from rooms import sample_room
from rotations import random_rotation

from spatial_change.alignment.scale import (
    RoomAlignment,
    ScalePolicy,
    estimate_scale,
    gravity_align,
)
from spatial_change.domain import SE3, Sim3
from spatial_change.geometry.floor import FloorConfig
from spatial_change.geometry.projection import unproject_depth
from spatial_change.synthetic.scene import SceneConfig, generate_scene_pair
from spatial_change.synthetic.trajectory import look_at


def test_model_policy_keeps_the_reconstruction_units() -> None:
    assert estimate_scale([0.3, 9.0], ScalePolicy.MODEL) == 1.0


def test_policy_given_as_a_plain_string_is_coerced() -> None:
    heights = np.array([0.7, 0.7, 0.7])

    assert estimate_scale(heights, "model") == 1.0
    assert estimate_scale(
        heights,
        "camera_height",
    ) == pytest.approx(2.0)


def test_unknown_policy_string_raises() -> None:
    with pytest.raises(ValueError, match="turbo"):
        estimate_scale([1.0], "turbo")


def test_camera_height_policy_scales_median_height_to_the_assumption() -> None:
    heights = np.array([0.7, 0.7, 0.7, 0.72, 5.0])  # one outlier camera

    scale = estimate_scale(heights, ScalePolicy.CAMERA_HEIGHT)

    assert scale == pytest.approx(1.4 / 0.7)
    assert estimate_scale(heights, ScalePolicy.CAMERA_HEIGHT, 1.5) == (
        pytest.approx(1.5 / 0.7)
    )


@pytest.mark.parametrize(
    ("heights", "policy", "assumed"),
    [
        ([], ScalePolicy.CAMERA_HEIGHT, 1.4),
        ([0.0, 0.0, 1.0], ScalePolicy.CAMERA_HEIGHT, 1.4),
        ([-1.0, -2.0], ScalePolicy.CAMERA_HEIGHT, 1.4),
        ([np.nan, 1.0], ScalePolicy.CAMERA_HEIGHT, 1.4),
        ([[1.0, 2.0]], ScalePolicy.CAMERA_HEIGHT, 1.4),
        ([1.0], ScalePolicy.CAMERA_HEIGHT, 0.0),
        ([1.0], ScalePolicy.MODEL, -1.0),
        ([], ScalePolicy.MODEL, 1.4),
    ],
)
def test_estimate_scale_rejects_invalid_inputs(
    heights: list[float], policy: ScalePolicy, assumed: float
) -> None:
    with pytest.raises(ValueError):
        estimate_scale(heights, policy, assumed)


def _world_cameras(world_from_room: Sim3, height: float) -> list[SE3]:
    # Cameras 1 m apart along room x at the given height, looking along +y.
    return [
        world_from_room.transform_pose(
            look_at((x, 1.0, height), (x, 2.0, height))
        )
        for x in (1.0, 2.0, 3.0)
    ]


@pytest.mark.parametrize("scale", [0.5, 20.0, 0.05])
def test_camera_height_policy_recovers_the_room_frame_and_scale(
    scale: float,
) -> None:
    rng = np.random.default_rng(0)
    rotation = random_rotation(rng)
    shift = rng.uniform(-3, 3, 3)
    world_from_room = Sim3(scale, rotation, shift)
    room = sample_room(rng)
    world = world_from_room.apply(room)
    cameras = _world_cameras(world_from_room, height=1.2)

    alignment = gravity_align(world, cameras, ScalePolicy.CAMERA_HEIGHT)

    assert isinstance(alignment, RoomAlignment)
    assert alignment.policy is ScalePolicy.CAMERA_HEIGHT
    assert alignment.scale == pytest.approx(1.4 / (1.2 * scale), rel=0.02)
    in_room = alignment.room_from_world.apply(world)
    floor_pts = np.abs(room[:, 2]) < 0.01
    assert np.abs(in_room[floor_pts, 2]).max() < 0.03
    assert alignment.floor.normal @ (rotation @ [0, 0, 1]) > 0.999
    centers = np.array([pose.translation for pose in cameras])
    heights = alignment.room_from_world.apply(centers)[:, 2]
    assert np.median(heights) == pytest.approx(1.4, rel=1e-6)
    # The returned floor stays in world units.
    assert alignment.floor.height(centers).mean() == pytest.approx(
        1.2 * scale, rel=0.02
    )
    # Camera forward is room +y: the first camera faces +y.
    first = alignment.room_from_world.transform_pose(cameras[0])
    np.testing.assert_allclose(first.rotation[:, 2], [0, 1, 0], atol=0.03)


def test_model_policy_is_rigid_and_accepts_a_plain_string() -> None:
    room = sample_room(np.random.default_rng(1))
    cameras = _world_cameras(Sim3(1.0, np.eye(3), np.zeros(3)), height=1.5)
    config = FloorConfig(seed=3)

    alignment = gravity_align(room, cameras, "model", floor_config=config)
    default = gravity_align(room, cameras, floor_config=config)

    assert alignment.policy is ScalePolicy.MODEL
    assert default.policy is ScalePolicy.MODEL
    assert alignment.scale == 1.0
    assert abs(alignment.floor.offset) < 0.01
    assert alignment.floor.normal @ [0, 0, 1] > 0.999
    with pytest.raises(ValueError, match="turbo"):
        gravity_align(room, cameras, "turbo")


def test_gravity_align_rejects_missing_cameras() -> None:
    room = sample_room(np.random.default_rng(2))

    with pytest.raises(ValueError, match="camera"):
        gravity_align(room, [])


def test_synthetic_scene_is_returned_to_z_up_with_the_floor_at_zero() -> None:
    config = SceneConfig(width=48, height=36, frames_per_epoch=6)
    pair = generate_scene_pair(0, config)
    rng = np.random.default_rng(10)
    rotation = random_rotation(rng)
    shift = rng.uniform(-4, 4, 3)
    world_from_old = SE3(rotation, shift)
    clouds, cameras = [], []
    for frame in pair.before.frames:
        assert frame.depth is not None
        clouds.append(
            world_from_old.apply(
                unproject_depth(
                    frame.depth, frame.intrinsics, frame.world_from_camera
                )
            )
        )
        cameras.append(world_from_old.compose(frame.world_from_camera))
    cloud = np.concatenate(clouds)

    alignment = gravity_align(cloud, cameras)

    back = alignment.room_from_world.apply(cloud)
    original = world_from_old.apply_inverse(cloud)
    on_floor = np.abs(original[:, 2]) < 1e-3
    assert on_floor.sum() > 100
    assert np.abs(back[on_floor, 2]).max() < 0.03
    assert back[:, 2].min() > -0.05
    assert back[:, 2].max() < config.room_size[2] + 0.05
    up = alignment.room_from_world.rotation @ alignment.floor.normal
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-9)
    assert alignment.floor.normal @ (rotation @ [0, 0, 1]) > 0.999


@pytest.mark.parametrize(
    "policy", [ScalePolicy.MODEL, ScalePolicy.CAMERA_HEIGHT]
)
def test_cameras_below_the_floor_are_rejected(policy: ScalePolicy) -> None:
    room = sample_room(np.random.default_rng(5))
    # An OpenGL-style camera has y up, so the up hint points down.
    gl_rotation = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    cameras = [
        SE3(gl_rotation, np.array([x, 1.0, 1.2])) for x in (1.0, 2.0, 3.0)
    ]

    with pytest.raises(ValueError, match="camera height"):
        gravity_align(room, cameras, policy)
