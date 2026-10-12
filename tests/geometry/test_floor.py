import time
import tracemalloc

import numpy as np
import pytest
from numpy.typing import NDArray
from rooms import (
    TABLE_HEIGHT,
    face,
    sample_faces,
    sample_room,
)
from rotations import angle_deg, pitched_cameras, random_rotation

from spatial_change.domain import SE3
from spatial_change.geometry import floor as floor_module
from spatial_change.geometry.floor import (
    FloorConfig,
    FloorNotFoundError,
    FloorPlane,
    camera_up_hint,
    estimate_up,
    fit_floor,
    floor_frame,
)
from spatial_change.synthetic.trajectory import look_at

Z = np.array([0.0, 0.0, 1.0])


def _tilted_up(
    rng: np.random.Generator, rotation: NDArray[np.float64]
) -> NDArray[np.float64]:
    # A camera-derived hint is never exact: tilt it by up to 8 degrees.
    jitter = rotation @ (Z + rng.normal(scale=0.1, size=3) * [1, 1, 0])
    return jitter / np.linalg.norm(jitter)


def _camera(center: tuple[float, float, float]) -> SE3:
    # Level camera looking along +x: its image y axis (down) is world -z, so
    # its up vector is +z.
    return look_at(center, (center[0] + 1.0, center[1], center[2]))


LEVEL = _camera((0.0, 0.0, 0.0)).rotation


def test_up_hint_is_the_mean_camera_up() -> None:
    poses = [_camera((0, 0, 1.4)), _camera((1, 0, 1.5))]

    np.testing.assert_allclose(camera_up_hint(poses), Z, atol=1e-12)


def test_up_hint_averages_pitched_cameras_and_normalizes() -> None:
    def pitched(deg: float) -> SE3:
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return look_at((0, 0, 1.4), (c, 0, 1.4 + s))

    up = camera_up_hint([pitched(30.0), pitched(-30.0)])

    np.testing.assert_allclose(up, Z, atol=1e-12)


def test_up_hint_rejects_empty_and_cancelling_poses() -> None:
    flipped = LEVEL @ np.diag([1.0, -1.0, -1.0])  # upside down camera

    with pytest.raises(ValueError, match="at least one"):
        camera_up_hint([])
    with pytest.raises(ValueError, match="zero"):
        camera_up_hint([_camera((0, 0, 0)), SE3(flipped, np.zeros(3))])


@pytest.mark.parametrize("seed", range(8))
def test_floor_recovered_under_random_world_rotation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rotation = random_rotation(rng)
    shift = rng.uniform(-5, 5, 3)
    room = sample_room(rng)
    world_from_room = SE3(rotation, shift)
    cloud = world_from_room.apply(room)

    floor = fit_floor(cloud, _tilted_up(rng, rotation))

    assert angle_deg(floor.normal, rotation @ Z) < 0.3
    on_floor = np.abs(room[:, 2]) < 0.01
    assert np.abs(floor.height(cloud[on_floor])).max() < 0.02
    above = world_from_room.apply([[3.0, 2.0, 1.0]])
    assert floor.height(above)[0] == pytest.approx(1.0, abs=0.01)
    assert floor.inlier_count > 0.9 * on_floor.sum()


@pytest.mark.parametrize("scale", [1e-3, 1e-6])
def test_floor_is_found_in_a_tiny_world(scale: float) -> None:
    room = sample_room(np.random.default_rng(0))
    config = FloorConfig(distance_threshold=0.02 * scale)

    floor = fit_floor(scale * room, Z, config)

    assert floor.normal @ Z > 0.999
    assert abs(floor.offset) < 0.01 * scale
    on_floor = np.abs(room[:, 2]) < 0.01
    assert floor.inlier_count > 0.9 * on_floor.sum()


def test_floor_normal_points_toward_the_up_hint() -> None:
    rng = np.random.default_rng(0)
    room = sample_room(rng)

    down = fit_floor(room * [1, 1, -1], -Z)
    up = fit_floor(room, Z)

    assert down.normal @ -Z > 0.99
    assert up.normal @ Z > 0.99


def test_lowest_plane_beats_a_larger_table_top_and_ceiling() -> None:
    rng = np.random.default_rng(1)
    floor_patch = face((0, 0, 0), (2.0, 0, 0), (0, 2.0, 0))
    table = face((0, 0, TABLE_HEIGHT), (5.0, 0, 0), (0, 5.0, 0))
    ceiling = face((0, 0, 2.5), (6.0, 0, 0), (0, 6.0, 0))
    cloud = sample_faces(rng, [floor_patch, table, ceiling], density=100.0)

    floor = fit_floor(cloud, Z)

    assert floor.height(np.array([[1.0, 1.0, 0.0]]))[0] == pytest.approx(
        0.0, abs=0.01
    )
    assert floor.inlier_count < 0.2 * len(cloud)


def test_wall_is_never_chosen_as_the_floor() -> None:
    rng = np.random.default_rng(2)
    # The wall reaches below the floor patch, so only the tilt rule (not the
    # lowest-plane rule) can keep it from winning.
    wall = face((0, 0, -3.0), (8.0, 0, 0), (0, 0, 4.0))
    floor_patch = face((0, 0, 0), (0, 3.0, 0), (3.0, 0, 0))
    cloud = sample_faces(rng, [wall, floor_patch], density=100.0)

    floor = fit_floor(cloud, Z)

    assert floor.normal @ Z > 0.99


def test_cloud_without_a_horizontal_plane_raises() -> None:
    rng = np.random.default_rng(3)
    walls = [
        face((0, 0, 0), (6.0, 0, 0), (0, 0, 3.0)),
        face((0, 0, 0), (0, 6.0, 0), (0, 0, 3.0)),
    ]
    cloud = sample_faces(rng, walls, density=100.0)

    with pytest.raises(FloorNotFoundError, match="no horizontal plane"):
        fit_floor(cloud, Z)


def test_small_support_floor_is_rejected() -> None:
    rng = np.random.default_rng(4)
    patch = face((0, 0, 0), (1.0, 0, 0), (0, 1.0, 0))
    wall = face((0, 0, 0), (8.0, 0, 0), (0, 0, 8.0))
    cloud = sample_faces(rng, [patch, wall], density=100.0)

    with pytest.raises(ValueError, match="no horizontal plane"):
        fit_floor(cloud, Z, FloorConfig(min_support_fraction=0.2))


def test_fit_is_deterministic_for_a_seed() -> None:
    cloud = sample_room(np.random.default_rng(5))

    first = fit_floor(cloud, Z, FloorConfig(seed=7))
    second = fit_floor(cloud, Z, FloorConfig(seed=7))

    np.testing.assert_array_equal(first.normal, second.normal)
    assert (first.offset, first.inlier_count) == (
        second.offset,
        second.inlier_count,
    )


def test_large_cloud_is_fitted_quickly_in_bounded_memory() -> None:
    rng = np.random.default_rng(8)
    faces = [
        face((0, 0, 0), (6.0, 0, 0), (0, 5.0, 0)),
        face((0, 0, TABLE_HEIGHT), (1.4, 0, 0), (0, 1.0, 0)),
        face((0, 0, 2.5), (6.0, 0, 0), (0, 5.0, 0)),
        face((0, 0, 0), (6.0, 0, 0), (0, 0, 2.5)),
    ]
    cloud = sample_faces(rng, faces, count=400_000)

    tracemalloc.start()
    start = time.perf_counter()
    floor = fit_floor(cloud, Z)
    elapsed = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert floor.normal @ Z > 0.999
    assert abs(floor.offset) < 0.005
    assert floor.inlier_count > 0.35 * len(cloud)
    assert elapsed < 5.0
    # Scoring must not materialize points x hypotheses temporaries, which
    # take more than 200 MB at this size.
    assert peak < 100 * 2**20


def test_a_round_without_hypotheses_does_not_end_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = sample_room(np.random.default_rng(9))
    real = floor_module._hypotheses
    calls: list[int] = []

    def flaky(*args: object, **kwargs: object) -> tuple[object, object]:
        calls.append(1)
        if len(calls) == 1:
            return np.empty((0, 3)), np.empty(0)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(floor_module, "_hypotheses", flaky)

    floor = fit_floor(cloud, Z)

    assert len(calls) >= 2
    assert abs(floor.offset) < 0.01


def test_fit_floor_rejects_invalid_inputs() -> None:
    cloud = sample_room(np.random.default_rng(6))

    with pytest.raises(ValueError, match="shape"):
        fit_floor(cloud[:, :2], Z)
    with pytest.raises(ValueError, match="at least 3"):
        fit_floor(cloud[:2], Z)
    with pytest.raises(ValueError, match="finite"):
        fit_floor(np.where(cloud > 100, cloud, np.nan), Z)
    with pytest.raises(ValueError, match="up_hint"):
        fit_floor(cloud, np.zeros(3))
    with pytest.raises(ValueError, match="up_hint"):
        fit_floor(cloud, [0.0, 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance_threshold": 0.0},
        {"max_tilt_deg": 0.0},
        {"max_tilt_deg": 90.0},
        {"iterations": 0},
        {"max_planes": 0},
        {"min_support_fraction": 0.0},
        {"min_support_fraction": 1.5},
    ],
)
def test_floor_config_validates(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FloorConfig(**kwargs)  # type: ignore[arg-type]


def _floor(normal: NDArray[np.float64], offset: float) -> FloorPlane:
    return FloorPlane(normal / np.linalg.norm(normal), offset, 100)


@pytest.mark.parametrize("seed", range(5))
def test_room_frame_puts_floor_at_zero_and_up_on_z(seed: int) -> None:
    rng = np.random.default_rng(seed)
    normal = random_rotation(rng) @ Z
    floor = _floor(normal, float(rng.uniform(-3, 3)))
    forward = rng.normal(size=3)

    frame = floor_frame(floor, forward)

    on_plane = rng.normal(size=(20, 3))
    on_plane -= np.outer(floor.height(on_plane), floor.normal)
    assert np.abs(frame.apply(on_plane)[:, 2]).max() < 1e-9
    np.testing.assert_allclose(frame.rotation @ floor.normal, Z, atol=1e-12)
    assert np.linalg.det(frame.rotation) == pytest.approx(1.0)


def test_room_frame_maps_forward_to_plus_y_with_x_to_the_right() -> None:
    floor = _floor(np.array([0.0, 0.0, 1.0]), -0.3)
    forward = np.array([1.0, 1.0, 0.7])  # tilted forward, projects to (1, 1)

    frame = floor_frame(floor, forward)

    y_dir = frame.rotation @ np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    np.testing.assert_allclose(y_dir, [0.0, 1.0, 0.0], atol=1e-12)
    x_axis_world = frame.rotation.T[:, 0]
    np.testing.assert_allclose(
        np.cross(x_axis_world, frame.rotation.T[:, 1]),
        frame.rotation.T[:, 2],
        atol=1e-12,
    )
    # Facing +y (here along (1, 1)) means x points to the right of the view.
    assert x_axis_world @ np.array([1.0, -1.0, 0.0]) > 0


def test_room_origin_is_the_projection_of_the_world_origin() -> None:
    floor = _floor(np.array([0.0, 0.0, 1.0]), -2.0)  # floor at world z = 2

    frame = floor_frame(floor, np.array([0.0, 1.0, 0.0]))

    np.testing.assert_allclose(
        frame.apply([[0.0, 0.0, 2.0]]), [[0, 0, 0]], atol=1e-12
    )
    np.testing.assert_allclose(
        frame.apply([[0.0, 0.0, 0.0]]), [[0, 0, -2.0]], atol=1e-12
    )


@pytest.mark.parametrize(
    ("normal", "forward", "expected_y"),
    [
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        # World x is the normal itself, so the world y axis is the fallback.
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ],
)
def test_degenerate_forward_falls_back_to_a_world_axis(
    normal: list[float], forward: list[float], expected_y: list[float]
) -> None:
    frame = floor_frame(_floor(np.array(normal), 0.0), np.array(forward))

    np.testing.assert_allclose(
        frame.rotation.T @ [0, 1, 0], expected_y, atol=1e-12
    )
    np.testing.assert_allclose(frame.rotation @ normal, Z, atol=1e-12)


def test_floor_frame_rejects_a_malformed_forward_hint() -> None:
    with pytest.raises(ValueError, match="forward_hint"):
        floor_frame(_floor(Z.copy(), 0.0), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="forward_hint"):
        floor_frame(_floor(Z.copy(), 0.0), np.array([np.nan, 0.0, 1.0]))


def test_estimate_up_ignores_the_pitch_the_mean_up_cannot() -> None:
    poses = pitched_cameras(40.0, list(np.linspace(-60.0, 60.0, 25)))

    assert angle_deg(camera_up_hint(poses), Z) > 20.0
    assert angle_deg(estimate_up(poses), Z) < 0.5


def test_estimate_up_works_for_a_capture_turning_in_place() -> None:
    yaws = list(np.linspace(0.0, 300.0, 40))
    poses = pitched_cameras(20.0, yaws, spread=0.01)

    assert angle_deg(estimate_up(poses), Z) < 0.5


def test_estimate_up_falls_back_without_yaw_spread() -> None:
    poses = [
        look_at((0, 0, 1.5), (np.cos(p), 0, 1.5 - np.sin(p)))
        for p in np.radians(np.linspace(10.0, 50.0, 9))
    ]

    np.testing.assert_array_equal(estimate_up(poses), camera_up_hint(poses))


def test_estimate_up_falls_back_when_roll_rivals_the_yaw_spread() -> None:
    rolls = np.radians([-35.0, 35.0, -35.0, 35.0, -35.0, 35.0])
    poses = []
    for yaw, roll in zip(
        np.radians([-35, -20, -5, 5, 20, 35]), rolls, strict=True
    ):
        base = look_at((0, 0, 1.5), (np.cos(yaw), np.sin(yaw), 1.5))
        c, s = np.cos(roll), np.sin(roll)
        about_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        poses.append(SE3(base.rotation @ about_z, base.translation))

    np.testing.assert_array_equal(estimate_up(poses), camera_up_hint(poses))


@pytest.mark.parametrize("pitch_deg", [40.0, -40.0])
def test_estimate_up_points_with_the_mean_up(pitch_deg: float) -> None:
    poses = pitched_cameras(pitch_deg, list(np.linspace(-60.0, 60.0, 25)))

    assert estimate_up(poses) @ camera_up_hint(poses) > 0.0
    assert estimate_up(poses) @ Z > 0.99


def test_estimate_up_rejects_empty_poses() -> None:
    with pytest.raises(ValueError, match="at least one"):
        estimate_up([])
