import numpy as np
import pytest

from spatial_change.domain import Intrinsics
from spatial_change.synthetic.trajectory import (
    Waypoint,
    look_at,
    sees_points,
    sweep,
)

INTRINSICS = Intrinsics(fx=48.0, fy=48.0, cx=31.5, cy=23.5, width=64, height=48)


def waypoints() -> list[Waypoint]:
    return [
        Waypoint(np.array([0.0, 0.0, 1.5]), np.array([4.0, -1.0, 0.5])),
        Waypoint(np.array([0.0, 0.5, 1.5]), np.array([4.0, 0.0, 0.5])),
        Waypoint(np.array([0.0, 1.0, 1.4]), np.array([4.0, 1.0, 0.5])),
    ]


def test_look_at_uses_opencv_axes() -> None:
    pose = look_at(np.array([0.0, 0.0, 1.5]), np.array([3.0, 0.0, 1.5]))

    np.testing.assert_allclose(pose.rotation[:, 2], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(pose.rotation[:, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(pose.rotation[:, 1], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(pose.translation, [0.0, 0.0, 1.5])


def test_look_at_points_the_optical_axis_at_the_target() -> None:
    position, target = np.array([1.0, 2.0, 1.5]), np.array([4.0, 3.0, 0.5])

    pose = look_at(position, target)
    distance = np.linalg.norm(target - position)

    np.testing.assert_allclose(
        pose.apply(np.array([[0.0, 0.0, distance]]))[0], target, atol=1e-9
    )
    assert pose.rotation[2, 0] == pytest.approx(0.0, abs=1e-12)  # no roll


def test_look_at_rejects_coincident_and_vertical_views() -> None:
    with pytest.raises(ValueError, match="must differ"):
        look_at(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="vertical"):
        look_at(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 3.0]))


def test_sweep_without_jitter_hits_first_and_last_waypoint() -> None:
    poses = sweep(waypoints(), 9, np.random.default_rng(0), 0.0, 0.0)

    assert len(poses) == 9
    np.testing.assert_allclose(poses[0].translation, [0.0, 0.0, 1.5])
    np.testing.assert_allclose(poses[4].translation, [0.0, 0.5, 1.5])
    np.testing.assert_allclose(poses[-1].translation, [0.0, 1.0, 1.4])
    forward_end = poses[-1].rotation[:, 2]
    assert forward_end[1] == pytest.approx(0.0, abs=1e-9)


def test_sweep_is_seeded_and_jitter_is_small() -> None:
    a = sweep(waypoints(), 8, np.random.default_rng(1), 0.01, 0.03)
    b = sweep(waypoints(), 8, np.random.default_rng(1), 0.01, 0.03)
    c = sweep(waypoints(), 8, np.random.default_rng(2), 0.01, 0.03)
    clean = sweep(waypoints(), 8, np.random.default_rng(1), 0.0, 0.0)

    for pa, pb in zip(a, b, strict=True):
        np.testing.assert_array_equal(pa.matrix, pb.matrix)
    assert any(
        not np.array_equal(pa.matrix, pc.matrix)
        for pa, pc in zip(a, c, strict=True)
    )
    for pa, pk in zip(a, clean, strict=True):
        assert np.linalg.norm(pa.translation - pk.translation) < 0.06


def test_sweep_needs_two_frames() -> None:
    with pytest.raises(ValueError):
        sweep(waypoints(), 1, np.random.default_rng(0), 0.0, 0.0)


def test_sees_points_matches_frustum() -> None:
    pose = look_at(np.array([0.0, 0.0, 1.5]), np.array([3.0, 0.0, 1.5]))
    points = np.array(
        [
            [3.0, 0.0, 1.5],  # straight ahead
            [-3.0, 0.0, 1.5],  # behind
            [3.0, 5.0, 1.5],  # far left, outside the image
            [3.0, 0.0, 9.0],  # far above, outside the image
        ]
    )

    seen = sees_points(points, INTRINSICS, pose)

    assert seen.tolist() == [True, False, False, False]
