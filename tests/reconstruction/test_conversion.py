import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.typing import NDArray
from rotations import random_rotation

from spatial_change.domain import SE3
from spatial_change.reconstruction.conversion import (
    intrinsics_from_local_points,
    nearest_rotation,
    to_reconstruction_result,
    working_size,
)

PIXEL_LIMIT = 153_000


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1280, 720, (518, 294)),
        (1920, 1080, (518, 294)),
        (640, 360, (518, 294)),
        (720, 1280, (294, 518)),
        (518, 294, (518, 294)),
    ],
)
def test_working_size_matches_pi3_rule(
    width: int, height: int, expected: tuple[int, int]
) -> None:
    assert working_size(width, height, PIXEL_LIMIT) == expected


def test_working_size_upscales_small_images_to_the_limit() -> None:
    assert working_size(160, 120, PIXEL_LIMIT) == (448, 336)


def test_working_size_shrinks_the_rounded_grid_to_fit_the_limit() -> None:
    # Rounding gives 28 x 28 patches (153664 px), one row over the limit.
    assert working_size(1000, 1000, PIXEL_LIMIT) == (392, 378)


def test_working_size_keeps_at_least_one_patch_per_side() -> None:
    assert working_size(100, 100, 100) == (14, 14)
    assert working_size(10000, 10, PIXEL_LIMIT) == (12376, 14)


@pytest.mark.parametrize(
    ("width", "height", "pixel_limit"),
    [(0, 720, PIXEL_LIMIT), (1280, -1, PIXEL_LIMIT), (1280, 720, 0)],
)
def test_working_size_rejects_non_positive_arguments(
    width: int, height: int, pixel_limit: int
) -> None:
    with pytest.raises(ValueError):
        working_size(width, height, pixel_limit)


@given(
    width=st.integers(100, 4000),
    height=st.integers(100, 4000),
    pixel_limit=st.integers(20_000, 500_000),
)
def test_working_size_is_a_patch_grid_within_the_limit(
    width: int, height: int, pixel_limit: int
) -> None:
    assume(1 / 8 <= width / height <= 8)

    out_w, out_h = working_size(width, height, pixel_limit)

    assert out_w % 14 == 0
    assert out_h % 14 == 0
    assert out_w >= 14
    assert out_h >= 14
    assert out_w * out_h <= pixel_limit


def _point_map(
    fx: float,
    fy: float,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Camera-frame points whose pixels project back onto a pixel grid."""
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    z = rng.uniform(1.0, 5.0, size=(height, width))
    x = (cols - (width - 1) / 2) / fx * z
    y = (rows - (height - 1) / 2) / fy * z
    return np.stack([x, y, z], axis=-1)


def test_intrinsics_are_recovered_from_a_clean_point_map() -> None:
    rng = np.random.default_rng(0)
    points = _point_map(412.0, 398.0, 80, 48, rng)

    k = intrinsics_from_local_points(points, np.ones((48, 80)), 80, 48)

    assert k.fx == pytest.approx(412.0, rel=1e-4)
    assert k.fy == pytest.approx(398.0, rel=1e-4)


def test_intrinsics_fix_the_principal_point_at_the_image_center() -> None:
    rng = np.random.default_rng(1)
    points = _point_map(300.0, 300.0, 64, 36, rng)

    k = intrinsics_from_local_points(points, np.ones((36, 64)), 64, 36)

    assert (k.cx, k.cy) == (31.5, 17.5)
    assert (k.width, k.height) == (64, 36)


def test_intrinsics_tolerate_noise_and_varying_confidence() -> None:
    rng = np.random.default_rng(2)
    points = _point_map(500.0, 480.0, 96, 54, rng)
    points += rng.normal(scale=2e-4, size=points.shape)
    confidence = rng.uniform(0.2, 1.0, size=(54, 96))

    k = intrinsics_from_local_points(points, confidence, 96, 54)

    assert k.fx == pytest.approx(500.0, rel=1e-4)
    assert k.fy == pytest.approx(480.0, rel=1e-4)


def test_intrinsics_are_not_biased_by_point_noise() -> None:
    rng = np.random.default_rng(6)
    points = _point_map(500.0, 480.0, 160, 90, rng)
    points += rng.normal(scale=0.05, size=points.shape)

    k = intrinsics_from_local_points(points, np.ones((90, 160)), 160, 90)

    assert k.fx == pytest.approx(500.0, rel=1e-2)
    assert k.fy == pytest.approx(480.0, rel=1e-2)


def test_intrinsics_ignore_invalid_points() -> None:
    rng = np.random.default_rng(3)
    points = _point_map(350.0, 360.0, 64, 48, rng)
    points[:12, :, 0] = 1e3
    points[:16, :, 1] = 1e3
    points[:4, :, 2] = 0.0
    points[4:8, :, 2] = -1.0
    points[8:12, :, 2] = np.nan
    points[12:16, :, 0] = np.inf

    k = intrinsics_from_local_points(points, np.ones((48, 64)), 64, 48)

    assert k.fx == pytest.approx(350.0, rel=1e-4)
    assert k.fy == pytest.approx(360.0, rel=1e-4)


def test_intrinsics_ignore_points_without_confidence() -> None:
    rng = np.random.default_rng(4)
    points = _point_map(350.0, 360.0, 64, 48, rng)
    confidence = np.ones((48, 64))
    points[:16, :, :2] *= 3.0
    confidence[:16] = 0.0
    confidence[16:20, :10] = np.nan

    k = intrinsics_from_local_points(points, confidence, 64, 48)

    assert k.fx == pytest.approx(350.0, rel=1e-4)
    assert k.fy == pytest.approx(360.0, rel=1e-4)


def test_intrinsics_weight_points_by_confidence() -> None:
    rng = np.random.default_rng(5)
    points = _point_map(350.0, 350.0, 64, 48, rng)
    points[:24, :, :2] *= 2.0
    heavy = np.ones((48, 64))
    heavy[24:] = 0.001
    light = np.ones((48, 64))
    light[:24] = 0.001

    k_heavy = intrinsics_from_local_points(points, heavy, 64, 48)
    k_light = intrinsics_from_local_points(points, light, 64, 48)

    assert k_light.fx == pytest.approx(350.0, rel=0.01)
    assert k_heavy.fx == pytest.approx(175.0, rel=0.01)


def test_intrinsics_weight_valid_points_equally_without_confidence() -> None:
    rng = np.random.default_rng(8)
    points = _point_map(320.0, 310.0, 64, 48, rng)
    points[:8, :, 2] = -1.0

    unweighted = intrinsics_from_local_points(points, np.ones((48, 64)), 64, 48)
    for confidence in (np.zeros((48, 64)), np.full((48, 64), np.nan)):
        k = intrinsics_from_local_points(points, confidence, 64, 48)

        assert k == unweighted
    assert unweighted.fx == pytest.approx(320.0, rel=1e-4)
    assert unweighted.fy == pytest.approx(310.0, rel=1e-4)


@pytest.mark.parametrize("confidence_value", [1.0, 0.0])
def test_intrinsics_reject_a_map_without_usable_points(
    confidence_value: float,
) -> None:
    points = np.zeros((6, 8, 3))

    with pytest.raises(ValueError, match="valid points"):
        intrinsics_from_local_points(
            points, np.full((6, 8), confidence_value), 8, 6
        )


def test_intrinsics_reject_mismatched_shapes() -> None:
    points = np.ones((6, 8, 3))

    with pytest.raises(ValueError, match="local_points"):
        intrinsics_from_local_points(points, np.ones((6, 8)), 9, 6)
    with pytest.raises(ValueError, match="confidence"):
        intrinsics_from_local_points(points, np.ones((6, 9)), 8, 6)


def test_intrinsics_reject_a_mirrored_point_map() -> None:
    rng = np.random.default_rng(7)
    points = _point_map(300.0, 300.0, 32, 24, rng)
    points[..., 0] *= -1.0

    with pytest.raises(ValueError, match="pinhole"):
        intrinsics_from_local_points(points, np.ones((24, 32)), 32, 24)


def test_intrinsics_reject_points_on_the_optical_axis() -> None:
    points = np.zeros((24, 32, 3))
    points[..., 2] = 2.0

    with pytest.raises(ValueError, match="pinhole"):
        intrinsics_from_local_points(points, np.ones((24, 32)), 32, 24)


def test_nearest_rotation_keeps_a_proper_rotation() -> None:
    rotation = random_rotation(np.random.default_rng(0))

    np.testing.assert_allclose(nearest_rotation(rotation), rotation, atol=1e-12)


@given(st.integers(0, 2**32 - 1))
def test_nearest_rotation_repairs_low_precision_rotations(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rotation = random_rotation(rng)
    # bfloat16 keeps about three significant digits.
    noisy = rotation * (1.0 + rng.uniform(-4e-3, 4e-3, size=(3, 3)))

    repaired = nearest_rotation(noisy)

    SE3(repaired, np.zeros(3))
    assert np.linalg.norm(repaired - rotation) <= np.linalg.norm(
        noisy - rotation
    )


def test_nearest_rotation_rejects_what_se3_rejects_without_it() -> None:
    rng = np.random.default_rng(1)
    noisy = random_rotation(rng) * (1.0 + rng.uniform(-4e-3, 4e-3, (3, 3)))

    with pytest.raises(ValueError):
        SE3(noisy, np.zeros(3))
    SE3(nearest_rotation(noisy), np.zeros(3))


def test_nearest_rotation_turns_a_reflection_into_a_rotation() -> None:
    reflection = random_rotation(np.random.default_rng(2)) @ np.diag(
        [1.0, 1.0, -1.0]
    )

    repaired = nearest_rotation(reflection)

    assert np.linalg.det(repaired) == pytest.approx(1.0)
    SE3(repaired, np.zeros(3))


@pytest.mark.parametrize(
    "matrix", [np.eye(4), np.full((3, 3), np.nan), np.full((3, 3), np.inf)]
)
def test_nearest_rotation_rejects_invalid_matrices(
    matrix: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError):
        nearest_rotation(matrix)


def _model_outputs(
    n: int, width: int, height: int, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Poses, camera-frame points, and confidence logits like Pi3X emits."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        poses[i, :3, :3] = random_rotation(rng)
        poses[i, :3, 3] = rng.uniform(-2.0, 2.0, size=3)
    points = np.stack(
        [_point_map(60.0, 55.0, width, height, rng) for _ in range(n)]
    )
    logits = rng.normal(scale=2.0, size=(n, height, width, 1))
    return (
        poses.astype(np.float32),
        points.astype(np.float32),
        logits.astype(np.float32),
    )


def test_result_has_one_entry_per_view_at_the_working_resolution() -> None:
    poses, points, logits = _model_outputs(3, 40, 24, seed=0)

    result = to_reconstruction_result(poses, points, logits)

    assert len(result.intrinsics) == len(result.world_from_camera) == 3
    for k, depth, confidence in zip(
        result.intrinsics, result.depth, result.confidence, strict=True
    ):
        assert (k.width, k.height) == (40, 24)
        assert depth.shape == confidence.shape == (24, 40)
        assert depth.dtype == confidence.dtype == np.float32


def test_result_recovers_intrinsics_and_keeps_poses() -> None:
    poses, points, logits = _model_outputs(3, 40, 24, seed=1)

    result = to_reconstruction_result(poses, points, logits)

    for k, pose, expected in zip(
        result.intrinsics, result.world_from_camera, poses, strict=True
    ):
        assert k.fx == pytest.approx(60.0, rel=1e-4)
        assert k.fy == pytest.approx(55.0, rel=1e-4)
        np.testing.assert_allclose(pose.matrix, expected, atol=1e-6)


def test_result_depth_is_the_camera_frame_z() -> None:
    poses, points, logits = _model_outputs(2, 40, 24, seed=2)

    result = to_reconstruction_result(poses, points, logits)

    for depth, view in zip(result.depth, points, strict=True):
        np.testing.assert_array_equal(depth, view[..., 2])


def test_result_marks_invalid_depth_as_zero() -> None:
    poses, points, logits = _model_outputs(1, 40, 24, seed=3)
    points[0, 0, :4, 2] = [0.0, -1.0, np.nan, np.inf]

    result = to_reconstruction_result(poses, points, logits)

    np.testing.assert_array_equal(result.depth[0][0, :4], 0.0)
    assert np.all(result.depth[0][1:] > 0)


def test_result_confidence_is_the_sigmoid_of_the_logits() -> None:
    poses, points, logits = _model_outputs(1, 40, 24, seed=4)
    logits[0, 0, :3, 0] = [-1000.0, 0.0, 1000.0]
    logits[0, 1, 0, 0] = np.nan

    with np.errstate(all="raise"):
        result = to_reconstruction_result(poses, points, logits)

    confidence = result.confidence[0]
    np.testing.assert_array_equal(confidence[0, :3], [0.0, 0.5, 1.0])
    assert confidence[1, 0] == 0.0
    np.testing.assert_allclose(
        confidence[2:], 1 / (1 + np.exp(-logits[0, 2:, :, 0])), atol=1e-6
    )


def test_result_weights_the_intrinsics_fit_by_confidence() -> None:
    poses, points, logits = _model_outputs(1, 40, 24, seed=5)
    points[0, :8, :, :2] *= 3.0
    logits[0, :8] = -30.0

    result = to_reconstruction_result(poses, points, logits)

    assert result.intrinsics[0].fx == pytest.approx(60.0, rel=1e-3)


def test_result_survives_a_view_whose_logits_are_all_very_negative() -> None:
    poses, points, logits = _model_outputs(2, 40, 24, seed=6)
    logits[1] = -40.0

    result = to_reconstruction_result(poses, points, logits)

    assert not result.confidence[1].any()
    assert result.intrinsics[1].fx == pytest.approx(60.0, rel=1e-4)
    assert result.intrinsics[1].fy == pytest.approx(55.0, rel=1e-4)
    assert result.confidence[0].any()


def test_result_repairs_slightly_non_orthonormal_rotations() -> None:
    poses, points, logits = _model_outputs(2, 40, 24, seed=6)
    poses[:, :3, :3] *= 1.003

    result = to_reconstruction_result(poses, points, logits)

    for pose in result.world_from_camera:
        np.testing.assert_allclose(
            pose.rotation @ pose.rotation.T, np.eye(3), atol=1e-9
        )


@pytest.mark.parametrize(
    ("poses_shape", "points_shape", "logits_shape"),
    [
        ((2, 4, 4), (3, 24, 40, 3), (3, 24, 40, 1)),
        ((3, 4, 4), (3, 24, 40, 2), (3, 24, 40, 1)),
        ((3, 3, 4), (3, 24, 40, 3), (3, 24, 40, 1)),
        ((3, 4, 4), (3, 24, 40, 3), (3, 24, 40)),
        ((3, 4, 4), (3, 24, 40, 3), (3, 24, 41, 1)),
    ],
)
def test_result_rejects_inconsistent_shapes(
    poses_shape: tuple[int, ...],
    points_shape: tuple[int, ...],
    logits_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="shape"):
        to_reconstruction_result(
            np.zeros(poses_shape, np.float32),
            np.ones(points_shape, np.float32),
            np.zeros(logits_shape, np.float32),
        )
