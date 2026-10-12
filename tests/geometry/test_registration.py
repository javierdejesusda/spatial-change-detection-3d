import numpy as np
import pytest
from matching_scenes import box_surface, chair, chair_pair, l_surface
from numpy.typing import NDArray
from rotations import rotation_z

from spatial_change import geometry
from spatial_change.domain import SE3
from spatial_change.geometry import (
    RegistrationConfig,
    register_objects,
    rotation_angle_deg,
)


def _placement_error(
    estimate: SE3, truth: SE3, points: NDArray[np.float64]
) -> float:
    """Gives the mean distance between where the two transforms put points."""
    gap = estimate.apply(points) - truth.apply(points)
    return float(np.linalg.norm(gap, axis=1).mean())


def test_identical_clouds_register_to_the_identity() -> None:
    cloud = chair(seed=0)

    result = register_objects(cloud, cloud.copy())

    assert result.succeeded
    assert rotation_angle_deg(result.transform.rotation) < 1.0
    assert np.linalg.norm(result.transform.translation) < 0.01
    assert result.fitness > 0.95


def test_a_translated_cuboid_is_not_reported_as_a_half_turn() -> None:
    source, target = chair_pair(0.6, -0.3)

    result = register_objects(source, target)

    assert result.succeeded
    assert rotation_angle_deg(result.transform.rotation) < 2.0
    assert result.transform.translation == pytest.approx(
        [0.6, -0.3, 0.0], abs=0.03
    )
    assert result.fitness > 0.9


def test_a_small_move_is_recovered() -> None:
    source, target = chair_pair(0.08)

    result = register_objects(source, target)

    assert result.transform.translation == pytest.approx(
        [0.08, 0.0, 0.0], abs=0.02
    )
    assert rotation_angle_deg(result.transform.rotation) < 2.0


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("yaw", [90.0, 40.0, 180.0])
def test_a_turned_l_shape_is_recovered_in_place(yaw: float, seed: int) -> None:
    center = np.array([1.0, 2.0, 0.2])
    source = l_surface((1.0, 2.0, 0.2), seed=2 * seed + 1)
    target = l_surface((1.0, 2.0, 0.2), yaw_deg=yaw, seed=2 * seed + 2)

    result = register_objects(source, target)

    rotation = rotation_z(yaw)
    truth = SE3(rotation, center - rotation @ center)
    assert result.succeeded
    assert _placement_error(result.transform, truth, source) < 0.02
    assert result.fitness > 0.9


def test_a_small_object_that_moved_and_turned_is_recovered() -> None:
    size = (0.07, 0.05, 0.06)
    source = box_surface(
        (0.0, 0.0, 0.03), size, seed=1, spacing=0.004, noise=0.0005
    )
    target = box_surface(
        (0.8, 0.6, 0.03),
        size,
        yaw_deg=30.0,
        seed=2,
        spacing=0.004,
        noise=0.0005,
    )

    result = register_objects(source, target)

    rotation = rotation_z(30.0)
    truth = SE3(rotation, np.array([0.8, 0.6, 0.03]) - rotation @ [0, 0, 0.03])
    assert result.succeeded
    assert _placement_error(result.transform, truth, source) < 0.01
    assert result.fitness > 0.9


def test_a_turned_and_moved_l_shape_is_recovered() -> None:
    source = l_surface((0.0, 0.0, 0.2), seed=1)
    target = l_surface((0.5, 0.3, 0.2), yaw_deg=60.0, seed=2)

    result = register_objects(source, target)

    assert rotation_angle_deg(result.transform.rotation) == pytest.approx(
        60.0, abs=3.0
    )
    assert result.fitness > 0.9


def test_a_partial_view_of_the_same_object_still_fits() -> None:
    full = l_surface((0.0, 0.0, 0.2), seed=1)
    moved = l_surface((0.4, 0.0, 0.2), seed=2)
    partial = moved[moved[:, 0] > 0.4]

    result = register_objects(full, partial)

    assert result.succeeded
    assert result.fitness > 0.9
    assert result.transform.translation == pytest.approx(
        [0.4, 0.0, 0.0], abs=0.04
    )


def test_the_partial_side_may_be_the_source() -> None:
    full = l_surface((0.4, 0.0, 0.2), seed=2)
    original = l_surface((0.0, 0.0, 0.2), seed=1)
    partial = original[original[:, 0] > 0.0]

    result = register_objects(partial, full)

    assert result.fitness > 0.9
    assert result.transform.translation == pytest.approx(
        [0.4, 0.0, 0.0], abs=0.04
    )


def test_different_shapes_fit_badly() -> None:
    cube = box_surface((0.0, 0.0, 0.3), (0.6, 0.6, 0.6))
    thin = box_surface((0.0, 0.0, 0.05), (1.2, 0.3, 0.1))

    result = register_objects(cube, thin)

    assert result.fitness < 0.6


def test_swapping_source_and_target_inverts_the_transform() -> None:
    first = l_surface((0.0, 0.0, 0.2), seed=1)
    second = l_surface((0.5, 0.3, 0.2), yaw_deg=60.0, seed=2)

    forward = register_objects(first, second)
    backward = register_objects(second, first)

    product = backward.transform.compose(forward.transform)
    assert rotation_angle_deg(product.rotation) < 3.0
    assert np.linalg.norm(product.translation) < 0.05


def test_two_runs_give_the_same_transform() -> None:
    first = l_surface((0.0, 0.0, 0.2), seed=1)
    second = l_surface((0.5, 0.3, 0.2), yaw_deg=60.0, seed=2)

    a = register_objects(first, second)
    b = register_objects(first, second)

    assert np.array_equal(a.transform.matrix, b.transform.matrix)
    assert a.fitness == b.fitness


def test_repeated_runs_on_a_partial_view_give_the_same_transform() -> None:
    full = l_surface((0.0, 0.0, 0.2), seed=1)
    moved = l_surface((0.4, 0.0, 0.2), seed=3)
    partial = moved[moved[:, 0] > 0.4]

    runs = [register_objects(full, partial).transform.matrix for _ in range(20)]

    assert all(np.array_equal(runs[0], run) for run in runs)


def test_too_few_points_fail_without_raising() -> None:
    tiny = np.random.default_rng(0).normal(size=(5, 3)) * 0.1

    result = register_objects(tiny, tiny)

    assert not result.succeeded
    assert result.fitness == 0.0


def test_a_degenerate_line_does_not_raise() -> None:
    x = np.linspace(0.0, 1.0, 200)
    line = np.stack([x, np.zeros(200), np.zeros(200)], axis=1)

    result = register_objects(line, line + 0.1)

    assert isinstance(result.succeeded, bool)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voxel_size", 0.0),
        ("voxel_size", float("inf")),
        ("normal_radius", 0.0),
        ("feature_radius", -1.0),
        ("inlier_distance", 0.0),
        ("ransac_iterations", 0),
        ("ransac_confidence", 1.5),
        ("edge_length_ratio", 0.0),
        ("yaw_starts", 0),
        ("max_tilt_deg", 181.0),
        ("tie_tolerance", -0.1),
        ("min_points", 2),
        ("min_voxels_across", 2),
    ],
)
def test_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        RegistrationConfig(**{field: value})


def test_the_inputs_are_not_modified() -> None:
    source: NDArray[np.float64] = l_surface((0.0, 0.0, 0.2), seed=1)
    target = l_surface((0.5, 0.3, 0.2), yaw_deg=60.0, seed=2)
    kept = source.copy(), target.copy()

    register_objects(source, target)

    assert np.array_equal(source, kept[0])
    assert np.array_equal(target, kept[1])


def test_the_package_exports_the_registration_names() -> None:
    assert geometry.__all__ == sorted(geometry.__all__)
    assert {
        "Registration",
        "RegistrationConfig",
        "register_objects",
        "rotation_angle_deg",
    } <= set(geometry.__all__)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cut", [0.80, 0.83, 0.85])
def test_a_top_slab_registers_onto_the_whole_object(
    cut: float, seed: int
) -> None:
    whole = chair(seed=seed + 1)
    moved = chair(x=0.5, seed=seed + 100)
    slab = moved[moved[:, 2] > cut]

    result = register_objects(whole, slab)

    assert result.succeeded
    assert rotation_angle_deg(result.transform.rotation) < 5.0
    assert result.transform.translation == pytest.approx(
        [0.5, 0.0, 0.0], abs=0.02
    )


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("cut", [0.3, 0.45])
def test_a_bottom_part_registers_onto_the_whole_object(
    cut: float, seed: int
) -> None:
    whole = chair(seed=seed + 1)
    moved = chair(x=0.5, yaw_deg=40.0, seed=seed + 100)
    part = moved[moved[:, 2] < cut]

    result = register_objects(whole, part)

    assert result.succeeded
    assert rotation_angle_deg(result.transform.rotation) == pytest.approx(
        40.0, abs=5.0
    )
