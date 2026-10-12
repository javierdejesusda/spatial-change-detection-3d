import numpy as np
import pytest
from rotations import rotation_z

from spatial_change.domain import SE3
from spatial_change.synthetic.objects import (
    CLASS_COLORS,
    FACTORIES,
    ObjectModel,
    place,
    world_box,
    world_vertices,
)
from spatial_change.synthetic.scene import _ROLES

LABELS = sorted(FACTORIES)


def build(label: str, seed: int = 0) -> ObjectModel:
    return FACTORIES[label](np.random.default_rng(seed))


def test_catalog_has_required_furniture() -> None:
    assert set(_ROLES.values()) <= set(FACTORIES)


def test_class_colors_are_the_model_colors_as_bytes() -> None:
    assert set(CLASS_COLORS) == set(FACTORIES)
    for label in LABELS:
        expected = tuple(round(c * 255) for c in build(label).color)
        assert CLASS_COLORS[label] == expected


def test_class_colors_are_read_only() -> None:
    with pytest.raises(TypeError):
        CLASS_COLORS["table"] = (0, 0, 0)  # type: ignore[index]


@pytest.mark.parametrize("label", LABELS)
def test_mesh_is_valid_and_rests_on_floor(label: str) -> None:
    model = build(label)

    assert model.label == label
    assert model.vertices.ndim == 2 and model.vertices.shape[1] == 3
    assert model.triangles.shape[1] == 3
    assert model.triangles.min() >= 0
    assert model.triangles.max() < len(model.vertices)
    assert model.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert all(0.0 <= c <= 1.0 for c in model.color)


@pytest.mark.parametrize("label", LABELS)
def test_local_box_is_tight(label: str) -> None:
    model = build(label)
    lo, hi = model.vertices.min(axis=0), model.vertices.max(axis=0)

    np.testing.assert_allclose(model.local_box.center, (lo + hi) / 2)
    np.testing.assert_allclose(model.local_box.extent, hi - lo)
    np.testing.assert_allclose(model.local_box.rotation, np.eye(3))


def test_table_has_top_and_legs() -> None:
    table = build("table")

    # A table top alone is one box (8 vertices); legs add more.
    assert len(table.vertices) >= 5 * 8
    assert 0.6 < table.local_box.extent[2] < 1.0


@pytest.mark.parametrize("label", LABELS)
def test_same_seed_same_mesh_and_other_seed_differs(label: str) -> None:
    a, b, c = build(label, 3), build(label, 3), build(label, 4)

    np.testing.assert_array_equal(a.vertices, b.vertices)
    np.testing.assert_array_equal(a.triangles, b.triangles)
    assert not np.array_equal(a.vertices, c.vertices)


def test_place_puts_box_center_at_requested_footprint() -> None:
    model = build("chair")

    pose = place(model, np.array([2.0, 3.0]), yaw=0.6, z=0.25)
    box = world_box(model, pose)

    np.testing.assert_allclose(box.center[:2], [2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(box.rotation, rotation_z(np.degrees(0.6)))
    assert box.center[2] - box.extent[2] / 2 == pytest.approx(0.25)


@pytest.mark.parametrize("label", LABELS)
def test_world_box_is_tight_around_world_vertices(label: str) -> None:
    model = build(label)
    pose = place(model, np.array([1.5, 2.5]), yaw=1.1, z=0.0)
    box = world_box(model, pose)

    local = SE3(box.rotation, box.center).apply_inverse(
        world_vertices(model, pose)
    )

    np.testing.assert_allclose(local.max(axis=0), box.extent / 2, atol=1e-9)
    np.testing.assert_allclose(local.min(axis=0), -box.extent / 2, atol=1e-9)
