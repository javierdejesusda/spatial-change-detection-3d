"""Furniture-like meshes assembled from box and cylinder primitives.

Every model lives in its own local frame: z is up, the object rests on
``z = 0``, and the local origin sits at the footprint anchor chosen by the
factory. A model is posed in the world with a yaw-only ``SE3`` from
``place``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from spatial_change.domain import SE3, OrientedBox
from spatial_change.geometry import yaw_rotation

_CYLINDER_RESOLUTION = 24

_COLORS: dict[str, tuple[float, float, float]] = {
    "table": (0.60, 0.42, 0.25),
    "chair": (0.20, 0.35, 0.65),
    "stool": (0.85, 0.65, 0.15),
    "crate": (0.70, 0.30, 0.20),
    "bottle": (0.15, 0.60, 0.35),
    "drum": (0.55, 0.20, 0.60),
    "shelf": (0.80, 0.78, 0.70),
}


@dataclass(frozen=True, eq=False)
class ObjectModel:
    """A triangle mesh in its local frame with a deterministic color.

    Attributes:
        label: Class name of the object.
        vertices: Local vertices of shape ``(N, 3)``, float64.
        triangles: Triangle vertex indices of shape ``(M, 3)``, int32.
        color: Base RGB color in ``[0, 1]``.
        local_box: Tight axis-aligned box of the vertices in the local
            frame, as an ``OrientedBox`` with identity rotation.
    """

    label: str
    vertices: NDArray[np.float64]
    triangles: NDArray[np.int32]
    color: tuple[float, float, float]
    local_box: OrientedBox


def _box_part(
    size: tuple[float, float, float],
    center_xy: tuple[float, float],
    z0: float,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    mesh.translate((center_xy[0] - size[0] / 2, center_xy[1] - size[1] / 2, z0))
    return mesh


def _cylinder_part(
    radius: float,
    height: float,
    center_xy: tuple[float, float],
    z0: float,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius, height, resolution=_CYLINDER_RESOLUTION
    )
    mesh.translate((center_xy[0], center_xy[1], z0 + height / 2))
    return mesh


def _four_legs(
    leg: float, height: float, reach_x: float, reach_y: float
) -> Iterator[o3d.geometry.TriangleMesh]:
    """Yields four square legs standing on the floor around the origin.

    Args:
        leg: Side length of the square leg section.
        height: Leg height.
        reach_x: Absolute x offset of every leg center.
        reach_y: Absolute y offset of every leg center.

    Yields:
        The legs for signs ``(-, -)``, ``(-, +)``, ``(+, -)``, ``(+, +)`` of
        the ``(x, y)`` offsets, in that order.
    """
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            yield _box_part(
                (leg, leg, height), (sx * reach_x, sy * reach_y), 0.0
            )


def _assemble(
    label: str, parts: Sequence[o3d.geometry.TriangleMesh]
) -> ObjectModel:
    merged = parts[0]
    for part in parts[1:]:
        merged = merged + part
    vertices = np.array(merged.vertices, dtype=np.float64)
    triangles = np.array(merged.triangles, dtype=np.int32)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return ObjectModel(
        label=label,
        vertices=vertices,
        triangles=triangles,
        color=_COLORS[label],
        local_box=OrientedBox((low + high) / 2, high - low, np.eye(3)),
    )


def make_table(rng: np.random.Generator) -> ObjectModel:
    """Builds a table: a thin top on four legs.

    Args:
        rng: Source of the size variation.

    Returns:
        The table model.
    """
    length, width = rng.uniform(1.0, 1.4), rng.uniform(0.6, 0.8)
    height, top, leg = rng.uniform(0.70, 0.80), 0.05, 0.06
    inset = 0.04
    parts = [_box_part((length, width, top), (0.0, 0.0), height - top)]
    parts.extend(
        _four_legs(
            leg,
            height - top,
            length / 2 - inset - leg / 2,
            width / 2 - inset - leg / 2,
        )
    )
    return _assemble("table", parts)


def make_chair(rng: np.random.Generator) -> ObjectModel:
    """Builds a chair: seat, four legs, and a back on the local -y side.

    Args:
        rng: Source of the size variation.

    Returns:
        The chair model.
    """
    side, seat_h = rng.uniform(0.40, 0.46), rng.uniform(0.43, 0.48)
    back_h, thick, leg = rng.uniform(0.40, 0.48), 0.04, 0.04
    reach = side / 2 - leg / 2
    parts = [_box_part((side, side, thick), (0.0, 0.0), seat_h - thick)]
    parts.extend(_four_legs(leg, seat_h - thick, reach, reach))
    parts.append(
        _box_part((side, thick, back_h), (0.0, -(side - thick) / 2), seat_h)
    )
    return _assemble("chair", parts)


def make_stool(rng: np.random.Generator) -> ObjectModel:
    """Builds a stool: a round seat on four legs.

    Args:
        rng: Source of the size variation.

    Returns:
        The stool model.
    """
    radius, height = rng.uniform(0.18, 0.22), rng.uniform(0.42, 0.52)
    thick, leg = 0.04, 0.04
    reach = (radius * 0.7 - leg / 2) / 1.4142
    parts = [_cylinder_part(radius, thick, (0.0, 0.0), height - thick)]
    parts.extend(_four_legs(leg, height - thick, reach, reach))
    return _assemble("stool", parts)


def make_crate(rng: np.random.Generator) -> ObjectModel:
    """Builds a solid box.

    Args:
        rng: Source of the size variation.

    Returns:
        The crate model.
    """
    size = (
        rng.uniform(0.40, 0.70),
        rng.uniform(0.35, 0.60),
        rng.uniform(0.35, 0.70),
    )
    return _assemble("crate", [_box_part(size, (0.0, 0.0), 0.0)])


def make_bottle(rng: np.random.Generator) -> ObjectModel:
    """Builds a bottle: a cylinder body with a narrow neck.

    Args:
        rng: Source of the size variation.

    Returns:
        The bottle model.
    """
    radius, height = rng.uniform(0.04, 0.06), rng.uniform(0.22, 0.30)
    neck = 0.08
    return _assemble(
        "bottle",
        [
            _cylinder_part(radius, height, (0.0, 0.0), 0.0),
            _cylinder_part(radius * 0.4, neck, (0.0, 0.0), height),
        ],
    )


def make_drum(rng: np.random.Generator) -> ObjectModel:
    """Builds a barrel-like cylinder.

    Args:
        rng: Source of the size variation.

    Returns:
        The drum model.
    """
    radius, height = rng.uniform(0.20, 0.28), rng.uniform(0.50, 0.80)
    return _assemble("drum", [_cylinder_part(radius, height, (0.0, 0.0), 0.0)])


def make_shelf(rng: np.random.Generator) -> ObjectModel:
    """Builds a tall cabinet that is wide along local y.

    Args:
        rng: Source of the size variation.

    Returns:
        The shelf model: plinth, body, and top cap.
    """
    depth, width = rng.uniform(0.40, 0.50), rng.uniform(1.2, 1.5)
    height = rng.uniform(1.9, 2.2)
    plinth, cap = 0.08, 0.04
    return _assemble(
        "shelf",
        [
            _box_part((depth - 0.04, width - 0.04, plinth), (0.0, 0.0), 0.0),
            _box_part(
                (depth - 0.02, width - 0.02, height - plinth - cap),
                (0.0, 0.0),
                plinth,
            ),
            _box_part((depth, width, cap), (0.0, 0.0), height - cap),
        ],
    )


FACTORIES: dict[str, Callable[[np.random.Generator], ObjectModel]] = {
    "table": make_table,
    "chair": make_chair,
    "stool": make_stool,
    "crate": make_crate,
    "bottle": make_bottle,
    "drum": make_drum,
    "shelf": make_shelf,
}


def place(
    model: ObjectModel, center_xy: NDArray[np.float64], yaw: float, z: float
) -> SE3:
    """Poses a model so its box center lies above a footprint point.

    Args:
        model: The model to place.
        center_xy: World ``(x, y)`` the box center should take.
        yaw: Rotation about the z axis, in radians.
        z: World height of the model's local ``z = 0`` plane.

    Returns:
        The ``world_from_object`` pose.
    """
    rotation = yaw_rotation(yaw)
    offset = rotation @ model.local_box.center
    translation = np.array(
        [center_xy[0] - offset[0], center_xy[1] - offset[1], z]
    )
    return SE3(rotation, translation)


def world_box(model: ObjectModel, pose: SE3) -> OrientedBox:
    """Returns the tight oriented box of a posed model.

    Args:
        model: The model.
        pose: The ``world_from_object`` pose.

    Returns:
        The box in world coordinates.
    """
    box = model.local_box
    return OrientedBox(
        pose.apply(box.center[None, :])[0],
        box.extent,
        pose.rotation @ box.rotation,
    )


def world_vertices(model: ObjectModel, pose: SE3) -> NDArray[np.float64]:
    """Returns the mesh vertices of a posed model in world coordinates.

    Args:
        model: The model.
        pose: The ``world_from_object`` pose.

    Returns:
        Vertices of shape ``(N, 3)``.
    """
    return pose.apply(model.vertices)
