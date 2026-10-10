"""CPU ray-cast rendering of synthetic rooms.

Rays are the unit-depth unprojection of every pixel through the domain
``Intrinsics`` (pixel centers at integer coordinates, OpenCV axes). Each
ray direction has camera-frame z equal to 1, so the hit distance along the
ray is exactly the depth along the camera z axis and renders agree with
``geometry.projection``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from spatial_change.domain import SE3, Intrinsics
from spatial_change.geometry.projection import unproject_depth

_INVALID_ID = int(o3d.t.geometry.RaycastingScene.INVALID_ID)
_LIGHT = np.array([0.35, 0.25, 0.90]) / np.linalg.norm([0.35, 0.25, 0.90])
_AMBIENT = 0.40
_CHECKER_CELL_M = 0.5
_NOISE_CELL_M = 0.08

_FLOOR = np.array([0.55, 0.50, 0.45])
_CEILING = np.array([0.92, 0.92, 0.90])
_WALL_X = np.array([0.78, 0.72, 0.66])
_WALL_Y = np.array([0.66, 0.72, 0.78])


@dataclass(frozen=True, eq=False)
class PlacedMesh:
    """A mesh in world coordinates with its instance index and color.

    Attributes:
        index: Value written to the instance map where this mesh is hit.
        vertices: World vertices of shape ``(N, 3)``.
        triangles: Triangle indices of shape ``(M, 3)``.
        color: Base RGB color in ``[0, 1]``.
    """

    index: int
    vertices: NDArray[np.float64]
    triangles: NDArray[np.int32]
    color: tuple[float, float, float]


@dataclass(frozen=True, eq=False)
class RenderedView:
    """The outputs of rendering one camera.

    Attributes:
        rgb: ``(H, W, 3)`` uint8 image.
        depth: ``(H, W)`` float32 depth along the camera z axis, 0 where
            nothing is hit.
        instance: ``(H, W)`` int32 object index, -1 for the room shell or
            a miss.
    """

    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    instance: NDArray[np.int32]


class RenderScene:
    """A closed room plus posed objects, ready for ray casting."""

    def __init__(
        self,
        room_size: tuple[float, float, float],
        meshes: Sequence[PlacedMesh],
    ) -> None:
        """Builds the ray-casting scene.

        Args:
            room_size: Room extent ``(x, y, z)`` in meters; the room spans
                ``[0, x] x [0, y] x [0, z]``.
            meshes: Objects in the room.
        """
        self._scene = o3d.t.geometry.RaycastingScene()
        shell = o3d.geometry.TriangleMesh.create_box(*room_size)
        self._shell_id = self._add(
            np.asarray(shell.vertices), np.asarray(shell.triangles)
        )
        ids = [self._add(m.vertices, m.triangles) for m in meshes]
        size = max([self._shell_id, *ids]) + 1
        self._instance_of_geometry = np.full(size, -1, dtype=np.int32)
        self._albedo_of_geometry = np.zeros((size, 3))
        for geometry_id, mesh in zip(ids, meshes, strict=True):
            self._instance_of_geometry[geometry_id] = mesh.index
            self._albedo_of_geometry[geometry_id] = mesh.color

    def _add(
        self, vertices: NDArray[np.float64], triangles: NDArray[np.int32]
    ) -> int:
        return int(
            self._scene.add_triangles(
                o3d.core.Tensor.from_numpy(
                    np.ascontiguousarray(vertices, dtype=np.float32)
                ),
                o3d.core.Tensor.from_numpy(
                    np.ascontiguousarray(triangles, dtype=np.uint32)
                ),
            )
        )

    def render(
        self, intrinsics: Intrinsics, world_from_camera: SE3
    ) -> RenderedView:
        """Renders depth, color, and instance maps for one camera.

        Args:
            intrinsics: Camera intrinsics, defining the image size.
            world_from_camera: Camera-to-world pose.

        Returns:
            The rendered maps.
        """
        shape = (intrinsics.height, intrinsics.width)
        directions_camera = unproject_depth(
            np.ones(shape, np.float32), intrinsics, SE3.identity()
        ).reshape(*shape, 3)
        directions = directions_camera @ world_from_camera.rotation.T
        origin = world_from_camera.translation
        rays = np.concatenate(
            [np.broadcast_to(origin, directions.shape), directions], axis=-1
        )
        hits = self._scene.cast_rays(
            o3d.core.Tensor.from_numpy(
                np.ascontiguousarray(rays, dtype=np.float32)
            )
        )
        t_hit = hits["t_hit"].numpy()
        geometry = hits["geometry_ids"].numpy().astype(np.int64)
        normals = hits["primitive_normals"].numpy().astype(np.float64)

        hit = np.isfinite(t_hit) & (geometry != _INVALID_ID)
        depth = np.where(hit, t_hit, 0.0).astype(np.float32)
        slot = np.where(hit, geometry, self._shell_id)
        instance = np.where(hit, self._instance_of_geometry[slot], -1)
        rgb = self._shade(hit, slot, depth, origin, directions, normals)
        return RenderedView(rgb, depth, instance.astype(np.int32))

    def _shade(
        self,
        hit: NDArray[np.bool_],
        slot: NDArray[np.int64],
        depth: NDArray[np.float32],
        origin: NDArray[np.float64],
        directions: NDArray[np.float64],
        normals: NDArray[np.float64],
    ) -> NDArray[np.uint8]:
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.where(norm > 0, norm, 1.0)
        facing = np.sum(normals * directions, axis=-1, keepdims=True)
        normals = np.where(facing > 0, -normals, normals)

        points = origin + depth[..., None] * directions
        albedo = self._albedo_of_geometry[slot]
        shell = hit & (slot == self._shell_id)
        albedo[shell] = _shell_albedo(normals[shell], points[shell])

        lambert = np.clip(normals @ _LIGHT, 0.0, 1.0)
        shade = _AMBIENT + (1.0 - _AMBIENT) * lambert
        rgb = np.rint(255.0 * albedo * shade[..., None])
        rgb[~hit] = 0.0
        return np.clip(rgb, 0, 255).astype(np.uint8)


def _shell_albedo(
    normals: NDArray[np.float64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Colors room-shell hits by surface and a world-space texture.

    Args:
        normals: Unit normals facing the camera, shape ``(N, 3)``.
        points: Hit points, shape ``(N, 3)``.

    Returns:
        Albedo of shape ``(N, 3)``: a per-surface color times a checker
        and a hashed noise pattern fixed in world coordinates.
    """
    base = np.where(
        (np.abs(normals[:, 2]) > 0.5)[:, None],
        np.where((normals[:, 2] > 0)[:, None], _FLOOR, _CEILING),
        np.where((np.abs(normals[:, 0]) > 0.5)[:, None], _WALL_X, _WALL_Y),
    )
    checker_cells = np.floor(points / _CHECKER_CELL_M).astype(np.int64)
    checker = np.where(checker_cells.sum(axis=1) % 2 == 0, 1.0, 0.8)
    noise_cells = np.floor(points / _NOISE_CELL_M).astype(np.int64)
    hashed = (
        (noise_cells[:, 0] * 73856093)
        ^ (noise_cells[:, 1] * 19349663)
        ^ (noise_cells[:, 2] * 83492791)
    ) & 0xFF
    noise = 0.88 + 0.12 * (hashed / 255.0)
    return base * (checker * noise)[:, None]
