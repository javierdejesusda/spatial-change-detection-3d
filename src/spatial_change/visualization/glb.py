"""Annotated glTF 2.0 binary export of a change report.

The file holds one POINTS primitive per epoch with vertex colors, one node
per change whose mesh draws the change boxes and, for a move, an arrow from
the old to the new box center, and one gray box node per region that only
one capture observed. Everything but the points is LINES. A change node is
named ``"<change_id> <type> <label>"``, colored by its type, and carries
the fields of the change in its ``extras``; a region node is named
``"unobserved-<region_id>"``.

The container is written by hand with numpy and ``struct``. Coordinates are
the z-up room frame of the report, converted to the y-up frame of glTF with
one rotation of -90 degrees about x: ``(x, y, z) -> (x, z, -y)``. The
rotation is applied to the vertices, so no reader has to honor a node
transform.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import Change, ChangeReport, Epoch
from spatial_change.visualization.palette import (
    CHANGE_COLORS,
    RGB,
    UNOBSERVED_COLOR,
)
from spatial_change.visualization.points import ScenePoints, subsample
from spatial_change.visualization.segments import (
    Role,
    change_segments,
    region_segments,
)

_MAGIC = 0x46546C67
_VERSION = 2
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_FLOAT = 5126
_UBYTE = 5121
_ARRAY_BUFFER = 34962
_POINTS = 0
_LINES = 1
_UNLIT = "KHR_materials_unlit"


@dataclass(frozen=True)
class GlbConfig:
    """Parameters of the GLB export.

    Attributes:
        max_points: Largest number of points written per epoch; larger
            clouds are thinned evenly.
    """

    max_points: int = 200_000

    def __post_init__(self) -> None:
        """Checks the values.

        Raises:
            ValueError: If ``max_points`` is below 1.
        """
        if self.max_points < 1:
            raise ValueError(
                f"max_points must be at least 1, got {self.max_points}"
            )


def _y_up(points: NDArray[np.float64]) -> NDArray[np.float32]:
    """Rotates z-up room points to y-up: ``(x, y, z) -> (x, z, -y)``."""
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1).astype(
        np.float32
    )


class _Binary:
    """Collects accessors over one buffer, each view 4-byte aligned."""

    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def add(self, data: NDArray[Any], component_type: int, kind: str) -> int:
        """Appends an array as a buffer view and accessor.

        Float positions record their per-component min and max, which glTF
        requires of ``POSITION``. Bytes are read as normalized colors.

        Args:
            data: Array of shape ``(N, width)``, at least one row.
            component_type: glTF component type of the data.
            kind: Accessor type, for example ``"VEC3"``.

        Returns:
            The accessor index.
        """
        raw = np.ascontiguousarray(data).tobytes()
        self.blob.extend(b"\0" * (-len(self.blob) % 4))
        self.views.append(
            {
                "buffer": 0,
                "byteOffset": len(self.blob),
                "byteLength": len(raw),
                "target": _ARRAY_BUFFER,
            }
        )
        self.blob.extend(raw)
        accessor: dict[str, Any] = {
            "bufferView": len(self.views) - 1,
            "componentType": component_type,
            "count": len(data),
            "type": kind,
        }
        if component_type == _UBYTE:
            accessor["normalized"] = True
        if component_type == _FLOAT and kind == "VEC3":
            accessor["min"] = data.min(axis=0).tolist()
            accessor["max"] = data.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _chunk(kind: int, payload: bytes, pad: bytes) -> bytes:
    payload += pad * (-len(payload) % 4)
    return struct.pack("<II", len(payload), kind) + payload


def _change_extras(change: Change) -> dict[str, Any]:
    return {
        "change_id": change.change_id,
        "type": change.type.value,
        "label": change.label,
        "description": change.description,
        "confidence": change.confidence,
        "displacement_m": change.displacement_m,
        "rotation_deg": change.rotation_deg,
        "evidence": change.evidence.model_dump(),
    }


def _material(name: str, color: RGB) -> dict[str, Any]:
    factor = [round(c / 255, 4) for c in color] + [1.0]
    return {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": factor,
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "extensions": {_UNLIT: {}},
    }


class _SceneBuilder:
    """Collects nodes, meshes, and materials over one binary buffer."""

    def __init__(self) -> None:
        self.binary = _Binary()
        self.nodes: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        self._material_index: dict[str, int] = {}

    def _material(self, name: str, color: RGB) -> int:
        if name not in self._material_index:
            self._material_index[name] = len(self.materials)
            self.materials.append(_material(name, color))
        return self._material_index[name]

    def _node(
        self, name: str, primitives: list[dict[str, Any]]
    ) -> dict[str, Any]:
        node: dict[str, Any] = {"name": name}
        if primitives:
            self.meshes.append({"name": name, "primitives": primitives})
            node["mesh"] = len(self.meshes) - 1
        self.nodes.append(node)
        return node

    def _lines(
        self,
        parts: list[tuple[Role, NDArray[np.float64]]],
        material: int,
    ) -> list[dict[str, Any]]:
        primitives = []
        for role, segments in parts:
            vertices = _y_up(segments.reshape(-1, 3))
            position = self.binary.add(vertices, _FLOAT, "VEC3")
            primitives.append(
                {
                    "attributes": {"POSITION": position},
                    "mode": _LINES,
                    "material": material,
                    "extras": {"role": role.value},
                }
            )
        return primitives

    def add_clouds(
        self, points: Mapping[Epoch, ScenePoints], max_points: int
    ) -> None:
        """Adds a POINTS node per epoch that has any finite point."""
        for epoch in Epoch:
            cloud = subsample(
                points.get(epoch, ScenePoints.empty()), max_points
            )
            if len(cloud.points) == 0:
                continue
            rgba = np.full((len(cloud.points), 4), 255, dtype=np.uint8)
            rgba[:, :3] = cloud.colors
            attributes = {
                "POSITION": self.binary.add(
                    _y_up(cloud.points), _FLOAT, "VEC3"
                ),
                "COLOR_0": self.binary.add(rgba, _UBYTE, "VEC4"),
            }
            self._node(
                f"points-{epoch.value}",
                [{"attributes": attributes, "mode": _POINTS}],
            )

    def add_changes(self, changes: list[Change]) -> None:
        """Adds a node per change, with a mesh when it has any geometry."""
        for change in changes:
            parts = change_segments(change)
            material = (
                self._material(
                    f"change-{change.type.value}", CHANGE_COLORS[change.type]
                )
                if parts
                else 0
            )
            name = f"{change.change_id} {change.type.value} {change.label}"
            node = self._node(name, self._lines(parts, material))
            node["extras"] = _change_extras(change)

    def add_regions(self, report: ChangeReport) -> None:
        """Adds a gray box node per unobserved region."""
        for region in report.unobserved_regions:
            material = self._material("unobserved", UNOBSERVED_COLOR)
            node = self._node(
                f"unobserved-{region.region_id}",
                self._lines(region_segments(region), material),
            )
            node["extras"] = {
                "region_id": region.region_id,
                "epoch_seen": region.epoch_seen.value,
            }

    def to_bytes(self) -> bytes:
        """Returns the scene as the bytes of a ``.glb`` file."""
        binary = self.binary
        gltf: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": "spatial-change"},
            "scene": 0,
            "scenes": [
                {"nodes": list(range(len(self.nodes)))} if self.nodes else {}
            ],
        }
        if self.nodes:
            gltf["nodes"] = self.nodes
        if self.materials:
            gltf["materials"] = self.materials
            gltf["extensionsUsed"] = [_UNLIT]
        if self.meshes:
            gltf["meshes"] = self.meshes
        if binary.blob:
            gltf["buffers"] = [{"byteLength": len(binary.blob)}]
            gltf["bufferViews"] = binary.views
            gltf["accessors"] = binary.accessors
        text = json.dumps(gltf, separators=(",", ":"), allow_nan=False)
        chunks = _chunk(_JSON_CHUNK, text.encode(), b" ")
        if binary.blob:
            chunks += _chunk(_BIN_CHUNK, bytes(binary.blob), b"\0")
        return struct.pack("<III", _MAGIC, _VERSION, 12 + len(chunks)) + chunks


def export_glb(
    points: Mapping[Epoch, ScenePoints],
    report: ChangeReport,
    config: GlbConfig | None = None,
) -> bytes:
    """Builds the annotated scene as a binary glTF.

    Args:
        points: Colored points per epoch in the z-up room frame. An epoch
            that is missing or has no finite point is left out.
        report: The changes and unobserved regions to draw.
        config: Export parameters; the defaults when ``None``.

    Returns:
        The bytes of a ``.glb`` file. Equal input gives equal bytes.
    """
    config = config or GlbConfig()
    scene = _SceneBuilder()
    scene.add_clouds(points, config.max_points)
    scene.add_changes(report.changes)
    scene.add_regions(report)
    return scene.to_bytes()


def write_glb(
    points: Mapping[Epoch, ScenePoints],
    report: ChangeReport,
    path: Path,
    config: GlbConfig | None = None,
) -> None:
    """Writes the annotated scene to a ``.glb`` file.

    Args:
        points: Colored points per epoch in the z-up room frame.
        report: The changes and unobserved regions to draw.
        path: Output file; missing parent directories are created.
        config: Export parameters; the defaults when ``None``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(export_glb(points, report, config))
