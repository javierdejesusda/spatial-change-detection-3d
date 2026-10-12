import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from report_scenes import box, change, cloud, empty_report, one_of_each_report

from spatial_change.domain import ChangeReport, ChangeType, Epoch
from spatial_change.visualization import (
    CHANGE_COLORS,
    UNOBSERVED_COLOR,
    GlbConfig,
    ScenePoints,
    export_glb,
    write_glb,
)

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT = 5126
UBYTE = 5121
POINTS, LINES = 0, 1
COMPONENTS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
DTYPES = {FLOAT: np.float32, UBYTE: np.uint8}


def _parse(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Parses a GLB, asserting the container rules on the way."""
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    assert version == 2
    assert length == len(data)
    assert length % 4 == 0
    offset = 12
    chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
    assert chunk_type == JSON_CHUNK
    assert chunk_length % 4 == 0
    gltf = json.loads(data[offset + 8 : offset + 8 + chunk_length])
    offset += 8 + chunk_length
    blob = b""
    if offset < length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        assert chunk_type == BIN_CHUNK
        assert chunk_length % 4 == 0
        blob = data[offset + 8 : offset + 8 + chunk_length]
        offset += 8 + chunk_length
    assert offset == length
    return gltf, blob


def _read(gltf: dict[str, Any], blob: bytes, index: int) -> NDArray[Any]:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    assert view["byteOffset"] % 4 == 0
    assert view["byteOffset"] + view["byteLength"] <= len(blob)
    width = COMPONENTS[accessor["type"]]
    dtype = np.dtype(DTYPES[accessor["componentType"]])
    assert view["byteLength"] == accessor["count"] * width * dtype.itemsize
    start = view["byteOffset"] + accessor.get("byteOffset", 0)
    flat = np.frombuffer(
        blob, dtype=dtype, count=accessor["count"] * width, offset=start
    )
    return flat.reshape(accessor["count"], width)


def _points() -> dict[Epoch, ScenePoints]:
    return {Epoch.BEFORE: cloud(50, 0), Epoch.AFTER: cloud(60, 1)}


def _nodes(gltf: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["name"]: node for node in gltf["nodes"]}


def _build() -> tuple[dict[str, Any], bytes]:
    return _parse(export_glb(_points(), one_of_each_report()))


def test_container_follows_the_binary_gltf_layout() -> None:
    gltf, blob = _build()

    assert gltf["asset"]["version"] == "2.0"
    (buffer,) = gltf["buffers"]
    assert len(blob) - 4 < buffer["byteLength"] <= len(blob)
    assert gltf["scenes"][0]["nodes"] == list(range(len(gltf["nodes"])))
    assert gltf["scene"] == 0


def test_every_view_and_accessor_stays_inside_the_buffer() -> None:
    gltf, blob = _build()

    for view in gltf["bufferViews"]:
        assert view["buffer"] == 0
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(blob)
    for index in range(len(gltf["accessors"])):
        _read(gltf, blob, index)


def test_position_accessors_carry_exact_min_and_max() -> None:
    gltf, blob = _build()

    seen = 0
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            index = primitive["attributes"]["POSITION"]
            accessor = gltf["accessors"][index]
            data = _read(gltf, blob, index)
            assert accessor["componentType"] == FLOAT
            np.testing.assert_array_equal(accessor["min"], data.min(axis=0))
            np.testing.assert_array_equal(accessor["max"], data.max(axis=0))
            seen += 1
    assert seen == 2 + 3 + 1 + 1 + 2 + 1


def test_each_epoch_is_a_points_primitive_with_colors() -> None:
    gltf, blob = _build()
    nodes = _nodes(gltf)
    clouds = _points()

    for epoch in Epoch:
        mesh = gltf["meshes"][nodes[f"points-{epoch.value}"]["mesh"]]
        (primitive,) = mesh["primitives"]
        assert primitive["mode"] == POINTS
        index = primitive["attributes"]["COLOR_0"]
        accessor = gltf["accessors"][index]
        assert accessor["componentType"] == UBYTE
        assert accessor["normalized"] is True
        assert accessor["type"] == "VEC4"
        colors = _read(gltf, blob, index)
        assert len(colors) == len(clouds[epoch].points)
        np.testing.assert_array_equal(colors[:, :3], clouds[epoch].colors)
        assert np.all(colors[:, 3] == 255)


def test_z_up_points_become_y_up() -> None:
    points = {
        Epoch.BEFORE: ScenePoints(
            np.array([[1.0, 2.0, 3.0]]), np.array([[9, 8, 7]], np.uint8)
        )
    }

    gltf, blob = _parse(export_glb(points, empty_report()))

    primitive = gltf["meshes"][0]["primitives"][0]
    position = _read(gltf, blob, primitive["attributes"]["POSITION"])
    np.testing.assert_array_equal(position, [[1.0, 3.0, -2.0]])


def test_point_cap_limits_each_epoch() -> None:
    config = GlbConfig(max_points=20)
    gltf, _ = _parse(export_glb(_points(), one_of_each_report(), config))

    nodes = _nodes(gltf)
    for epoch in Epoch:
        mesh = gltf["meshes"][nodes[f"points-{epoch.value}"]["mesh"]]
        index = mesh["primitives"][0]["attributes"]["POSITION"]
        assert gltf["accessors"][index]["count"] == 20


def test_each_change_is_a_named_node_with_its_fields() -> None:
    gltf, _ = _build()

    nodes = _nodes(gltf)
    for ident, kind, label in [
        ("c1", "moved", "chair"),
        ("c2", "added", "lamp"),
        ("c3", "removed", "crate"),
        ("c4", "uncertain", "rug"),
    ]:
        node = nodes[f"{ident} {kind} {label}"]
        assert node["extras"]["change_id"] == ident
        assert node["extras"]["type"] == kind
        assert node["extras"]["label"] == label
    moved = nodes["c1 moved chair"]["extras"]
    assert moved["displacement_m"] == 1.25
    assert moved["rotation_deg"] == 30.0
    assert moved["confidence"] == 0.87
    assert moved["description"] == "It changed."


def _roles(gltf: dict[str, Any], name: str) -> list[str]:
    mesh = gltf["meshes"][_nodes(gltf)[name]["mesh"]]
    return [p["extras"]["role"] for p in mesh["primitives"]]


def test_changes_draw_the_boxes_they_have_and_arrows_for_moves() -> None:
    gltf, _ = _build()

    assert _roles(gltf, "c1 moved chair") == [
        "before_box",
        "after_box",
        "arrow",
    ]
    assert _roles(gltf, "c2 added lamp") == ["after_box"]
    assert _roles(gltf, "c3 removed crate") == ["before_box"]
    assert _roles(gltf, "c4 uncertain rug") == ["before_box", "after_box"]


def test_boxes_are_twelve_line_segments_and_arrows_three() -> None:
    gltf, blob = _build()

    mesh = gltf["meshes"][_nodes(gltf)["c1 moved chair"]["mesh"]]
    counts = []
    for primitive in mesh["primitives"]:
        assert primitive["mode"] == LINES
        index = primitive["attributes"]["POSITION"]
        counts.append(len(_read(gltf, blob, index)))
    assert counts == [24, 24, 6]


def test_box_vertices_hold_the_converted_corners() -> None:
    gltf, blob = _build()

    mesh = gltf["meshes"][_nodes(gltf)["c2 added lamp"]["mesh"]]
    index = mesh["primitives"][0]["attributes"]["POSITION"]
    vertices = _read(gltf, blob, index)
    np.testing.assert_allclose(vertices.min(axis=0), [2.5, 0.1, -3.25])
    np.testing.assert_allclose(vertices.max(axis=0), [3.5, 0.9, -2.75])


def test_each_type_has_its_palette_color_on_the_material() -> None:
    gltf, _ = _build()

    nodes = _nodes(gltf)
    for name, kind in [
        ("c1 moved chair", ChangeType.MOVED),
        ("c2 added lamp", ChangeType.ADDED),
        ("c3 removed crate", ChangeType.REMOVED),
        ("c4 uncertain rug", ChangeType.UNCERTAIN),
    ]:
        mesh = gltf["meshes"][nodes[name]["mesh"]]
        for primitive in mesh["primitives"]:
            material = gltf["materials"][primitive["material"]]
            factor = material["pbrMetallicRoughness"]["baseColorFactor"]
            expected = [c / 255 for c in CHANGE_COLORS[kind]] + [1.0]
            np.testing.assert_allclose(factor, expected, atol=1e-4)
            assert material["name"] == f"change-{kind.value}"
            assert "KHR_materials_unlit" in material["extensions"]
    assert "KHR_materials_unlit" in gltf["extensionsUsed"]


def test_unobserved_regions_are_exported_as_gray_boxes() -> None:
    gltf, blob = _build()

    node = _nodes(gltf)["unobserved-u1"]
    assert node["extras"] == {"region_id": "u1", "epoch_seen": "before"}
    mesh = gltf["meshes"][node["mesh"]]
    (primitive,) = mesh["primitives"]
    assert primitive["mode"] == LINES
    assert primitive["extras"]["role"] == "region_box"
    index = primitive["attributes"]["POSITION"]
    assert len(_read(gltf, blob, index)) == 24
    material = gltf["materials"][primitive["material"]]
    factor = material["pbrMetallicRoughness"]["baseColorFactor"]
    expected = [c / 255 for c in UNOBSERVED_COLOR] + [1.0]
    np.testing.assert_allclose(factor, expected, atol=1e-4)
    assert material["name"] == "unobserved"
    assert not any(node["name"].startswith("u1") for node in gltf["nodes"])


def test_same_input_gives_the_same_bytes() -> None:
    first = export_glb(_points(), one_of_each_report())
    second = export_glb(_points(), one_of_each_report())

    assert first == second


def test_empty_epochs_are_left_out() -> None:
    points = {Epoch.BEFORE: ScenePoints.empty(), Epoch.AFTER: cloud(5, 0)}

    gltf, _ = _parse(export_glb(points, one_of_each_report()))

    names = [node["name"] for node in gltf["nodes"]]
    assert "points-before" not in names
    assert "points-after" in names


def _assert_valid(gltf: dict[str, Any]) -> None:
    """Checks the glTF rules that forbid empty arrays and dangling refs."""
    for key, value in gltf.items():
        if isinstance(value, list):
            assert value, f"{key} must not be empty"
    for mesh in gltf.get("meshes", []):
        assert mesh["primitives"]
    used = {
        p["material"]
        for mesh in gltf.get("meshes", [])
        for p in mesh["primitives"]
        if "material" in p
    }
    assert used == set(range(len(gltf.get("materials", []))))
    meshes = [n["mesh"] for n in gltf.get("nodes", []) if "mesh" in n]
    assert sorted(meshes) == list(range(len(gltf.get("meshes", []))))
    for scene in gltf["scenes"]:
        assert all(i < len(gltf["nodes"]) for i in scene.get("nodes", []))


def test_nothing_at_all_is_still_a_valid_glb() -> None:
    data = export_glb({}, empty_report())

    gltf, blob = _parse(data)
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["scenes"] == [{}]
    assert "nodes" not in gltf
    assert "meshes" not in gltf
    assert "buffers" not in gltf
    assert blob == b""
    _assert_valid(gltf)


def test_a_full_scene_has_no_empty_arrays_or_unused_materials() -> None:
    gltf, _ = _build()

    _assert_valid(gltf)


def test_a_change_without_boxes_is_a_node_with_no_mesh() -> None:
    report = ChangeReport(
        changes=[change("c1", ChangeType.UNCERTAIN, "rug", None, None)],
        unobserved_regions=[],
        metadata={},
    )

    gltf, _ = _parse(export_glb({}, report))

    (node,) = gltf["nodes"]
    assert node["name"] == "c1 uncertain rug"
    assert "mesh" not in node
    assert node["extras"]["change_id"] == "c1"
    assert "meshes" not in gltf
    assert "materials" not in gltf
    _assert_valid(gltf)


def test_a_report_with_no_clouds_still_draws_the_boxes() -> None:
    gltf, blob = _parse(export_glb({}, one_of_each_report()))

    assert len(gltf["nodes"]) == 5
    assert len(blob) > 0
    _assert_valid(gltf)


def test_a_description_with_odd_characters_survives() -> None:
    text = 'quote " <b> é \\ end'
    report = ChangeReport(
        changes=[
            change("c1", ChangeType.ADDED, "lamp", None, box((0, 0, 0)), text)
        ],
        unobserved_regions=[],
        metadata={},
    )

    gltf, _ = _parse(export_glb({}, report))

    assert gltf["nodes"][0]["extras"]["description"] == text


def test_config_rejects_a_cap_below_one() -> None:
    with pytest.raises(ValueError, match="max_points"):
        GlbConfig(max_points=0)


def test_write_glb_writes_the_exported_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out" / "scene.glb"

    write_glb(_points(), one_of_each_report(), path)

    assert path.read_bytes() == export_glb(_points(), one_of_each_report())


def test_open3d_reads_the_file_without_raising(tmp_path: Path) -> None:
    o3d = pytest.importorskip("open3d")
    path = tmp_path / "scene.glb"
    write_glb(_points(), one_of_each_report(), path)

    mesh = o3d.io.read_triangle_mesh(str(path))

    assert isinstance(mesh, o3d.geometry.TriangleMesh)
