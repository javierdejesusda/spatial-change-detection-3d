import dataclasses
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from rrd_reading import Rrd, rgba
from visualization_scenes import hand_scene, record

from spatial_change.domain import ChangeType
from spatial_change.geometry import in_image, project
from spatial_change.visualization import (
    RerunConfig,
)

FRAMES = 6
BEFORE, AFTER = 0, 1


def panel_frames(rrd: Rrd, epoch: str) -> list[int]:
    prefix = f"/world/{epoch}/cameras/frame_"
    return sorted(
        int(e.removeprefix(prefix))
        for e in rrd.entities
        if e.startswith(prefix) and "/" not in e.removeprefix(prefix)
    )


def overlay(epoch: str, frame: int, change_id: str) -> str:
    return f"/world/{epoch}/cameras/frame_{frame}/overlay/{change_id}"


def test_every_frame_with_visible_changes_gets_a_panel(rrd: Rrd) -> None:
    assert panel_frames(rrd, "before") == list(range(FRAMES))
    assert panel_frames(rrd, "after") == list(range(FRAMES))


def test_the_panel_count_is_capped_per_epoch(tmp_path: Path) -> None:
    config = dataclasses.replace(RerunConfig(), max_image_panels=3)

    rrd = record(tmp_path, hand_scene(), config)

    assert len(panel_frames(rrd, "before")) == 3
    assert len(panel_frames(rrd, "after")) == 3


def test_a_capped_selection_spreads_over_the_frames(tmp_path: Path) -> None:
    config = dataclasses.replace(RerunConfig(), max_image_panels=2)

    rrd = record(tmp_path, hand_scene(), config)

    assert panel_frames(rrd, "before") == [0, FRAMES - 1]
    assert panel_frames(rrd, "after") == [0, FRAMES - 1]


def test_no_panels_are_drawn_when_the_cap_is_zero(tmp_path: Path) -> None:
    config = dataclasses.replace(RerunConfig(), max_image_panels=0)

    rrd = record(tmp_path, hand_scene(), config)

    assert not any("/cameras/" in e for e in rrd.entities)
    assert rrd.views() == [("3D", "Scene", "world")]


def test_a_panel_has_the_frame_pose_and_downscaled_intrinsics(
    rrd: Rrd,
) -> None:
    frame = hand_scene().after.frames[2]
    path = "/world/after/cameras/frame_2"

    translation = rrd.at(path, "Transform3D:translation", AFTER)
    rotation = rrd.at(path, "Transform3D:mat3x3", AFTER)
    resolution = rrd.at(path, "Pinhole:resolution", AFTER)
    projection = rrd.at(path, "Pinhole:image_from_camera", AFTER)

    assert np.allclose(translation, frame.world_from_camera.translation)
    stored = np.reshape(rotation, (3, 3), order="F")
    assert not np.allclose(frame.world_from_camera.rotation, np.eye(3))
    assert not np.allclose(frame.world_from_camera.rotation.T, stored)
    assert np.allclose(stored, frame.world_from_camera.rotation)
    assert resolution == [[32.0, 24.0]]
    matrix = np.reshape(projection, (3, 3), order="F")
    assert matrix[0, 0] == pytest.approx(25.0)
    assert matrix[1, 1] == pytest.approx(25.0)


def test_a_panel_holds_the_downscaled_image(rrd: Rrd) -> None:
    frame = hand_scene().before.frames[1]
    assert frame.image is not None

    buffer = rrd.at(
        "/world/before/cameras/frame_1/image", "Image:buffer", BEFORE
    )

    assert len(buffer[0]) == 24 * 32 * 3
    assert abs(np.mean(buffer[0]) - frame.image.mean()) < 3.0


def test_frames_without_pixels_get_no_panel(tmp_path: Path) -> None:
    scene = hand_scene()
    frames = [
        dataclasses.replace(f, image=None) if f.index % 2 else f
        for f in scene.after.frames
    ]
    after = dataclasses.replace(scene.after, frames=frames)

    rrd = record(tmp_path, dataclasses.replace(scene, after=after))

    assert panel_frames(rrd, "after") == [0, 2, 4]


def test_overlays_follow_the_boxes_of_the_epoch(rrd: Rrd) -> None:
    for change_id in ("removed-1", "moved-2", "uncertain-3"):
        assert overlay("before", 2, change_id) in rrd.entities
    assert overlay("before", 2, "added-0") not in rrd.entities
    for change_id in ("added-0", "moved-2"):
        assert overlay("after", 2, change_id) in rrd.entities
    for change_id in ("removed-1", "uncertain-3"):
        assert overlay("after", 2, change_id) not in rrd.entities


def test_an_overlay_is_a_labelled_box_around_projected_points(rrd: Rrd) -> None:
    path = overlay("before", 2, "removed-1")

    points = np.array(rrd.at(path, "Points2D:positions", BEFORE))
    center = np.array(rrd.at(path, "Boxes2D:centers", BEFORE)[0])
    half = np.array(rrd.at(path, "Boxes2D:half_sizes", BEFORE)[0])

    assert len(points) > 5
    assert np.allclose(points.min(axis=0), center - half)
    assert np.allclose(points.max(axis=0), center + half)
    assert rrd.at(path, "Boxes2D:labels", BEFORE) == ["removed-object"]
    assert points.min() >= 0.0
    assert points[:, 0].max() < 32
    assert points[:, 1].max() < 24


def test_overlays_use_rerun_pixel_coordinates(rrd: Rrd) -> None:
    scene = hand_scene()
    frame = scene.before.frames[2]
    box = scene.report.changes[1].before_box
    assert box is not None
    inside = scene.before.points[box.to_box().contains(scene.before.points)]
    uv, depth = project(inside, frame.intrinsics, frame.world_from_camera)
    seen = uv[in_image(uv, depth, frame.intrinsics)]
    expected = 0.5 * (seen + 0.5)

    logged = np.array(
        rrd.at(overlay("before", 2, "removed-1"), "Points2D:positions", BEFORE)
    )

    gaps = np.linalg.norm(logged[:, None] - expected[None], axis=2).min(axis=1)
    assert gaps.max() < 1e-3


def test_the_pinhole_puts_the_principal_point_in_rerun_coordinates(
    rrd: Rrd,
) -> None:
    frame = hand_scene().after.frames[2]
    path = "/world/after/cameras/frame_2"

    projection = rrd.at(path, "Pinhole:image_from_camera", AFTER)

    matrix = np.reshape(projection, (3, 3), order="F")
    scaled = frame.intrinsics.scaled(0.5, 0.5)
    assert matrix[0, 2] == pytest.approx(scaled.cx + 0.5)
    assert matrix[1, 2] == pytest.approx(scaled.cy + 0.5)


def test_overlays_are_colored_by_change_type(rrd: Rrd) -> None:
    path = overlay("before", 2, "removed-1")
    expected = (*RerunConfig().change_colors[ChangeType.REMOVED], 255)

    assert rgba(rrd.at(path, "Boxes2D:colors", BEFORE)[0]) == expected
    colors = {rgba(c) for c in rrd.at(path, "Points2D:colors", BEFORE)}
    assert colors == {expected}


def test_overlay_points_are_capped(tmp_path: Path) -> None:
    config = dataclasses.replace(RerunConfig(), panel_points=7)

    rrd = record(tmp_path, hand_scene(), config)

    positions = rrd.at(
        overlay("before", 2, "removed-1"), "Points2D:positions", BEFORE
    )
    assert 0 < len(positions) <= 7


def test_points_behind_a_nearer_surface_are_left_out(tmp_path: Path) -> None:
    scene = hand_scene()
    near = []
    for frame in scene.before.frames:
        assert frame.depth is not None
        near.append(
            dataclasses.replace(frame, depth=np.full_like(frame.depth, 1.0))
        )
    before = dataclasses.replace(scene.before, frames=near)

    rrd = record(tmp_path, dataclasses.replace(scene, before=before))

    assert not any(
        e.startswith("/world/before/cameras/") and "/overlay/" in e
        for e in rrd.entities
    )


def test_an_epoch_without_changes_still_gets_evenly_spread_panels(
    tmp_path: Path,
) -> None:
    scene = hand_scene()
    empty = scene.report.model_copy(update={"changes": []})
    config = dataclasses.replace(RerunConfig(), max_image_panels=3)

    rrd = record(tmp_path, dataclasses.replace(scene, report=empty), config)

    assert panel_frames(rrd, "before") == [0, 2, FRAMES - 1]
    assert not any("/overlay/" in e for e in rrd.entities)


def test_frames_without_depth_use_plain_projection(tmp_path: Path) -> None:
    scene = hand_scene()
    flat = [dataclasses.replace(f, depth=None) for f in scene.before.frames]
    before = dataclasses.replace(scene.before, frames=flat)

    rrd = record(tmp_path, dataclasses.replace(scene, before=before))

    assert overlay("before", 2, "removed-1") in rrd.entities


def _declared_axes(tmp_path: Path, axes: object) -> object:
    stream = rr.RecordingStream("axes")
    stream.log("world", axes, static=True)
    path = tmp_path / "axes.rrd"
    stream.save(path)
    stream.flush()
    stream.disconnect()
    return Rrd(path).at("/world", "ViewCoordinates:xyz", None)


def test_the_world_is_declared_z_up(rrd: Rrd, tmp_path: Path) -> None:
    z_up = _declared_axes(tmp_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    y_up = _declared_axes(tmp_path, rr.ViewCoordinates.RIGHT_HAND_Y_UP)

    assert z_up != y_up
    assert rrd.at("/world", "ViewCoordinates:xyz", None) == z_up


def test_the_blueprint_has_a_3d_view_and_a_2d_view_per_panel(rrd: Rrd) -> None:
    views = rrd.views()

    assert ("3D", "Scene", "world") in views
    two_d = [v for v in views if v[0] == "2D"]
    assert sorted(v[2] for v in two_d) == sorted(
        f"world/{epoch}/cameras/frame_{i}"
        for epoch in ("before", "after")
        for i in range(FRAMES)
    )
    for _, name, origin in two_d:
        assert origin.split("/")[1] in name.lower()


def test_the_blueprint_uses_the_epoch_timeline(rrd: Rrd) -> None:
    assert rrd.blueprint_timeline() == "epoch"


def test_camera_panels_are_logged_at_their_own_epoch_step(rrd: Rrd) -> None:
    before = "/world/before/cameras/frame_2"
    after = "/world/after/cameras/frame_2"

    assert [r.step for r in rrd.rows(f"{before}/image", "Image:buffer")] == [
        BEFORE
    ]
    assert [r.step for r in rrd.rows(f"{after}/image", "Image:buffer")] == [
        AFTER
    ]
    assert [r.step for r in rrd.rows(before, "Pinhole:resolution")] == [BEFORE]
    assert [r.step for r in rrd.rows(after, "Pinhole:resolution")] == [AFTER]


def test_the_trajectories_stay_static_across_both_steps(rrd: Rrd) -> None:
    for epoch in ("before", "after"):
        rows = rrd.rows(f"/world/{epoch}/trajectory", "LineStrips3D:strips")
        assert [r.step for r in rows] == [None]
