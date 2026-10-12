import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rerun as rr
from rrd_reading import Rrd, rgba
from visualization_scenes import (
    ADDED_BOX,
    MOVED_FROM,
    MOVED_TO,
    OTHER_REGION_BOX,
    POINTS,
    REGION_BOX,
    REMOVED_BOX,
    UNCERTAIN_BOX,
    UNOBSERVED_POINTS,
    hand_scene,
    record,
)

from spatial_change.domain import ChangeType
from spatial_change.visualization import (
    RerunConfig,
    layers,
    save_recording,
)

BEFORE, AFTER = 0, 1
CHANGES = "/world/changes"


def test_the_scene_validates_its_epochs() -> None:
    scene = hand_scene()

    with pytest.raises(ValueError, match="before"):
        dataclasses.replace(scene, before=scene.after)
    with pytest.raises(ValueError, match="after"):
        dataclasses.replace(scene, after=scene.before)


def test_the_evidence_must_cover_every_point() -> None:
    scene = hand_scene()
    assert scene.evidence is not None
    short = dataclasses.replace(
        scene.evidence.before,
        counts=scene.evidence.before.counts[:-1],
        states=scene.evidence.before.states[:-1],
    )
    evidence = dataclasses.replace(scene.evidence, before=short)

    with pytest.raises(ValueError, match="evidence"):
        dataclasses.replace(scene, evidence=evidence)


def test_each_epoch_logs_its_points_at_its_own_step(rrd: Rrd) -> None:
    scene = hand_scene()

    before = rrd.at("/world/before/points", "Points3D:positions", BEFORE)
    after = rrd.at("/world/after/points", "Points3D:positions", AFTER)

    assert np.allclose(before, scene.before.points)
    assert np.allclose(after, scene.after.points)
    assert len(rrd.rows("/world/before/points", "Points3D:positions")) == 1


def test_points_keep_their_colors_and_the_configured_radius(rrd: Rrd) -> None:
    scene = hand_scene()

    colors = rrd.at("/world/after/points", "Points3D:colors", AFTER)
    radii = rrd.at("/world/after/points", "Points3D:radii", AFTER)

    assert [list(rgba(c)[:3]) for c in colors] == scene.after.colors.tolist()
    assert set(radii) == {np.float32(RerunConfig().point_radius)}


def test_the_before_subtree_is_cleared_once_when_the_after_step_starts(
    rrd: Rrd,
) -> None:
    clears = rrd.rows("/world/before", "Clear:is_recursive")

    assert [(c.step, c.value) for c in clears] == [(AFTER, [True])]
    assert not rrd.rows("/world/after", "Clear:is_recursive")
    below = [
        e
        for e in rrd.entities
        if e.startswith("/world/before/") and rrd.rows(e, "Clear:is_recursive")
    ]
    assert not below


def test_a_large_cloud_is_capped_and_subsampled_deterministically(
    tmp_path: Path,
) -> None:
    config = RerunConfig(max_points=100, max_image_panels=0)
    scene = hand_scene()

    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    one = record(tmp_path / "a", scene, config)
    two = record(tmp_path / "b", scene, config)
    other = record(tmp_path / "c", scene, dataclasses.replace(config, seed=1))

    kept = one.at("/world/after/points", "Points3D:positions", AFTER)
    again = two.at("/world/after/points", "Points3D:positions", AFTER)
    reseeded = other.at("/world/after/points", "Points3D:positions", AFTER)
    assert len(kept) == 100
    assert kept == again
    assert kept != reseeded
    rows = {tuple(p) for p in scene.after.points.astype(np.float32).tolist()}
    assert all(tuple(p) in rows for p in kept)
    assert len({tuple(p) for p in kept}) == 100


def test_both_camera_trajectories_are_static_line_strips(rrd: Rrd) -> None:
    scene = hand_scene()

    for epoch, recon in (("before", scene.before), ("after", scene.after)):
        (strip,) = rrd.at(
            f"/world/{epoch}/trajectory", "LineStrips3D:strips", None
        )
        expected = [f.world_from_camera.translation for f in recon.frames]
        assert np.allclose(strip, expected)


def test_the_changes_get_one_entity_group_each(rrd: Rrd) -> None:
    for change in hand_scene().report.changes:
        assert f"{CHANGES}/{change.change_id}/box" in rrd.entities


def _box_color(rrd: Rrd, change_id: str, step: int) -> tuple[int, ...]:
    colors = rrd.at(f"{CHANGES}/{change_id}/box", "Boxes3D:colors", step)
    return rgba(colors[0])


@pytest.mark.parametrize(
    ("change_id", "kind", "step"),
    [
        ("added-0", ChangeType.ADDED, AFTER),
        ("removed-1", ChangeType.REMOVED, BEFORE),
        ("moved-2", ChangeType.MOVED, AFTER),
        ("uncertain-3", ChangeType.UNCERTAIN, BEFORE),
    ],
)
def test_boxes_are_colored_by_change_type(
    rrd: Rrd, change_id: str, kind: ChangeType, step: int
) -> None:
    expected = (*RerunConfig().change_colors[kind], 255)

    assert _box_color(rrd, change_id, step) == expected


def test_the_four_colors_differ(rrd: Rrd) -> None:
    colors = {
        _box_color(rrd, "added-0", AFTER),
        _box_color(rrd, "removed-1", BEFORE),
        _box_color(rrd, "moved-2", AFTER),
        _box_color(rrd, "uncertain-3", BEFORE),
    }

    assert len(colors) == 4


def test_boxes_carry_the_change_label_and_geometry(rrd: Rrd) -> None:
    path = f"{CHANGES}/removed-1/box"

    assert rrd.at(path, "Boxes3D:labels", BEFORE) == ["removed-object"]
    assert np.allclose(rrd.at(path, "Boxes3D:centers", BEFORE), [REMOVED_BOX])
    assert np.allclose(
        rrd.at(path, "Boxes3D:half_sizes", BEFORE), [[0.2, 0.2, 0.2]]
    )


def test_a_box_is_oriented_by_its_quaternion(tmp_path: Path) -> None:
    scene = hand_scene()
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = scene.report.changes[0].after_box
    assert rotated is not None
    change = scene.report.changes[0].model_copy(
        update={
            "after_box": rotated.model_copy(update={"rotation": yaw.tolist()})
        }
    )
    report = scene.report.model_copy(
        update={"changes": [change, *scene.report.changes[1:]]}
    )

    rrd = record(tmp_path, dataclasses.replace(scene, report=report))

    (quaternion,) = rrd.at(
        f"{CHANGES}/added-0/box", "Boxes3D:quaternions", AFTER
    )
    root = np.sqrt(0.5)
    assert np.allclose(quaternion, [0.0, 0.0, root, root])


def test_a_removed_box_disappears_at_the_after_step(rrd: Rrd) -> None:
    for change_id in ("removed-1", "uncertain-3"):
        clears = rrd.rows(f"{CHANGES}/{change_id}/box", "Clear:is_recursive")
        assert [c.step for c in clears] == [AFTER]
    assert not rrd.rows(f"{CHANGES}/added-0/box", "Clear:is_recursive")


def test_an_added_box_exists_only_at_the_after_step(rrd: Rrd) -> None:
    rows = rrd.rows(f"{CHANGES}/added-0/box", "Boxes3D:centers")

    assert [r.step for r in rows] == [AFTER]
    assert np.allclose(rows[0].value, [ADDED_BOX])


def test_a_moved_object_has_a_box_at_each_step(rrd: Rrd) -> None:
    path = f"{CHANGES}/moved-2/box"

    assert np.allclose(rrd.at(path, "Boxes3D:centers", BEFORE), [MOVED_FROM])
    assert np.allclose(rrd.at(path, "Boxes3D:centers", AFTER), [MOVED_TO])


def test_a_moved_object_has_a_translucent_ghost_of_its_old_box(
    rrd: Rrd,
) -> None:
    path = f"{CHANGES}/moved-2/ghost"
    rows = rrd.rows(path, "Boxes3D:centers")

    assert [r.step for r in rows] == [AFTER]
    assert np.allclose(rows[0].value, [MOVED_FROM])
    alpha = rgba(rrd.at(path, "Boxes3D:colors", AFTER)[0])[3]
    assert alpha == round(RerunConfig().ghost_alpha * 255)
    assert rrd.at(path, "Boxes3D:fill_mode", AFTER) == [3]


def test_a_moved_object_has_an_arrow_from_old_to_new_center(rrd: Rrd) -> None:
    path = f"{CHANGES}/moved-2/arrow"

    origins = rrd.at(path, "Arrows3D:origins", AFTER)
    vectors = rrd.at(path, "Arrows3D:vectors", AFTER)

    assert np.allclose(origins, [MOVED_FROM])
    assert np.allclose(vectors, [np.subtract(MOVED_TO, MOVED_FROM)])
    expected = (*RerunConfig().change_colors[ChangeType.MOVED], 255)
    assert rgba(rrd.at(path, "Arrows3D:colors", AFTER)[0]) == expected


def test_only_moved_changes_have_a_ghost_and_an_arrow(rrd: Rrd) -> None:
    for change_id in ("added-0", "removed-1", "uncertain-3"):
        assert f"{CHANGES}/{change_id}/ghost" not in rrd.entities
        assert f"{CHANGES}/{change_id}/arrow" not in rrd.entities


def test_an_uncertain_change_is_drawn_where_it_has_a_box(rrd: Rrd) -> None:
    path = f"{CHANGES}/uncertain-3/box"

    assert np.allclose(rrd.at(path, "Boxes3D:centers", BEFORE), [UNCERTAIN_BOX])


def test_unobserved_regions_are_translucent_boxes(rrd: Rrd) -> None:
    path = "/world/unknown/region-0"

    center = rrd.at(path, "Boxes3D:centers", AFTER)
    colors = rrd.at(path, "Boxes3D:colors", AFTER)

    assert np.allclose(center, [REGION_BOX])
    alpha = rgba(colors[0])[3]
    assert alpha == round(RerunConfig().unknown_alpha * 255)
    assert alpha < 255
    assert rrd.at(path, "Boxes3D:fill_mode", AFTER) == [3]


def test_a_region_seen_before_is_unknown_only_after(rrd: Rrd) -> None:
    path = "/world/unknown/region-0"

    assert [r.step for r in rrd.rows(path, "Boxes3D:centers")] == [AFTER]
    assert not rrd.rows(path, "Clear:is_recursive")


def test_a_region_seen_after_is_unknown_only_before(rrd: Rrd) -> None:
    path = "/world/unknown/region-1"

    assert [r.step for r in rrd.rows(path, "Boxes3D:centers")] == [BEFORE]
    assert np.allclose(
        rrd.at(path, "Boxes3D:centers", BEFORE), [OTHER_REGION_BOX]
    )
    assert [r.step for r in rrd.rows(path, "Clear:is_recursive")] == [AFTER]


def test_the_move_layers_land_on_the_after_step_from_any_step(
    tmp_path: Path,
) -> None:
    config = RerunConfig()
    moved = next(
        c for c in hand_scene().report.changes if c.type is ChangeType.MOVED
    )
    stream = rr.RecordingStream("move")
    stream.set_time("epoch", sequence=BEFORE)
    layers._log_move(
        stream, moved, config.change_colors[ChangeType.MOVED], config
    )
    path = tmp_path / "move.rrd"
    stream.save(path)
    stream.flush()
    stream.disconnect()

    rrd = Rrd(path)

    ghost = f"{CHANGES}/moved-2/ghost"
    assert [r.step for r in rrd.rows(ghost, "Boxes3D:centers")] == [AFTER]
    arrow = f"{CHANGES}/moved-2/arrow"
    assert [r.step for r in rrd.rows(arrow, "Arrows3D:origins")] == [AFTER]


class _Calls:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.names: list[str] = []
        for name in ("save", "log", "disconnect"):
            original = getattr(rr.RecordingStream, name)
            monkeypatch.setattr(
                rr.RecordingStream, name, self._wrap(name, original)
            )

    def _wrap(self, name: str, original: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.names.append(name)
            return original(*args, **kwargs)

        return call


def test_the_file_sink_is_attached_before_anything_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _Calls(monkeypatch)

    save_recording(hand_scene(), tmp_path / "a.rrd", RerunConfig())

    assert calls.names[0] == "save"
    assert calls.names.count("save") == 1
    assert calls.names[-1] == "disconnect"


def test_the_stream_is_disconnected_when_saving_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _Calls(monkeypatch)

    with pytest.raises(Exception):  # noqa: B017
        save_recording(hand_scene(), tmp_path, RerunConfig())

    assert calls.names[-1] == "disconnect"


def test_the_stream_is_disconnected_when_logging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _Calls(monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(layers, "log_points", fail)

    with pytest.raises(RuntimeError, match="boom"):
        save_recording(hand_scene(), tmp_path / "b.rrd", RerunConfig())

    assert calls.names[-1] == "disconnect"


def test_the_saved_file_is_complete(tmp_path: Path) -> None:
    scene = hand_scene()

    save_recording(scene, tmp_path / "c.rrd", RerunConfig())

    rrd = Rrd(tmp_path / "c.rrd")
    points = rrd.at("/world/after/points", "Points3D:positions", AFTER)
    assert len(points) == POINTS
    assert rrd.views()


def test_never_observed_points_are_drawn_translucent(rrd: Rrd) -> None:
    scene = hand_scene()
    for epoch, recon in (("before", scene.before), ("after", scene.after)):
        path = f"/world/{epoch}/unobserved"
        step = BEFORE if epoch == "before" else AFTER

        positions = rrd.at(path, "Points3D:positions", step)
        colors = rrd.at(path, "Points3D:colors", step)

        assert np.allclose(positions, recon.points[:UNOBSERVED_POINTS])
        assert {rgba(c)[3] for c in colors} == {
            round(RerunConfig().unknown_alpha * 255)
        }


def test_no_unobserved_points_without_evidence(tmp_path: Path) -> None:
    rrd = record(tmp_path, hand_scene(with_evidence=False))

    assert "/world/before/unobserved" not in rrd.entities
    assert "/world/after/unobserved" not in rrd.entities
    assert (
        len(rrd.at("/world/before/points", "Points3D:positions", 0)) == POINTS
    )


def test_unobserved_points_can_be_turned_off(tmp_path: Path) -> None:
    config = RerunConfig(max_unobserved_points=0, max_image_panels=0)

    rrd = record(tmp_path, config=config)

    assert "/world/before/unobserved" not in rrd.entities
