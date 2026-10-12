import dataclasses
import functools
from collections.abc import Sequence

import numpy as np
import pytest
from fakes import ScriptedLabeler
from lifting_scenes import Cuboid, HandScene, hand_scene, ring_poses

from spatial_change.domain import Epoch, EpochReconstruction
from spatial_change.geometry.views import ViewConfig
from spatial_change.geometry.walls import WallConfig
from spatial_change.perception import FakeLabeler
from spatial_change.semantics.labeling import LabelConfig
from spatial_change.semantics.landmarks import (
    LandmarkConfig,
    _changed_near,
    extract_landmarks,
)
from spatial_change.semantics.vocabulary import FURNITURE, STRUCTURE
from spatial_change.visibility import CrossEvidence, PointEvidence, PointState

TABLE = (200, 20, 20)
SHELF = (20, 200, 20)
DOOR = (20, 20, 200)
COLORS = {"table": TABLE, "shelf": SHELF, "door": DOOR}
TABLE_BOX = Cuboid((-1.1, -0.6, 0.0), (-0.4, 0.2, 0.75))
SHELF_BOX = Cuboid((0.5, -0.3, 0.0), (0.9, 0.5, 1.6))
WALL = Cuboid((-1.3, 2.8, 0.0), (1.3, 2.85, 2.3))
BESIDE_BOX = Cuboid((1.45, 2.4, 0.0), (1.85, 2.8, 0.8))
CRATE_BOX = Cuboid((0.0, -1.2, 0.0), (0.4, -0.8, 0.4))
POSES = ring_poses(10, radius=2.4, height=1.4, target=(0.0, 0.2, 0.7))
LOOSE = LandmarkConfig(
    labeling=LabelConfig(
        views=ViewConfig(min_visible_points=10), min_crop_px=1
    ),
    min_points=20,
)


@dataclasses.dataclass(frozen=True, eq=False)
class Room:
    """A hand-built room with the evidence of its points."""

    reconstruction: EpochReconstruction
    evidence: CrossEvidence
    scene: HandScene


def _room(
    boxes: Sequence[Cuboid],
    colors: Sequence[tuple[int, int, int]],
    states: dict[int, PointState] | None = None,
    ground_color: tuple[int, int, int] | None = None,
    epoch: Epoch = Epoch.BEFORE,
) -> Room:
    """Renders boxes in the colors, with a state per object index."""
    return _render(
        tuple(boxes),
        tuple(colors),
        tuple(sorted((states or {}).items())),
        ground_color,
        epoch,
    )


@functools.cache
def _render(
    boxes: tuple[Cuboid, ...],
    colors: tuple[tuple[int, int, int], ...],
    state_items: tuple[tuple[int, PointState], ...],
    ground_color: tuple[int, int, int] | None,
    epoch: Epoch,
) -> Room:
    """Renders a room once per distinct set of arguments."""
    states = dict(state_items)
    scene = hand_scene(boxes, POSES, epoch)
    frames = []
    for frame, instances in zip(scene.frames, scene.instance_maps, strict=True):
        assert frame.image is not None
        image = frame.image.copy()
        for index, color in enumerate(colors):
            image[instances == index] = color
        if ground_color is not None:
            image[instances == -1] = ground_color
        frames.append(dataclasses.replace(frame, image=image))
    scene = HandScene(frames, scene.instance_maps, epoch)
    chunks = [scene.object_points(i) for i in range(-1, len(boxes))]
    points = np.concatenate(chunks)
    codes = np.concatenate(
        [
            np.full(
                len(chunk),
                states.get(index, PointState.PRESENT),
                np.int8,
            )
            for index, chunk in zip(range(-1, len(boxes)), chunks, strict=True)
        ]
    )
    reconstruction = EpochReconstruction(
        epoch=epoch,
        frames=frames,
        points=points,
        colors=np.zeros((len(points), 3), np.uint8),
        confidence=np.ones(len(points), np.float32),
    )
    evidence = PointEvidence(np.zeros((len(points), 4), np.int32), codes)
    return Room(reconstruction, CrossEvidence(evidence, evidence), scene)


def _find(
    room: Room,
    config: LandmarkConfig = LOOSE,
    colors: dict[str, tuple[int, int, int]] | None = None,
) -> list[tuple[str, str]]:
    found = extract_landmarks(
        room.reconstruction,
        room.evidence,
        FakeLabeler(colors or COLORS),
        config,
    )
    return [(item.landmark_id, item.label) for item in found]


@pytest.mark.parametrize(
    "overrides",
    [
        {"voxel_size": 0.0},
        {"floor_clearance": -0.1},
        {"max_height": 0.01},
        {"link_radius": 0},
        {"min_points": 0},
        {"min_extent": 0.0},
        {"max_extent": 0.1},
        {"min_height": -0.1},
        {"min_score": -0.1},
        {"min_score": 1.1},
        {"max_changed_share": 0.0},
        {"max_changed_share": 1.5},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LandmarkConfig(**overrides)  # type: ignore[arg-type]


def test_a_table_and_a_shelf_are_found_in_order_of_position() -> None:
    room = _room([TABLE_BOX, SHELF_BOX], [TABLE, SHELF])

    found = extract_landmarks(
        room.reconstruction, room.evidence, FakeLabeler(COLORS), LOOSE
    )

    assert [(i.landmark_id, i.label) for i in found] == [
        ("landmark-0", "table"),
        ("landmark-1", "shelf"),
    ]
    np.testing.assert_allclose(
        found[0].box.center[:2], [-0.75, -0.2], atol=0.15
    )
    np.testing.assert_allclose(found[1].box.center[:2], [0.7, 0.1], atol=0.15)
    assert found[1].box.center[2] == pytest.approx(0.8, abs=0.15)


def test_the_labeler_gets_one_call_over_furniture_and_structure() -> None:
    room = _room([TABLE_BOX, SHELF_BOX], [TABLE, SHELF])
    spy = ScriptedLabeler(inner=FakeLabeler(COLORS))

    extract_landmarks(room.reconstruction, room.evidence, spy, LOOSE)

    assert spy.vocabularies == [list(FURNITURE + STRUCTURE)]
    assert spy.crop_counts[0] >= 2


def test_a_wall_is_removed_before_the_clusters_are_labeled() -> None:
    room = _room([TABLE_BOX, WALL], [TABLE, SHELF])

    removed = _find(room)
    kept = _find(
        room, dataclasses.replace(LOOSE, walls=WallConfig(max_planes=0))
    )

    assert removed == [("landmark-0", "table")]
    assert sorted(label for _, label in kept) == ["shelf", "table"]


def test_the_wall_removal_is_not_tied_to_the_seed() -> None:
    room = _room([TABLE_BOX, WALL], [TABLE, SHELF])

    for seed in (1, 2, 3):
        assert _find(
            room, dataclasses.replace(LOOSE, walls=WallConfig(seed=seed))
        ) == [("landmark-0", "table")]


def test_a_table_beside_a_wall_line_survives() -> None:
    room = _room([BESIDE_BOX, WALL], [TABLE, SHELF])

    found = extract_landmarks(
        room.reconstruction,
        room.evidence,
        FakeLabeler({"table": TABLE}),
        LOOSE,
    )

    (table,) = found
    assert table.box.center[0] == pytest.approx(1.65, abs=0.2)


def test_the_floor_is_dropped() -> None:
    room = _room([TABLE_BOX], [TABLE], ground_color=SHELF)
    huge = dataclasses.replace(LOOSE, max_extent=50.0)
    with_floor = dataclasses.replace(huge, floor_clearance=0.0)

    assert _find(room, huge) == [("landmark-0", "table")]
    found = extract_landmarks(
        room.reconstruction, room.evidence, FakeLabeler(COLORS), with_floor
    )
    assert max(item.box.extent.max() for item in found) > 3.0


def test_points_above_the_maximum_height_are_dropped() -> None:
    tall = Cuboid((-0.4, -0.2, 0.0), (0.0, 0.2, 2.1))
    room = _room([tall], [TABLE])
    lowered = dataclasses.replace(LOOSE, max_height=0.5, min_extent=0.2)

    (full,) = extract_landmarks(
        room.reconstruction, room.evidence, FakeLabeler(COLORS), LOOSE
    )
    (cut,) = extract_landmarks(
        room.reconstruction, room.evidence, FakeLabeler(COLORS), lowered
    )

    assert full.box.center[2] + full.box.extent[2] / 2 > 1.5
    assert cut.box.center[2] + cut.box.extent[2] / 2 <= 0.5 + 1e-6


@pytest.mark.parametrize("state", [PointState.CHANGED, PointState.UNKNOWN])
def test_objects_that_are_not_confirmed_static_are_excluded(
    state: PointState,
) -> None:
    room = _room([TABLE_BOX, SHELF_BOX], [TABLE, SHELF], {1: state})

    assert _find(room) == [("landmark-0", "table")]


def test_a_cluster_that_is_too_small_is_dropped() -> None:
    cube = Cuboid((0.0, -0.8, 0.0), (0.15, -0.65, 0.2))
    room = _room([TABLE_BOX, cube], [TABLE, SHELF])

    assert _find(room) == [("landmark-0", "table")]
    small_ok = dataclasses.replace(LOOSE, min_extent=0.1)
    assert len(_find(room, small_ok)) == 2


def test_a_low_flat_patch_is_not_a_landmark() -> None:
    patch = Cuboid((0.0, -1.3, 0.0), (0.8, -0.7, 0.12))
    room = _room([TABLE_BOX, patch], [TABLE, SHELF])

    assert _find(room) == [("landmark-0", "table")]
    flat_ok = dataclasses.replace(LOOSE, min_height=0.05)
    assert sorted(label for _, label in _find(room, flat_ok)) == [
        "shelf",
        "table",
    ]


def test_a_static_cluster_inside_a_changed_object_is_dropped() -> None:
    room = _room([TABLE_BOX, SHELF_BOX], [TABLE, SHELF])
    points = room.reconstruction.points
    inside = SHELF_BOX.low[0] <= points[:, 0]
    inside &= points[:, 0] <= SHELF_BOX.high[0]
    inside &= (points[:, 1] >= SHELF_BOX.low[1] - 0.01) & (
        points[:, 1] <= SHELF_BOX.high[1] + 0.01
    )
    inside &= points[:, 2] > 0.05
    states = room.evidence.before.states.copy()
    states[inside] = PointState.CHANGED
    leak = np.random.default_rng(0).choice(
        np.flatnonzero(inside), int(0.2 * inside.sum()), replace=False
    )
    states[leak] = PointState.PRESENT
    evidence = PointEvidence(room.evidence.before.counts, states)
    mixed = Room(
        room.reconstruction, CrossEvidence(evidence, evidence), room.scene
    )

    kept = _find(mixed, dataclasses.replace(LOOSE, max_changed_share=1.0))
    dropped = _find(mixed)

    assert [label for _, label in kept] == ["table", "shelf"]
    assert dropped == [("landmark-0", "table")]


def test_a_cluster_that_is_too_large_is_dropped() -> None:
    room = _room([TABLE_BOX, SHELF_BOX], [TABLE, SHELF])

    assert _find(room, dataclasses.replace(LOOSE, max_extent=1.0)) == [
        ("landmark-0", "table")
    ]


def test_a_cluster_with_too_few_points_is_dropped() -> None:
    room = _room([TABLE_BOX], [TABLE])

    assert _find(room, dataclasses.replace(LOOSE, min_points=10**6)) == []


def test_a_cluster_that_is_not_furniture_is_not_a_landmark() -> None:
    room = _room([TABLE_BOX, CRATE_BOX], [TABLE, DOOR])

    assert _find(room) == [("landmark-0", "table")]


@pytest.mark.parametrize(("score", "expected"), [(0.2, 0), (0.3, 1)])
def test_a_cluster_below_the_minimum_score_is_not_a_landmark(
    score: float, expected: int
) -> None:
    room = _room([TABLE_BOX], [TABLE])
    lenient = dataclasses.replace(
        LOOSE, labeling=dataclasses.replace(LOOSE.labeling, min_score=0.0)
    )

    found = extract_landmarks(
        room.reconstruction,
        room.evidence,
        ScriptedLabeler(fixed=("table", score)),
        lenient,
    )

    assert len(found) == expected


def test_the_result_is_deterministic() -> None:
    room = _room([TABLE_BOX, SHELF_BOX, WALL], [TABLE, SHELF, SHELF])

    first = extract_landmarks(
        room.reconstruction, room.evidence, FakeLabeler(COLORS), LOOSE
    )
    second = extract_landmarks(
        room.reconstruction, room.evidence, FakeLabeler(COLORS), LOOSE
    )

    assert [i.landmark_id for i in first] == [i.landmark_id for i in second]
    for a, b in zip(first, second, strict=True):
        assert a.label == b.label
        np.testing.assert_array_equal(a.box.center, b.box.center)
        np.testing.assert_array_equal(a.box.extent, b.box.extent)


def test_no_static_points_give_no_landmarks() -> None:
    room = _room(
        [TABLE_BOX], [TABLE], {-1: PointState.CHANGED, 0: PointState.CHANGED}
    )

    assert _find(room) == []


def test_a_reconstruction_of_the_after_epoch_is_rejected() -> None:
    room = _room([TABLE_BOX], [TABLE], epoch=Epoch.AFTER)

    with pytest.raises(ValueError, match="BEFORE"):
        _find(room)


def test_evidence_of_another_length_is_rejected() -> None:
    room = _room([TABLE_BOX], [TABLE])
    short = PointEvidence(
        room.evidence.before.counts[:5], room.evidence.before.states[:5]
    )

    with pytest.raises(ValueError, match="evidence"):
        extract_landmarks(
            room.reconstruction,
            CrossEvidence(short, short),
            FakeLabeler(COLORS),
            LOOSE,
        )


def test_the_landmark_score_threshold_is_its_own() -> None:
    room = _room([TABLE_BOX], [TABLE])
    strict = dataclasses.replace(LOOSE, min_score=0.6)

    found = extract_landmarks(
        room.reconstruction,
        room.evidence,
        ScriptedLabeler(fixed=("table", 0.5)),
        strict,
    )

    assert found == ()


def test_a_changed_point_far_away_does_not_break_the_changed_count() -> None:
    cells = np.array([[0, 0, 0], [1, 0, 0]], np.int64)
    changed = np.array([[1, 1, 0], [10**7, 10**7, 10**7]], np.int64)

    counts = _changed_near(cells, np.array([0, 0]), changed, 1, 2)

    assert counts.tolist() == [1]


def test_changed_points_count_once_per_cluster_within_the_radius() -> None:
    cells = np.array([[0, 0, 0], [1, 0, 0], [20, 0, 0]], np.int64)
    owner = np.array([0, 0, 1])
    changed = np.array(
        [[0, 1, 0], [3, 0, 0], [4, 0, 0], [18, 0, 0], [30, 0, 0]], np.int64
    )

    counts = _changed_near(cells, owner, changed, 2, 2)

    assert counts.tolist() == [2, 1]
