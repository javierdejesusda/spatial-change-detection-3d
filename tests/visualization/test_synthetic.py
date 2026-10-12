from pathlib import Path

import pytest
from rrd_reading import Rrd
from visualization_scenes import synthetic_scene

from spatial_change.domain import ChangeType
from spatial_change.visualization import RerunConfig, save_recording

SEED = 0


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Rrd]:
    path = tmp_path_factory.mktemp("synthetic") / "scene.rrd"
    save_recording(synthetic_scene(SEED), path, RerunConfig())
    return path, Rrd(path)


def test_the_recording_is_a_nonempty_file_that_opens_again(
    recorded: tuple[Path, Rrd],
) -> None:
    path, rrd = recorded

    assert path.stat().st_size > 100_000
    assert "/world/before/points" in rrd.entities
    assert "/world/after/points" in rrd.entities


def test_the_three_changes_appear_with_their_type_specific_layers(
    recorded: tuple[Path, Rrd],
) -> None:
    _, rrd = recorded
    changes = synthetic_scene(SEED).report.changes

    assert sorted(c.type for c in changes) == [
        ChangeType.ADDED,
        ChangeType.MOVED,
        ChangeType.REMOVED,
    ]
    for change in changes:
        base = f"/world/changes/{change.change_id}"
        assert f"{base}/box" in rrd.entities
        moved = change.type is ChangeType.MOVED
        assert (f"{base}/ghost" in rrd.entities) is moved
        assert (f"{base}/arrow" in rrd.entities) is moved


def test_the_unobserved_zone_both_trajectories_and_panels_are_there(
    recorded: tuple[Path, Rrd],
) -> None:
    _, rrd = recorded

    assert "/world/unknown/unobserved-0" in rrd.entities
    assert "/world/before/trajectory" in rrd.entities
    assert "/world/after/trajectory" in rrd.entities
    assert any("/cameras/" in e and e.endswith("/image") for e in rrd.entities)
    assert any("/overlay/" in e for e in rrd.entities)
    assert ("3D", "Scene", "world") in rrd.views()
