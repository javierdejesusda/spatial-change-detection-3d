import json
import re
import time
from pathlib import Path

import pytest
from report_pipeline import crops_for, synthetic_report
from report_scenes import box, change
from visibility_scenes import scene_evidence

from spatial_change.domain import BoxModel, ChangeReport, ChangeType
from spatial_change.reporting import write_report
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SyntheticObject,
    SyntheticScenePair,
)
from spatial_change.visibility import VisibilityConfig
from spatial_change.visualization import write_glb

# Measured on seeds 0 to 2 (128x96 frames, 12 per epoch, 147,456 points per
# epoch): the page is 0.88 MB and written in 0.04 s, the GLB is 4.72 MB
# (the 200,000 point cap is not reached) and written in 0.03 to 0.04 s. The
# bounds leave room for slower machines.
MAX_PAGE_BYTES = 2_000_000
MAX_GLB_BYTES = 6_000_000
MAX_SECONDS = 20.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_synthetic_scene_pair_gives_a_report_and_a_scene(
    seed: int, tmp_path: Path
) -> None:
    built = synthetic_report(seed)
    inputs = built.inputs
    page_path = tmp_path / "report.html"
    glb_path = tmp_path / "scene.glb"

    started = time.perf_counter()
    write_report(inputs, page_path)
    write_glb(inputs.points, inputs.report, glb_path)
    seconds = time.perf_counter() - started

    page = page_path.read_text(encoding="utf-8")
    ids = re.findall(r'<article class="card" id="change-([^"]+)"', page)
    assert ids == [c.change_id for c in inputs.report.changes]
    assert sorted(c.type for c in inputs.report.changes) == [
        ChangeType.ADDED,
        ChangeType.MOVED,
        ChangeType.REMOVED,
    ]
    boxes = sum(
        (c.before_box is not None) + (c.after_box is not None)
        for c in inputs.report.changes
    )
    assert page.count("<img ") == boxes
    assert "zone-1" in page
    assert "http://" not in page and "https://" not in page
    assert 0 < page_path.stat().st_size < MAX_PAGE_BYTES
    assert 0 < glb_path.stat().st_size < MAX_GLB_BYTES
    assert seconds < MAX_SECONDS
    data = glb_path.read_bytes()
    assert data[:4] == b"glTF"
    assert len(data) == glb_path.stat().st_size
    assert b"unobserved-zone-1" in data


def test_the_changes_survive_the_json_report_round_trip() -> None:
    inputs = synthetic_report(0).inputs

    again = ChangeReport.from_json(inputs.report.to_json())

    assert again == inputs.report
    assert json.loads(inputs.report.to_json())["schema_version"] == "1"


def _pair() -> SyntheticScenePair:
    return scene_evidence(0, VisibilityConfig()).pair


def _truth(pair: SyntheticScenePair, status: ObjectStatus) -> SyntheticObject:
    (found,) = [o for o in pair.objects if o.status is status]
    return found


def test_crops_are_matched_to_the_object_under_the_boxes() -> None:
    pair = _pair()
    chair = _truth(pair, ObjectStatus.MOVED)
    assert chair.box_before is not None and chair.box_after is not None
    unsure = change(
        "c1",
        ChangeType.UNCERTAIN,
        "chair",
        BoxModel.from_box(chair.box_before),
        BoxModel.from_box(chair.box_after),
    )

    crops = crops_for(pair, unsure)

    assert crops.before is not None and crops.after is not None


def test_a_removed_change_has_only_a_before_crop() -> None:
    pair = _pair()
    crate = _truth(pair, ObjectStatus.REMOVED)
    assert crate.box_before is not None
    gone = change(
        "c1",
        ChangeType.REMOVED,
        "crate",
        BoxModel.from_box(crate.box_before),
        None,
    )

    crops = crops_for(pair, gone)

    assert crops.before is not None
    assert crops.after is None


def test_a_change_far_from_every_object_has_no_crops() -> None:
    nowhere = change(
        "c1",
        ChangeType.ADDED,
        "ghost",
        None,
        box((100.0, 100.0, 100.0)),
    )

    crops = crops_for(_pair(), nowhere)

    assert crops.before is None and crops.after is None


def test_a_change_with_no_boxes_has_no_crops() -> None:
    crops = crops_for(
        _pair(), change("c1", ChangeType.UNCERTAIN, "rug", None, None)
    )

    assert crops.before is None and crops.after is None
