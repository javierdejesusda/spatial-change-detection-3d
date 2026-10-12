"""A small hand-built report and clouds for the GLB and report tests."""

import json
import re
from typing import Any

import numpy as np

from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeReport,
    ChangeType,
    Epoch,
    EvidenceCounts,
    OrientedBox,
    UnobservedRegion,
)
from spatial_change.reporting import ChangeCrops, ReportInputs
from spatial_change.visualization import ScenePoints

EVIDENCE = EvidenceCounts(consistent=5, free=3, occluded=1, unobserved=0)


def box(
    center: tuple[float, float, float],
    extent: tuple[float, float, float] = (1.0, 0.5, 0.8),
) -> BoxModel:
    """Builds an axis-aligned box model."""
    return BoxModel.from_box(
        OrientedBox(np.array(center), np.array(extent), np.eye(3))
    )


def change(
    change_id: str,
    kind: ChangeType,
    label: str,
    before: BoxModel | None,
    after: BoxModel | None,
    description: str = "It changed.",
) -> Change:
    """Builds a change with fixed numbers."""
    moved = kind is ChangeType.MOVED
    return Change(
        change_id=change_id,
        type=kind,
        label=label,
        description=description,
        confidence=0.87,
        before_box=before,
        after_box=after,
        displacement_m=1.25 if moved else None,
        rotation_deg=30.0 if moved else None,
        evidence=EVIDENCE,
    )


def one_of_each_report(description: str = "It changed.") -> ChangeReport:
    """Returns a report with one change of every type and one region."""
    return ChangeReport(
        changes=[
            change(
                "c1",
                ChangeType.MOVED,
                "chair",
                box((1.0, 1.0, 0.4)),
                box((2.0, 1.75, 0.4)),
                description,
            ),
            change("c2", ChangeType.ADDED, "lamp", None, box((3, 3, 0.5))),
            change("c3", ChangeType.REMOVED, "crate", box((0, 4, 0.5)), None),
            change(
                "c4",
                ChangeType.UNCERTAIN,
                "rug",
                box((4, 1, 0.05)),
                box((4, 1, 0.05)),
            ),
        ],
        unobserved_regions=[
            UnobservedRegion(
                region_id="u1", box=box((5, 5, 1)), epoch_seen=Epoch.BEFORE
            )
        ],
        metadata={"frames_before": 12, "frames_after": 12},
    )


def empty_report() -> ChangeReport:
    """Returns a report with no changes and no regions."""
    return ChangeReport(changes=[], unobserved_regions=[], metadata={})


def cloud(count: int, seed: int) -> ScenePoints:
    """Returns random colored points in a 6 x 5 x 2.5 m room."""
    rng = np.random.default_rng(seed)
    points = rng.uniform((0, 0, 0), (6, 5, 2.5), size=(count, 3))
    colors = rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
    return ScenePoints(points, colors)


def inputs(
    report: ChangeReport | None = None,
    summary: str = "Two captures of a living room.",
    crops: dict[str, ChangeCrops] | None = None,
    count: int = 300,
) -> ReportInputs:
    """Builds report inputs with random clouds for both epochs."""
    return ReportInputs(
        report=report or one_of_each_report(),
        summary=summary,
        crops=crops or {},
        points={Epoch.BEFORE: cloud(count, 0), Epoch.AFTER: cloud(count, 1)},
    )


def scene_data(page: str) -> dict[str, Any]:
    """Parses the JSON data block of a rendered page."""
    match = re.search(r'id="scene-data">(.*?)</script>', page, re.S)
    assert match is not None
    parsed: dict[str, Any] = json.loads(match.group(1))
    return parsed


def cards(page: str) -> list[str]:
    """Returns the markup of each change card of a rendered page."""
    return re.findall(r'<article class="card"[^>]*>.*?</article>', page, re.S)
