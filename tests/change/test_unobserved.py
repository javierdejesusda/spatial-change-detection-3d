import itertools
from dataclasses import replace

import numpy as np
import pytest
from appearance_scenes import reconstruction
from numpy.typing import NDArray
from visibility_scenes import scene_evidence

from spatial_change.change import (
    UnobservedConfig,
    extract_unobserved,
    find_unobserved_regions,
)
from spatial_change.domain import Epoch
from spatial_change.visibility import (
    CrossEvidence,
    PointEvidence,
    PointState,
    VisibilityConfig,
)

VOXEL = 0.1
LOOSE = UnobservedConfig(
    voxel_size=VOXEL,
    link_radius=1,
    min_points=1,
    min_voxels=1,
    min_unobserved_fraction=0.5,
)
HIDDEN = (0, 0, 0, 4)
SEEN = (3, 0, 0, 1)
CHANGED = (0, 3, 0, 1)


def blob(
    low: tuple[int, int, int], high: tuple[int, int, int], per_voxel: int = 2
) -> NDArray[np.float64]:
    """Places points strictly inside a box of voxels, upper bound excluded."""
    ranges = [range(a, b) for a, b in zip(low, high, strict=True)]
    keys = np.array(list(itertools.product(*ranges)), dtype=np.float64)
    inside = (np.arange(per_voxel) + 0.5) / per_voxel
    points = keys[:, None, :] + inside[None, :, None]
    return points.reshape(-1, 3) * VOXEL


def evidence_of(
    parts: list[tuple[NDArray[np.float64], tuple[int, int, int, int]]],
) -> tuple[NDArray[np.float64], PointEvidence]:
    points = np.concatenate([p for p, _ in parts])
    counts = np.concatenate(
        [np.tile(np.array(c, dtype=np.int32), (len(p), 1)) for p, c in parts]
    )
    states = np.where(
        counts[:, 1] >= 2,
        PointState.CHANGED,
        np.where(counts[:, 0] >= 1, PointState.PRESENT, PointState.UNKNOWN),
    ).astype(np.int8)
    return points, PointEvidence(counts, states)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voxel_size", 0.0),
        ("voxel_size", float("inf")),
        ("link_radius", 0),
        ("min_points", 0),
        ("min_voxels", 0),
        ("min_unobserved_fraction", -0.1),
        ("min_unobserved_fraction", 1.1),
        ("min_unobserved_fraction", 0.0),
    ],
)
def test_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        UnobservedConfig(**{field: value})


def test_hidden_blob_gives_one_box_around_its_points() -> None:
    hidden = blob((0, 0, 0), (4, 3, 2))
    seen = blob((10, 0, 0), (14, 3, 2))
    changed = blob((20, 0, 0), (24, 3, 2))
    points, evidence = evidence_of(
        [(hidden, HIDDEN), (seen, SEEN), (changed, CHANGED)]
    )

    boxes = extract_unobserved(points, evidence, LOOSE)

    assert len(boxes) == 1
    assert boxes[0].contains(hidden).all()
    assert not boxes[0].contains(seen).any()
    assert not boxes[0].contains(changed).any()


def test_present_and_changed_points_never_give_a_region() -> None:
    points, evidence = evidence_of(
        [
            (blob((0, 0, 0), (5, 5, 5)), SEEN),
            (blob((9, 0, 0), (14, 5, 5)), CHANGED),
        ]
    )

    assert extract_unobserved(points, evidence, LOOSE) == ()


def test_votes_below_the_unobserved_fraction_do_not_count() -> None:
    half_seen = (0, 0, 2, 2)
    points, evidence = evidence_of([(blob((0, 0, 0), (4, 4, 4)), half_seen)])
    strict = replace(LOOSE, min_unobserved_fraction=0.8)

    assert extract_unobserved(points, evidence, strict) == ()
    assert len(extract_unobserved(points, evidence, LOOSE)) == 1


def test_a_present_point_is_not_unobserved_whatever_its_other_votes() -> None:
    points, evidence = evidence_of([(blob((0, 0, 0), (4, 4, 4)), (1, 0, 0, 9))])

    assert extract_unobserved(points, evidence, LOOSE) == ()


def test_a_point_with_no_votes_is_not_unobserved() -> None:
    points, evidence = evidence_of([(blob((0, 0, 0), (4, 4, 4)), (0, 0, 0, 0))])

    assert extract_unobserved(points, evidence, LOOSE) == ()


def test_clusters_below_the_floors_are_dropped() -> None:
    big = blob((0, 0, 0), (4, 4, 4), per_voxel=2)
    small = blob((20, 0, 0), (22, 1, 1), per_voxel=2)
    points, evidence = evidence_of([(big, HIDDEN), (small, HIDDEN)])
    floors = UnobservedConfig(
        voxel_size=VOXEL,
        link_radius=1,
        min_points=100,
        min_voxels=10,
        min_unobserved_fraction=0.5,
    )

    boxes = extract_unobserved(points, evidence, floors)

    assert len(boxes) == 1
    assert boxes[0].contains(big).all()


def test_nearby_blobs_link_and_distant_ones_stay_apart() -> None:
    near_a = blob((0, 0, 0), (3, 3, 3))
    near_b = blob((4, 0, 0), (7, 3, 3))
    far = blob((30, 0, 0), (33, 3, 3))
    points, evidence = evidence_of(
        [(near_a, HIDDEN), (near_b, HIDDEN), (far, HIDDEN)]
    )

    boxes = extract_unobserved(points, evidence, replace(LOOSE, link_radius=2))

    assert len(boxes) == 2
    assert boxes[0].contains(np.concatenate([near_a, near_b])).all()
    assert boxes[1].contains(far).all()


def test_regions_get_unique_ids_and_their_epoch() -> None:
    before_points, before_votes = evidence_of(
        [(blob((0, 0, 0), (4, 4, 4)), HIDDEN)]
    )
    after_points, after_votes = evidence_of(
        [
            (blob((0, 0, 0), (3, 3, 3)), HIDDEN),
            (blob((20, 0, 0), (22, 2, 2)), HIDDEN),
        ]
    )

    regions = find_unobserved_regions(
        reconstruction(Epoch.BEFORE, [], before_points),
        reconstruction(Epoch.AFTER, [], after_points),
        CrossEvidence(before_votes, after_votes),
        LOOSE,
    )

    assert [r.region_id for r in regions] == [
        "region-0",
        "region-1",
        "region-2",
    ]
    assert [r.epoch_seen for r in regions] == [
        Epoch.BEFORE,
        Epoch.AFTER,
        Epoch.AFTER,
    ]
    extents = [np.prod(r.box.extent) for r in regions]
    assert extents[1] > extents[2]


def test_evidence_rows_must_match_the_points() -> None:
    points, evidence = evidence_of([(blob((0, 0, 0), (2, 2, 2)), HIDDEN)])

    with pytest.raises(ValueError, match="rows"):
        extract_unobserved(points[:-1], evidence, LOOSE)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_zone_is_covered_by_regions_of_the_epoch_that_saw_it(
    seed: int,
) -> None:
    votes = scene_evidence(seed, VisibilityConfig())
    pair = votes.pair
    seen = pair.zone_seen_in
    points = votes.clouds[seen].points
    config = UnobservedConfig(voxel_size=0.1, min_points=20, min_voxels=5)

    boxes = extract_unobserved(points, votes.evidence[seen], config)

    inside_zone = pair.unobserved_zone.contains(points)
    covered = np.zeros(len(points), dtype=bool)
    for box in boxes:
        covered |= box.contains(points)
    assert inside_zone.sum() > 100
    assert covered[inside_zone].all()
