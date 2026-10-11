import itertools
from collections import deque
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from spatial_change.change import (
    CandidateConfig,
    extract_candidates,
    find_candidates,
)
from spatial_change.domain import Epoch, EpochReconstruction
from spatial_change.domain.report import EvidenceCounts
from spatial_change.visibility import CrossEvidence, PointEvidence, PointState

VOXEL = 0.1
LOOSE = CandidateConfig(
    voxel_size=VOXEL,
    link_radius=1,
    min_points=1,
    min_voxels=1,
    min_changed_fraction=0.0,
)
STATE_COUNTS = {
    PointState.CHANGED: (0, 3, 0, 1),
    PointState.PRESENT: (3, 0, 0, 1),
    PointState.UNKNOWN: (0, 0, 0, 4),
}
Part = tuple[NDArray[np.float64], PointState]


def block(
    low: tuple[int, int, int], high: tuple[int, int, int]
) -> NDArray[np.int64]:
    """Lists the voxel keys of a box of voxels, upper bound excluded."""
    ranges = [range(a, b) for a, b in zip(low, high, strict=True)]
    return np.array(list(itertools.product(*ranges)), dtype=np.int64)


def cells(
    keys: ArrayLike, per_voxel: int = 1, size: float = VOXEL
) -> NDArray[np.float64]:
    """Places points strictly inside the given voxels."""
    voxels = np.asarray(keys, dtype=np.float64).reshape(-1, 3)
    inside = (np.arange(per_voxel) + 0.5) / per_voxel
    points = voxels[:, None, :] + inside[None, :, None]
    return points.reshape(-1, 3) * size


def evidence_of(states: ArrayLike) -> PointEvidence:
    codes = np.asarray(states, dtype=np.int8)
    counts = [STATE_COUNTS[PointState(s)] for s in codes]
    return PointEvidence(np.array(counts, dtype=np.int32).reshape(-1, 4), codes)


def scene(*parts: Part) -> tuple[NDArray[np.float64], PointEvidence]:
    points = np.concatenate([p for p, _ in parts])
    states = np.concatenate(
        [np.full(len(p), s, dtype=np.int8) for p, s in parts]
    )
    return points, evidence_of(states)


def changed_cloud(keys: ArrayLike, per_voxel: int = 1) -> Part:
    return cells(keys, per_voxel), PointState.CHANGED


@pytest.mark.parametrize("epoch", list(Epoch))
def test_two_separated_blobs_give_two_candidates(epoch: Epoch) -> None:
    small = cells(block((10, 0, 0), (12, 2, 2)), 2)
    big = cells(block((0, 0, 0), (3, 3, 3)), 2)
    floor = cells(block((-5, -5, -3), (25, 25, -2)))
    points, evidence = scene(
        (small, PointState.CHANGED),
        (big, PointState.CHANGED),
        (floor, PointState.PRESENT),
    )

    found = extract_candidates(points, evidence, epoch, LOOSE)

    assert [c.candidate_id for c in found] == [
        f"{epoch.value}-0",
        f"{epoch.value}-1",
    ]
    first, second = found
    assert first.epoch == second.epoch == epoch
    np.testing.assert_array_equal(first.point_indices, np.arange(16, 70))
    np.testing.assert_array_equal(second.point_indices, np.arange(16))
    assert first.point_indices.dtype == np.int64
    assert (first.point_count, first.voxel_count) == (54, 27)
    assert (second.point_count, second.voxel_count) == (16, 8)
    assert first.evidence == EvidenceCounts(
        consistent=0, free=162, occluded=0, unobserved=54
    )
    assert second.evidence == EvidenceCounts(
        consistent=0, free=48, occluded=0, unobserved=16
    )
    assert first.changed_fraction == second.changed_fraction == 1.0
    for candidate in found:
        np.testing.assert_array_equal(
            candidate.points, points[candidate.point_indices]
        )
        assert candidate.box.contains(candidate.points).all()


GAP = np.concatenate([block((0, 0, 0), (3, 1, 1)), block((4, 0, 0), (7, 1, 1))])


@pytest.mark.parametrize(
    ("keys", "radius", "voxel_counts"),
    [
        pytest.param(GAP, 1, [3, 3], id="gap-of-one-voxel-radius-1"),
        pytest.param(GAP, 2, [6], id="gap-of-one-voxel-radius-2"),
        pytest.param(
            [(i, i, i) for i in range(4)], 1, [4], id="diagonal-neighbours"
        ),
        pytest.param(
            [(0, 0, 0), (2, 2, 2)], 1, [1, 1], id="two-diagonal-steps-apart"
        ),
    ],
)
def test_voxels_link_within_the_radius(
    keys: ArrayLike, radius: int, voxel_counts: list[int]
) -> None:
    points, evidence = scene(changed_cloud(keys))
    config = replace(LOOSE, link_radius=radius)

    found = extract_candidates(points, evidence, Epoch.BEFORE, config)

    assert [c.voxel_count for c in found] == voxel_counts


def test_voxel_keys_floor_toward_negative_infinity() -> None:
    points, evidence = scene(changed_cloud(block((-2, -2, -2), (2, 2, 2))))

    (found,) = extract_candidates(points, evidence, Epoch.BEFORE, LOOSE)

    assert found.voxel_count == 64


def test_a_long_chain_of_voxels_is_one_candidate() -> None:
    keys = [(i, 0, 0) for i in range(300)]
    points, evidence = scene(changed_cloud(keys))

    (found,) = extract_candidates(points, evidence, Epoch.BEFORE, LOOSE)

    assert found.voxel_count == 300
    assert found.point_count == 300


def reference_components(
    keys: NDArray[np.int64], radius: int
) -> list[frozenset[int]]:
    """Groups voxel rows by a breadth-first search over the neighbourhood."""
    row_of = {tuple(k): i for i, k in enumerate(keys.tolist())}
    steps = [
        d
        for d in itertools.product(range(-radius, radius + 1), repeat=3)
        if any(d)
    ]
    seen: set[int] = set()
    groups = []
    for start in range(len(keys)):
        if start in seen:
            continue
        seen.add(start)
        queue, group = deque([start]), {start}
        while queue:
            x, y, z = keys[queue.popleft()].tolist()
            for dx, dy, dz in steps:
                row = row_of.get((x + dx, y + dy, z + dz))
                if row is not None and row not in seen:
                    seen.add(row)
                    group.add(row)
                    queue.append(row)
        groups.append(frozenset(group))
    return groups


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(("radius", "density"), [(1, 0.05), (2, 0.012)])
def test_components_match_a_reference_search(
    seed: int, radius: int, density: float
) -> None:
    rng = np.random.default_rng(seed)
    keys = np.argwhere(rng.random((16, 16, 16)) < density).astype(np.int64)
    points, evidence = scene(changed_cloud(keys, 2))
    config = replace(LOOSE, link_radius=radius)

    found = extract_candidates(points, evidence, Epoch.BEFORE, config)

    got = {frozenset(c.point_indices.tolist()) for c in found}
    expected = {
        frozenset(p for row in group for p in (2 * row, 2 * row + 1))
        for group in reference_components(keys, radius)
    }
    assert len(expected) > 1
    assert got == expected


def test_a_blob_below_the_point_threshold_is_dropped() -> None:
    config = replace(LOOSE, min_points=10)
    nine = changed_cloud(block((0, 0, 0), (3, 1, 1)), 3)
    ten = changed_cloud(block((0, 0, 0), (2, 1, 1)), 5)

    assert not extract_candidates(*scene(nine), Epoch.BEFORE, config)
    (kept,) = extract_candidates(*scene(ten), Epoch.BEFORE, config)
    assert kept.point_count == 10


def test_a_blob_below_the_voxel_threshold_is_dropped() -> None:
    config = replace(LOOSE, min_voxels=3)
    two_voxels = changed_cloud(block((0, 0, 0), (2, 1, 1)), 20)
    three_voxels = changed_cloud(block((0, 0, 0), (3, 1, 1)), 1)

    assert not extract_candidates(*scene(two_voxels), Epoch.BEFORE, config)
    (kept,) = extract_candidates(*scene(three_voxels), Epoch.BEFORE, config)
    assert kept.voxel_count == 3


def speckled_plane() -> tuple[NDArray[np.float64], PointEvidence]:
    """A dense PRESENT plane with a thin patch of CHANGED points in it.

    The patch spans three voxels with 4 CHANGED among 45 points each.
    """
    patch = block((0, 0, 0), (3, 1, 1))
    present = cells(patch, 36)
    specks = cells(patch, 4)
    rest = cells(block((-6, -6, 0), (10, 10, 1)), 5)
    return scene(
        (rest, PointState.PRESENT),
        (present, PointState.PRESENT),
        (specks, PointState.CHANGED),
    )


def test_speckles_on_a_dense_present_plane_are_dropped() -> None:
    config = CandidateConfig(voxel_size=VOXEL, min_points=10, min_voxels=3)
    points, evidence = speckled_plane()

    assert not extract_candidates(points, evidence, Epoch.BEFORE, config)


def test_the_same_speckles_survive_without_a_changed_fraction() -> None:
    config = replace(LOOSE, min_points=10, min_voxels=3)
    points, evidence = speckled_plane()

    (found,) = extract_candidates(points, evidence, Epoch.BEFORE, config)

    assert found.point_count == 12
    assert found.changed_fraction < 0.3


def test_the_changed_fraction_ignores_unknown_points() -> None:
    inside = block((0, 0, 0), (3, 1, 1))
    points, evidence = scene(
        (cells(inside[:1], 6), PointState.CHANGED),
        (cells(inside[:1], 2), PointState.PRESENT),
        (cells(inside[1:2], 1), PointState.CHANGED),
        (cells(inside[1:2], 1), PointState.UNKNOWN),
        (cells(inside[2:], 1), PointState.CHANGED),
        (cells([(3, 0, 0)], 5), PointState.PRESENT),
    )

    (found,) = extract_candidates(points, evidence, Epoch.BEFORE, LOOSE)

    assert found.point_count == 8
    assert found.changed_fraction == pytest.approx(8 / 10)


def test_a_partly_observed_region_is_kept_whatever_the_unknown_share() -> None:
    inside = block((0, 0, 0), (3, 3, 3))
    points, evidence = scene(
        (cells(inside, 1), PointState.CHANGED),
        (cells(inside, 3), PointState.UNKNOWN),
    )
    config = CandidateConfig(voxel_size=VOXEL)

    (found,) = extract_candidates(points, evidence, Epoch.BEFORE, config)

    assert found.point_count == 27
    assert found.changed_fraction == 1.0


def test_the_changed_fraction_threshold_is_inclusive() -> None:
    patch = block((0, 0, 0), (1, 1, 1))
    points, evidence = scene(
        (cells(patch, 3), PointState.CHANGED),
        (cells(patch, 7), PointState.PRESENT),
    )
    config = replace(LOOSE, min_changed_fraction=0.3)

    assert extract_candidates(points, evidence, Epoch.BEFORE, config)


def test_equal_sized_candidates_rank_by_voxel_key() -> None:
    right = cells(block((10, 0, 0), (12, 2, 2)), 2)
    left = cells(block((0, 0, 0), (2, 2, 2)), 2)
    points, evidence = scene(
        (right, PointState.CHANGED), (left, PointState.CHANGED)
    )

    first, second = extract_candidates(points, evidence, Epoch.BEFORE, LOOSE)

    np.testing.assert_array_equal(first.point_indices, np.arange(16, 32))
    np.testing.assert_array_equal(second.point_indices, np.arange(16))


def test_permuting_the_points_gives_the_same_candidates() -> None:
    rng = np.random.default_rng(5)
    points, evidence = scene(
        (cells(block((0, 0, 0), (3, 3, 2)), 3), PointState.CHANGED),
        (cells(block((8, 0, 0), (11, 2, 2)), 3), PointState.CHANGED),
        (cells(block((0, 8, 0), (2, 10, 2)), 3), PointState.CHANGED),
        (cells(block((0, 0, 1), (3, 3, 3)), 2), PointState.PRESENT),
        (cells(block((20, 0, 0), (22, 2, 2)), 2), PointState.UNKNOWN),
    )
    perm = rng.permutation(len(points))
    shuffled = evidence_of(evidence.states[perm])
    config = CandidateConfig(voxel_size=VOXEL, min_points=5, min_voxels=3)

    a = extract_candidates(points, evidence, Epoch.AFTER, config)
    b = extract_candidates(points[perm], shuffled, Epoch.AFTER, config)

    assert len(a) == len(b) == 3
    for x, y in zip(a, b, strict=True):
        assert x.candidate_id == y.candidate_id
        np.testing.assert_array_equal(
            x.point_indices, np.sort(perm[y.point_indices])
        )
        assert x.evidence == y.evidence
        assert x.voxel_count == y.voxel_count
        assert x.changed_fraction == pytest.approx(y.changed_fraction)
        np.testing.assert_allclose(x.box.center, y.box.center, atol=1e-9)
        np.testing.assert_allclose(x.box.extent, y.box.extent, atol=1e-9)


def test_the_defaults_keep_a_real_blob_and_drop_a_speck() -> None:
    blob = cells(block((0, 0, 0), (4, 4, 4)), 2, 0.05)
    speck = cells([(40, 0, 0)], 5, 0.05)
    points, evidence = scene(
        (blob, PointState.CHANGED), (speck, PointState.CHANGED)
    )

    (found,) = extract_candidates(points, evidence, Epoch.BEFORE)

    assert found.point_count == 128
    assert found.voxel_count == 64
    assert found.box.contains(found.points).all()


def test_an_empty_cloud_gives_no_candidates() -> None:
    assert (
        extract_candidates(np.zeros((0, 3)), evidence_of([]), Epoch.BEFORE)
        == ()
    )


def test_a_cloud_without_changed_points_gives_no_candidates() -> None:
    points, evidence = scene(
        (cells(block((0, 0, 0), (4, 4, 4)), 2), PointState.PRESENT),
        (cells(block((9, 0, 0), (12, 2, 2)), 2), PointState.UNKNOWN),
    )

    assert extract_candidates(points, evidence, Epoch.BEFORE, LOOSE) == ()


def test_evidence_of_another_length_is_rejected() -> None:
    points, evidence = scene(changed_cloud(block((0, 0, 0), (2, 2, 2))))

    with pytest.raises(ValueError, match="evidence"):
        extract_candidates(points[:-1], evidence, Epoch.BEFORE, LOOSE)


@pytest.mark.parametrize(
    "points", [np.zeros((4, 2)), np.array([[0.0, 0.0, np.nan]])]
)
def test_invalid_points_are_rejected(points: NDArray[np.float64]) -> None:
    points, evidence = scene((points, PointState.CHANGED))

    with pytest.raises(ValueError, match="points"):
        extract_candidates(points, evidence, Epoch.BEFORE, LOOSE)


def reconstruction(epoch: Epoch, points: ArrayLike) -> EpochReconstruction:
    """Wraps points in a reconstruction without frames."""
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return EpochReconstruction(
        epoch=epoch,
        frames=[],
        points=cloud,
        colors=np.zeros((len(cloud), 3), dtype=np.uint8),
        confidence=np.ones(len(cloud), dtype=np.float32),
    )


def test_find_candidates_routes_each_epoch_to_its_direction() -> None:
    before_points, before_evidence = scene(
        changed_cloud(block((0, 0, 0), (3, 3, 3)), 2),
        (cells(block((-5, -5, -3), (5, 5, -2))), PointState.PRESENT),
    )
    after_points, after_evidence = scene(
        changed_cloud(block((0, 0, 0), (2, 2, 2))),
        changed_cloud(block((10, 0, 0), (12, 2, 2)), 2),
    )

    result = find_candidates(
        reconstruction(Epoch.BEFORE, before_points),
        reconstruction(Epoch.AFTER, after_points),
        CrossEvidence(before_evidence, after_evidence),
        LOOSE,
    )

    (gone,) = result.disappeared
    assert gone.epoch == Epoch.BEFORE
    assert gone.candidate_id == "before-0"
    assert gone.point_count == 54
    np.testing.assert_array_equal(
        gone.points, before_points[gone.point_indices]
    )
    new = result.appeared
    assert [c.epoch for c in new] == [Epoch.AFTER, Epoch.AFTER]
    assert [c.candidate_id for c in new] == ["after-0", "after-1"]
    assert [c.point_count for c in new] == [16, 8]
    np.testing.assert_array_equal(
        new[1].points, after_points[new[1].point_indices]
    )


def test_find_candidates_uses_the_default_config() -> None:
    points, evidence = scene(changed_cloud(block((0, 0, 0), (4, 4, 4)), 2))
    nothing = evidence_of([])

    result = find_candidates(
        reconstruction(Epoch.BEFORE, points / 2.0),
        reconstruction(Epoch.AFTER, []),
        CrossEvidence(evidence, nothing),
    )

    assert len(result.disappeared) == 1
    assert result.appeared == ()


def test_find_candidates_rejects_swapped_epochs() -> None:
    before = reconstruction(Epoch.BEFORE, [])
    after = reconstruction(Epoch.AFTER, [])
    nothing = evidence_of([])
    cross = CrossEvidence(nothing, nothing)

    with pytest.raises(ValueError, match="before must be the BEFORE"):
        find_candidates(after, after, cross)
    with pytest.raises(ValueError, match="after must be the AFTER"):
        find_candidates(before, before, cross)


def test_a_span_beyond_the_key_range_is_rejected() -> None:
    points, evidence = scene(
        (np.array([[0.0, 0.0, 0.0], [1e9, 1e9, 1e9]]), PointState.CHANGED)
    )

    with pytest.raises(ValueError, match="too many voxels"):
        extract_candidates(points, evidence, Epoch.BEFORE, LOOSE)


NEAR = [[0.01 * i, 0.0, 0.0] for i in range(20)]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "rows",
    [
        [[4e17, 0.0, 0.0], [-4e17, 0.0, 0.0], *NEAR],
        [[1.7e308, 0.0, 0.0], *NEAR],
        [[1.7e308, 0.0, 0.0]] * 20,
    ],
    ids=["int64-overflow", "infinite-key", "all-infinite-keys"],
)
def test_changed_outliers_beyond_the_int64_keys_are_rejected(
    rows: list[list[float]],
) -> None:
    points, evidence = scene((np.array(rows), PointState.CHANGED))
    config = CandidateConfig(voxel_size=0.05)

    with pytest.raises(ValueError, match="too many voxels"):
        extract_candidates(points, evidence, Epoch.BEFORE, config)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("far", [1e300, -1e300, 1.7e308])
def test_a_far_unchanged_point_leaves_the_candidates_alone(far: float) -> None:
    blob = changed_cloud(block((0, 0, 0), (3, 3, 3)), 2)
    outlier = (np.array([[far, 0.0, 0.0]]), PointState.PRESENT)
    expected = extract_candidates(*scene(blob), Epoch.BEFORE, LOOSE)

    found = extract_candidates(*scene(blob, outlier), Epoch.BEFORE, LOOSE)

    assert len(found) == len(expected) == 1
    np.testing.assert_array_equal(
        found[0].point_indices, expected[0].point_indices
    )
    assert found[0].voxel_count == expected[0].voxel_count
    assert found[0].changed_fraction == expected[0].changed_fraction


@pytest.mark.filterwarnings("error")
def test_changed_points_far_from_the_origin_do_not_overflow_the_keys() -> None:
    points, evidence = scene(
        (cells(block((0, 0, 0), (3, 3, 3)), 2) + 1e18, PointState.CHANGED)
    )

    found = extract_candidates(points, evidence, Epoch.BEFORE, LOOSE)

    assert [c.point_count for c in found] == [len(points)]
