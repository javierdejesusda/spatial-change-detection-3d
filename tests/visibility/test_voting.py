from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from rotations import perturbation

from spatial_change.domain import (
    SE3,
    Epoch,
    EpochReconstruction,
    EvidenceCounts,
    Frame,
    Intrinsics,
    Sim3,
)
from spatial_change.geometry.projection import in_image, project
from spatial_change.visibility import (
    PointEvidence,
    PointState,
    VisibilityConfig,
    Vote,
    vote_epochs,
    vote_points,
)

SIZE = 8
FOCAL = 8.0
CENTER = 4.0
INTRINSICS = Intrinsics(FOCAL, FOCAL, CENTER, CENTER, SIZE, SIZE)
PLANE_Z = 2.0
DEFAULT = VisibilityConfig()
NEIGHBOURHOOD_0 = VisibilityConfig(neighbourhood=0)


def make_frame(
    depth: NDArray[np.float32] | None,
    confidence: NDArray[np.float32] | None = None,
    pose: SE3 | None = None,
    index: int = 0,
) -> Frame:
    return Frame(
        epoch=Epoch.AFTER,
        index=index,
        timestamp_s=0.0,
        intrinsics=INTRINSICS,
        world_from_camera=SE3.identity() if pose is None else pose,
        depth=depth,
        confidence=confidence,
    )


def plane(z: float = PLANE_Z) -> NDArray[np.float32]:
    return np.full((SIZE, SIZE), z, dtype=np.float32)


def vote_one(
    point: tuple[float, float, float],
    frame: Frame,
    config: VisibilityConfig = NEIGHBOURHOOD_0,
) -> Vote:
    evidence = vote_points([point], [frame], config)
    assert evidence.counts.shape == (1, 4)
    assert evidence.counts.sum() == 1
    return Vote(int(np.argmax(evidence.counts[0])))


def test_one_vote_per_frame_per_point() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    frames = [make_frame(plane(), index=i) for i in range(5)]

    evidence = vote_points(points, frames)

    assert evidence.counts.dtype == np.int32
    assert evidence.counts.shape == (3, 4)
    np.testing.assert_array_equal(evidence.counts.sum(axis=1), [5, 5, 5])


def test_point_on_in_front_of_and_behind_the_plane() -> None:
    frame = make_frame(plane())

    assert vote_one((0.0, 0.0, 2.0), frame) == Vote.CONSISTENT
    assert vote_one((0.0, 0.0, 2.05), frame) == Vote.CONSISTENT
    assert vote_one((0.0, 0.0, 1.0), frame) == Vote.FREE
    assert vote_one((0.0, 0.0, 3.0), frame) == Vote.OCCLUDED


def test_point_behind_the_camera_is_unobserved() -> None:
    frame = make_frame(plane())

    assert vote_one((0.0, 0.0, -1.0), frame) == Vote.UNOBSERVED
    assert vote_one((0.0, 0.0, 0.0), frame) == Vote.UNOBSERVED


def test_image_bounds_follow_the_half_pixel_convention() -> None:
    frame = make_frame(plane())
    low = 4.5 * PLANE_Z / FOCAL
    high = 3.5 * PLANE_Z / FOCAL

    assert vote_one((-low, 0.0, PLANE_Z), frame) == Vote.CONSISTENT
    assert vote_one((-low - 1e-3, 0.0, PLANE_Z), frame) == Vote.UNOBSERVED
    assert vote_one((high, 0.0, PLANE_Z), frame) == Vote.UNOBSERVED
    assert vote_one((high - 1e-3, 0.0, PLANE_Z), frame) == Vote.CONSISTENT
    assert vote_one((0.0, -low, PLANE_Z), frame) == Vote.CONSISTENT
    assert vote_one((0.0, -low - 1e-3, PLANE_Z), frame) == Vote.UNOBSERVED
    assert vote_one((0.0, high, PLANE_Z), frame) == Vote.UNOBSERVED
    assert vote_one((0.0, high - 1e-3, PLANE_Z), frame) == Vote.CONSISTENT
    assert vote_one((0.0, 5.0, PLANE_Z), frame) == Vote.UNOBSERVED


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf, -1.0])
def test_invalid_center_depth_is_unobserved(bad: float) -> None:
    depth = plane()
    depth[4, 4] = bad

    assert vote_one((0.0, 0.0, 1.0), make_frame(depth)) == Vote.UNOBSERVED
    assert vote_one((0.0, 0.0, 2.0), make_frame(depth)) == Vote.UNOBSERVED
    assert vote_one((0.0, 0.0, 3.0), make_frame(depth)) == Vote.UNOBSERVED


def test_low_confidence_center_is_unobserved() -> None:
    confidence = np.ones((SIZE, SIZE), dtype=np.float32)
    confidence[4, 4] = 0.0625
    frame = make_frame(plane(), confidence)

    assert vote_one((0.0, 0.0, 2.0), frame) == Vote.UNOBSERVED
    assert vote_one((0.0, 0.0, 1.0), frame) == Vote.UNOBSERVED


def test_confidence_at_the_threshold_is_valid_and_none_acts_as_one() -> None:
    confidence = np.full((SIZE, SIZE), 0.125, dtype=np.float32)
    config = VisibilityConfig(neighbourhood=0, min_confidence=0.125)

    assert vote_one(
        (0.0, 0.0, 2.0), make_frame(plane(), confidence), config
    ) == (Vote.CONSISTENT)
    # tau = 0.25 at confidence 1 with these settings; offset 0.25 is inside.
    config = VisibilityConfig(
        neighbourhood=0, abs_tolerance=0.25, rel_tolerance=0.0
    )
    full = np.ones((SIZE, SIZE), dtype=np.float32)
    for confidence_map in (None, full):
        frame = make_frame(plane(), confidence_map)
        assert vote_one((0.0, 0.0, 2.25), frame, config) == Vote.CONSISTENT
        assert vote_one((0.0, 0.0, 2.5), frame, config) == Vote.OCCLUDED


def test_tolerance_widens_with_low_confidence_exactly() -> None:
    config = VisibilityConfig(
        neighbourhood=0,
        abs_tolerance=0.25,
        rel_tolerance=0.0,
        confidence_slack=1.0,
        min_confidence=0.0,
    )
    confidence = np.full((SIZE, SIZE), 0.5, dtype=np.float32)
    frame = make_frame(plane(), confidence)
    # tau = 0.25 * (1 + 1 * 0.5) = 0.375

    assert vote_one((0.0, 0.0, 2.375), frame, config) == Vote.CONSISTENT
    assert vote_one((0.0, 0.0, 2.5), frame, config) == Vote.OCCLUDED
    assert vote_one((0.0, 0.0, 1.625), frame, config) == Vote.CONSISTENT
    assert vote_one((0.0, 0.0, 1.5), frame, config) == Vote.FREE


def test_tolerance_grows_with_depth_and_slack_scales_the_widening() -> None:
    config = VisibilityConfig(
        neighbourhood=0,
        abs_tolerance=0.125,
        rel_tolerance=0.0625,
        confidence_slack=2.0,
        min_confidence=0.0,
    )
    confidence = np.full((SIZE, SIZE), 0.75, dtype=np.float32)
    frame = make_frame(plane(), confidence)
    # tau = (0.125 + 0.0625 * z) * (1 + 2 * 0.25) at the point: 0.41 at
    # z = 2.375, 0.42 at z = 2.5.

    assert vote_one((0.0, 0.0, 2.375), frame, config) == Vote.CONSISTENT
    assert vote_one((0.0, 0.0, 2.5), frame, config) == Vote.OCCLUDED


@pytest.mark.parametrize(
    ("layer", "row", "col", "value"),
    [
        ("depth", 3, 4, 1.0),
        ("depth", 5, 5, 0.0),
        ("depth", 5, 5, np.nan),
        ("confidence", 3, 3, 0.0),
    ],
    ids=["near pixel", "zero depth", "nan depth", "low confidence"],
)
def test_neighbourhood_blocks_free_votes_next_to_a_bad_pixel(
    layer: str, row: int, col: int, value: float
) -> None:
    maps = {
        "depth": plane(),
        "confidence": np.ones((SIZE, SIZE), dtype=np.float32),
    }
    maps[layer][row, col] = value
    frame = make_frame(maps["depth"], maps["confidence"])

    assert vote_one((0.0, 0.0, 1.5), frame, NEIGHBOURHOOD_0) == Vote.FREE
    assert vote_one((0.0, 0.0, 1.5), frame, DEFAULT) == Vote.UNOBSERVED


def test_the_lowest_window_confidence_sets_the_free_tolerance() -> None:
    config = VisibilityConfig(
        abs_tolerance=0.25, rel_tolerance=0.0, confidence_slack=1.0
    )
    confidence = np.ones((SIZE, SIZE), dtype=np.float32)
    confidence[3, 3] = 0.5
    frame = make_frame(plane(), confidence)
    # Center tau 0.25; the 0.5 pixel in the window widens it to 0.375.
    centered = VisibilityConfig(
        abs_tolerance=0.25, rel_tolerance=0.0, neighbourhood=0
    )

    assert vote_one((0.0, 0.0, 1.7), frame, centered) == Vote.FREE
    assert vote_one((0.0, 0.0, 1.7), frame, config) == Vote.UNOBSERVED
    assert vote_one((0.0, 0.0, 1.6), frame, config) == Vote.FREE


def test_neighbourhood_only_covers_its_radius() -> None:
    depth = plane()
    depth[4, 6] = 1.0
    frame = make_frame(depth)

    assert vote_one((0.0, 0.0, 1.5), frame, DEFAULT) == Vote.FREE
    assert vote_one(
        (0.0, 0.0, 1.5), frame, VisibilityConfig(neighbourhood=2)
    ) == (Vote.UNOBSERVED)


def test_neighbourhood_does_not_change_other_votes() -> None:
    depth = plane()
    depth[3, 4] = 1.0
    frame = make_frame(depth)

    assert vote_one((0.0, 0.0, 3.0), frame, DEFAULT) == Vote.OCCLUDED
    assert vote_one((0.0, 0.0, 2.0), frame, DEFAULT) == Vote.CONSISTENT


def test_window_pixels_outside_the_image_do_not_block_a_free_vote() -> None:
    frame = make_frame(plane())
    # Pixel (0, 0) has part of its 3x3 window outside the image.
    corner = (-4.0 * 1.0 / FOCAL, -4.0 * 1.0 / FOCAL, 1.0)

    assert vote_one(corner, frame, DEFAULT) == Vote.FREE


def test_pose_moves_the_point_into_the_camera() -> None:
    pose = SE3(np.eye(3), np.array([0.0, 0.0, 1.0]))
    frame = make_frame(plane(), pose=pose)

    assert vote_one((0.0, 0.0, 3.0), frame) == Vote.CONSISTENT
    assert vote_one((0.0, 0.0, 2.0), frame) == Vote.FREE


def test_empty_points_return_empty_arrays() -> None:
    evidence = vote_points(np.zeros((0, 3)), [make_frame(plane())])

    assert isinstance(evidence, PointEvidence)
    assert evidence.counts.shape == (0, 4)
    assert evidence.counts.dtype == np.int32
    assert evidence.states.shape == (0,)


def test_vote_points_validates_its_inputs() -> None:
    frame = make_frame(plane())

    with pytest.raises(ValueError, match="at least one frame"):
        vote_points([[0.0, 0.0, 1.0]], [])
    with pytest.raises(ValueError, match="depth"):
        vote_points([[0.0, 0.0, 1.0]], [frame, make_frame(None)])
    with pytest.raises(ValueError, match="finite"):
        vote_points([[0.0, np.nan, 1.0]], [frame])
    with pytest.raises(ValueError, match="points"):
        vote_points([0.0, 0.0, 1.0], [frame])


def state_of(
    consistent: int,
    free: int,
    config: VisibilityConfig,
    occluded: int = 0,
    unobserved: int = 0,
) -> PointState:
    """Votes a point at z = 2 against frames giving the wanted counts."""
    depths = (
        [PLANE_Z] * consistent
        + [3.0] * free
        + [1.0] * occluded
        + [0.0] * unobserved
    )
    frames = [make_frame(plane(z), index=i) for i, z in enumerate(depths)]
    evidence = vote_points([[0.0, 0.0, PLANE_Z]], frames, config)
    np.testing.assert_array_equal(
        evidence.counts[0], [consistent, free, occluded, unobserved]
    )
    return PointState(int(evidence.states[0]))


def test_changed_needs_min_free_votes_and_more_free_than_consistent() -> None:
    config = VisibilityConfig(neighbourhood=0, min_free_votes=2)

    assert state_of(0, 2, config) == PointState.CHANGED
    assert state_of(0, 1, config) == PointState.UNKNOWN
    assert state_of(1, 2, config) == PointState.CHANGED
    assert state_of(2, 2, config) == PointState.PRESENT
    assert state_of(3, 2, config) == PointState.PRESENT


def test_one_free_vote_without_a_consistent_one_is_changed_by_default() -> None:
    assert state_of(0, 1, DEFAULT) == PointState.CHANGED
    assert state_of(1, 1, DEFAULT) == PointState.PRESENT


def test_min_free_votes_is_a_threshold() -> None:
    config = VisibilityConfig(neighbourhood=0, min_free_votes=3)

    assert state_of(0, 2, config) == PointState.UNKNOWN
    assert state_of(0, 3, config) == PointState.CHANGED
    assert state_of(2, 3, config) == PointState.CHANGED


def test_present_needs_min_consistent_votes() -> None:
    config = VisibilityConfig(
        neighbourhood=0, min_free_votes=2, min_consistent_votes=2
    )

    assert state_of(1, 0, config) == PointState.UNKNOWN
    assert state_of(2, 0, config) == PointState.PRESENT
    assert state_of(1, 1, config) == PointState.UNKNOWN
    assert state_of(2, 1, config) == PointState.PRESENT


def test_occluded_and_unobserved_votes_never_decide() -> None:
    config = VisibilityConfig(neighbourhood=0)

    assert state_of(0, 0, config, occluded=5, unobserved=5) == (
        PointState.UNKNOWN
    )
    assert state_of(1, 0, config, occluded=5, unobserved=5) == (
        PointState.PRESENT
    )
    assert state_of(0, 2, config, occluded=5, unobserved=5) == (
        PointState.CHANGED
    )
    assert state_of(2, 2, config, occluded=9, unobserved=9) == (
        PointState.PRESENT
    )


@pytest.mark.parametrize(
    ("occluded", "unobserved"), [(3, 0), (0, 3), (2, 2), (1, 1)]
)
def test_the_veto_decides_what_a_point_nobody_confirms_becomes(
    occluded: int, unobserved: int
) -> None:
    kept = VisibilityConfig(neighbourhood=0)
    dropped = VisibilityConfig(neighbourhood=0, veto=False)

    assert kept.veto is True
    assert state_of(0, 0, kept, occluded, unobserved) == PointState.UNKNOWN
    assert state_of(0, 0, dropped, occluded, unobserved) == PointState.CHANGED


def test_without_the_veto_a_point_below_the_free_threshold_is_changed() -> None:
    kept = VisibilityConfig(neighbourhood=0, min_free_votes=2)
    dropped = VisibilityConfig(neighbourhood=0, min_free_votes=2, veto=False)

    assert state_of(0, 1, kept) == PointState.UNKNOWN
    assert state_of(0, 1, dropped) == PointState.CHANGED


def test_without_the_veto_a_confirmed_point_is_still_present() -> None:
    kept = VisibilityConfig(neighbourhood=0)
    dropped = replace(kept, veto=False)

    for config in (kept, dropped):
        assert state_of(1, 0, config) == PointState.PRESENT
        assert state_of(1, 0, config, occluded=5, unobserved=5) == (
            PointState.PRESENT
        )
        assert state_of(2, 2, config) == PointState.PRESENT


def test_the_veto_leaves_the_changed_rule_alone() -> None:
    kept = VisibilityConfig(neighbourhood=0, min_consistent_votes=2)
    dropped = replace(kept, veto=False)

    for config in (kept, dropped):
        assert state_of(0, 2, config) == PointState.CHANGED
        assert state_of(1, 2, config) == PointState.CHANGED
        assert state_of(2, 2, config) == PointState.PRESENT
    assert state_of(1, 0, kept) == PointState.UNKNOWN
    assert state_of(1, 0, dropped) == PointState.CHANGED


def test_evidence_properties_return_the_count_columns() -> None:
    frames = [make_frame(plane(3.0)), make_frame(plane(1.0))]
    evidence = vote_points([[0.0, 0.0, PLANE_Z]], frames, NEIGHBOURHOOD_0)

    np.testing.assert_array_equal(evidence.consistent, [0])
    np.testing.assert_array_equal(evidence.free, [1])
    np.testing.assert_array_equal(evidence.occluded, [1])
    np.testing.assert_array_equal(evidence.unobserved, [0])
    assert evidence.states.dtype == np.int8


def test_total_sums_the_votes_of_the_selected_points() -> None:
    big = np.iinfo(np.int32).max
    evidence = PointEvidence(
        np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [big, 0, 0, big], [9, 10, 11, 12]],
            dtype=np.int32,
        ),
        np.zeros(4, dtype=np.int8),
    )

    assert evidence.total(np.array([0, 3])) == EvidenceCounts(
        consistent=10, free=12, occluded=14, unobserved=16
    )
    assert evidence.total(np.array([2, 2])) == EvidenceCounts(
        consistent=2 * big, free=0, occluded=0, unobserved=2 * big
    )


def test_total_of_no_points_is_all_zeros() -> None:
    evidence = PointEvidence(
        np.ones((3, 4), dtype=np.int32), np.zeros(3, dtype=np.int8)
    )

    assert evidence.total(np.empty(0, dtype=np.int64)) == EvidenceCounts(
        consistent=0, free=0, occluded=0, unobserved=0
    )


CAMERA = Intrinsics(60.0, 58.0, 31.5, 23.5, 64, 48)


def random_frames(rng: np.random.Generator, count: int) -> list[Frame]:
    frames = []
    for index in range(count):
        pose = perturbation(
            rng.normal(size=3),
            rng.uniform(3.0, 20.0),
            rng.normal(scale=0.2, size=3),
        )
        depth = rng.uniform(1.0, 6.0, size=(48, 64)).astype(np.float32)
        depth[rng.random((48, 64)) < 0.05] = 0.0
        depth[rng.random((48, 64)) < 0.02] = np.nan
        confidence = rng.uniform(0.0, 1.0, size=(48, 64)).astype(np.float32)
        frames.append(
            Frame(
                epoch=Epoch.AFTER,
                index=index,
                timestamp_s=0.0,
                intrinsics=CAMERA,
                world_from_camera=pose,
                depth=depth,
                confidence=confidence if index % 2 else None,
            )
        )
    return frames


def random_cloud(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    cloud = rng.normal(size=(count, 3)) * [1.5, 1.2, 1.0]
    cloud[:, 2] += 3.5
    return cloud


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=15, deadline=None)
def test_counts_do_not_depend_on_the_batch_size(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cloud = random_cloud(rng, 300)
    frames = random_frames(rng, 3)

    reference = vote_points(cloud, frames)
    for batch_size in (1, 7):
        batched = vote_points(
            cloud, frames, VisibilityConfig(batch_size=batch_size)
        )
        np.testing.assert_array_equal(batched.counts, reference.counts)
        np.testing.assert_array_equal(batched.states, reference.states)


AMBIGUITY_MARGIN = 2e-3


def reference_votes(
    cloud: NDArray[np.float64],
    frame: Frame,
    config: VisibilityConfig,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Center-pixel votes in float64 from ``project`` and ``in_image``.

    Free votes are not modelled: a farther surface is left UNOBSERVED.

    Returns:
        The votes, and a mask of points whose votes are far from any
        rounding, image-bound, or tolerance boundary, where float32 and
        float64 must agree.
    """
    assert frame.depth is not None
    intrinsics = frame.intrinsics
    uv, z = project(cloud, intrinsics, frame.world_from_camera)
    inside = in_image(uv, z, intrinsics)
    safe = np.where(inside[:, None], uv, 0.0)
    cols = np.rint(safe[:, 0]).astype(int)
    rows = np.rint(safe[:, 1]).astype(int)
    d = frame.depth[rows, cols].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    conf = np.ones(len(cloud))
    if frame.confidence is not None:
        conf = frame.confidence[rows, cols].astype(np.float64)
        valid &= conf >= config.min_confidence
    tau = (config.abs_tolerance + config.rel_tolerance * z) * (
        1 + config.confidence_slack * (1 - conf)
    )
    observed = inside & valid
    votes = np.full(len(cloud), Vote.UNOBSERVED, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        votes[observed & (d < z - tau)] = Vote.OCCLUDED
        votes[observed & (np.abs(z - d) <= tau)] = Vote.CONSISTENT
        near_tau = observed & (np.abs(np.abs(z - d) - tau) < AMBIGUITY_MARGIN)
    to_round = np.abs(uv - np.floor(uv) - 0.5).min(axis=1)
    to_bound = np.minimum(
        np.abs(uv + 0.5).min(axis=1),
        np.abs(uv - [intrinsics.width - 0.5, intrinsics.height - 0.5]).min(
            axis=1
        ),
    )
    near_pixel = np.minimum(to_round, to_bound) < AMBIGUITY_MARGIN
    near_camera = np.abs(z) < AMBIGUITY_MARGIN
    return votes, ~(near_tau | near_pixel | near_camera)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_center_decisions_match_a_numpy_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cloud = random_cloud(rng, 4000)
    frames = random_frames(rng, 4)

    evidence = vote_points(cloud, frames, NEIGHBOURHOOD_0)

    expected = np.zeros((len(cloud), 4), dtype=np.int32)
    trusted = np.ones(len(cloud), dtype=bool)
    for frame in frames:
        votes, unambiguous = reference_votes(cloud, frame, NEIGHBOURHOOD_0)
        trusted &= unambiguous
        expected[np.arange(len(cloud)), votes] += 1
    got = evidence.counts[trusted]
    assert trusted.sum() > 1000
    assert got[:, Vote.CONSISTENT].sum() > 0
    assert got[:, Vote.OCCLUDED].sum() > 0
    assert got[:, Vote.FREE].sum() > 0
    got[:, Vote.UNOBSERVED] += got[:, Vote.FREE]
    got[:, Vote.FREE] = 0
    np.testing.assert_array_equal(got, expected[trusted])


def test_counts_are_translation_invariant() -> None:
    rng = np.random.default_rng(3)
    cloud = random_cloud(rng, 500)
    frames = random_frames(rng, 3)
    shift = np.array([1e4, -1e4, 1e4])
    translation = Sim3(1.0, np.eye(3), shift)
    moved = [frame.transformed(translation) for frame in frames]

    base = vote_points(cloud, frames)
    shifted = vote_points(cloud + shift, moved)

    np.testing.assert_array_equal(shifted.counts, base.counts)
    np.testing.assert_array_equal(shifted.states, base.states)


def reconstruction(
    epoch: Epoch, frames: list[Frame], points: list[list[float]]
) -> EpochReconstruction:
    cloud = np.array(points, dtype=np.float64).reshape(-1, 3)
    return EpochReconstruction(
        epoch=epoch,
        frames=[replace(frame, epoch=epoch) for frame in frames],
        points=cloud,
        colors=np.zeros((len(cloud), 3), dtype=np.uint8),
        confidence=np.ones(len(cloud), dtype=np.float32),
    )


def test_vote_epochs_votes_each_epoch_with_the_other_epochs_frames() -> None:
    # BEFORE sees the wall at z = 2; AFTER sees it at z = 3.
    frames_before = [make_frame(plane(2.0), index=i) for i in range(2)]
    frames_after = [make_frame(plane(3.0), index=i) for i in range(2)]
    before = reconstruction(Epoch.BEFORE, frames_before, [[0.0, 0.0, 2.0]])
    after = reconstruction(Epoch.AFTER, frames_after, [[0.0, 0.0, 3.0]])

    cross = vote_epochs(before, after, NEIGHBOURHOOD_0)

    # The BEFORE surface at z = 2 is in front of the AFTER wall: free in both
    # AFTER frames, so it vanished.
    np.testing.assert_array_equal(cross.before.counts, [[0, 2, 0, 0]])
    assert cross.before.states[0] == PointState.CHANGED
    # The AFTER surface at z = 3 is behind the BEFORE wall: occluded.
    np.testing.assert_array_equal(cross.after.counts, [[0, 0, 2, 0]])
    assert cross.after.states[0] == PointState.UNKNOWN


def test_vote_epochs_rejects_swapped_epochs() -> None:
    before = reconstruction(Epoch.BEFORE, [make_frame(plane())], [[0, 0, 2]])
    after = reconstruction(Epoch.AFTER, [make_frame(plane())], [[0, 0, 2]])

    with pytest.raises(ValueError, match="before"):
        vote_epochs(after, after)
    with pytest.raises(ValueError, match="after"):
        vote_epochs(before, before)
