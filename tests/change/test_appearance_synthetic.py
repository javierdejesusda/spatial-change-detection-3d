from dataclasses import dataclass

import numpy as np
import pytest
from appearance_scenes import recolored_frames, reconstruction
from numpy.typing import NDArray
from visibility_scenes import (
    SEEDS,
    LabelledCloud,
    changed_objects,
    changed_points_of,
    dominant,
    majority,
    scene_evidence,
)

from spatial_change.change import (
    AppearanceConfig,
    Candidate,
    compare_appearance,
    find_candidates,
    fuse_appearance,
)
from spatial_change.domain import Epoch, EpochReconstruction
from spatial_change.perception import FakeFeatureEncoder
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SyntheticObject,
    SyntheticScenePair,
)
from spatial_change.visibility import (
    CrossEvidence,
    PointEvidence,
    PointState,
    VisibilityConfig,
)

# The fake encoder compares hue and saturation, so a flat gray object
# against a saturated one gives a cosine near 0.6; this threshold sits
# between that and the static scene. Measured over seeds 0 to 2 with a
# patch of 8 pixels on the 128x96 scenes:
#
# * Static surfaces keep a cosine of at least 0.936 (1st percentile 0.991),
#   so no PRESENT point flips (0 of about 78,000 BEFORE and 140,000 AFTER
#   points) and the candidates equal the geometric ones.
# * The recolored shelf has a median cosine of 0.628 on both sides, and
#   0.74 to 0.77 of its BEFORE and 0.94 to 0.96 of its AFTER points flip.
#   Every candidate holding most of its CHANGED points has the shelf as
#   majority label with purity 1.000, holds all of them, and has all of
#   them inside its box, with the box center 0.20 to 0.29 m from the true
#   one (the shelf is only partly seen, so the center is biased).
PATCH = 8
CONFIG = AppearanceConfig(changed_below=0.8)
MAX_STATIC_FLIP_SHARE = 0.001
MIN_PURITY = 0.95
MIN_DOMINANT_SHARE = 0.9
MIN_SOLID_SHARE = 0.85
MIN_MOVED_SHARE = 0.5
MIN_BEFORE_CHANGED_SHARE = 0.6
MIN_AFTER_CHANGED_SHARE = 0.85
MAX_CENTER_ERROR = 0.4


@dataclass(frozen=True, eq=False)
class Run:
    """A scene with its geometric and fused evidence per epoch."""

    pair: SyntheticScenePair
    clouds: dict[Epoch, LabelledCloud]
    reconstructions: dict[Epoch, EpochReconstruction]
    evidence: dict[Epoch, PointEvidence]
    fused: dict[Epoch, PointEvidence]


def run_scene(seed: int, recolor_shelf: bool) -> Run:
    votes = scene_evidence(seed, VisibilityConfig())
    pair = votes.pair
    after_frames = pair.after.frames
    if recolor_shelf:
        shelf = shelf_of(pair)
        after_frames = recolored_frames(
            after_frames, pair.after.instance_maps, shelf.index
        )
    reconstructions = {
        Epoch.BEFORE: reconstruction(
            Epoch.BEFORE, pair.before.frames, votes.clouds[Epoch.BEFORE].points
        ),
        Epoch.AFTER: reconstruction(
            Epoch.AFTER, after_frames, votes.clouds[Epoch.AFTER].points
        ),
    }
    appearance = compare_appearance(
        reconstructions[Epoch.BEFORE],
        reconstructions[Epoch.AFTER],
        FakeFeatureEncoder(PATCH),
        CONFIG,
    )
    fused = fuse_appearance(
        CrossEvidence(
            votes.evidence[Epoch.BEFORE], votes.evidence[Epoch.AFTER]
        ),
        appearance,
        CONFIG,
    )
    return Run(
        pair,
        votes.clouds,
        reconstructions,
        votes.evidence,
        {Epoch.BEFORE: fused.before, Epoch.AFTER: fused.after},
    )


def shelf_of(pair: SyntheticScenePair) -> SyntheticObject:
    """Returns the unchanged object seen in both epochs: the large shelf."""
    return next(o for o in pair.objects if o.label == "shelf")


@pytest.fixture(scope="module", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def plain(request: pytest.FixtureRequest) -> Run:
    return run_scene(request.param, recolor_shelf=False)


@pytest.fixture(scope="module", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def recolored(request: pytest.FixtureRequest) -> Run:
    return run_scene(request.param, recolor_shelf=True)


def candidates_of(run: Run, fused: bool) -> dict[Epoch, tuple[Candidate, ...]]:
    evidence = run.fused if fused else run.evidence
    found = find_candidates(
        run.reconstructions[Epoch.BEFORE],
        run.reconstructions[Epoch.AFTER],
        CrossEvidence(evidence[Epoch.BEFORE], evidence[Epoch.AFTER]),
    )
    return {Epoch.BEFORE: found.disappeared, Epoch.AFTER: found.appeared}


def side(run: Run, epoch: Epoch, fused: bool) -> NDArray[np.int8]:
    return (run.fused if fused else run.evidence)[epoch].states


def dominant_of(
    run: Run, epoch: Epoch, obj: SyntheticObject, fused: bool
) -> tuple[Candidate, float]:
    return dominant(
        run.clouds[epoch],
        side(run, epoch, fused),
        obj,
        candidates_of(run, fused)[epoch],
    )


@pytest.mark.parametrize("epoch", list(Epoch))
def test_unchanged_appearance_flips_almost_no_static_points(
    plain: Run, epoch: Epoch
) -> None:
    before = side(plain, epoch, fused=False)
    after = side(plain, epoch, fused=True)
    present = before == PointState.PRESENT

    flipped = int((present & (after == PointState.CHANGED)).sum())

    assert flipped / int(present.sum()) <= MAX_STATIC_FLIP_SHARE
    np.testing.assert_array_equal(after[~present], before[~present])


@pytest.mark.parametrize("epoch", list(Epoch))
def test_unchanged_appearance_adds_no_false_candidate(
    plain: Run, epoch: Epoch
) -> None:
    geometric = candidates_of(plain, fused=False)[epoch]
    fused = candidates_of(plain, fused=True)[epoch]
    changed = {o.index for o in changed_objects(plain.pair, epoch)}

    assert len(fused) == len(geometric)
    for candidate in fused:
        label, purity = majority(plain.clouds[epoch], candidate.point_indices)
        assert purity >= MIN_PURITY, candidate.candidate_id
        assert label in changed, candidate.candidate_id
    for obj in changed_objects(plain.pair, epoch):
        _, share = dominant_of(plain, epoch, obj, fused=True)
        floor = (
            MIN_MOVED_SHARE
            if obj.status == ObjectStatus.MOVED
            else MIN_SOLID_SHARE
        )
        assert share >= floor, obj.object_id


@pytest.mark.parametrize("epoch", list(Epoch))
def test_geometry_alone_does_not_see_the_recolored_shelf(
    recolored: Run, epoch: Epoch
) -> None:
    shelf = shelf_of(recolored.pair)
    changed = side(recolored, epoch, fused=False) == PointState.CHANGED

    for candidate in candidates_of(recolored, fused=False)[epoch]:
        label, _ = majority(recolored.clouds[epoch], candidate.point_indices)
        assert label != shelf.index, candidate.candidate_id
    shelf_points = recolored.clouds[epoch].labels == shelf.index
    assert changed[shelf_points].mean() < 0.01


@pytest.mark.parametrize(
    ("epoch", "min_share"),
    [
        (Epoch.BEFORE, MIN_BEFORE_CHANGED_SHARE),
        (Epoch.AFTER, MIN_AFTER_CHANGED_SHARE),
    ],
)
def test_the_recolored_shelf_turns_changed(
    recolored: Run, epoch: Epoch, min_share: float
) -> None:
    shelf = shelf_of(recolored.pair)
    shelf_points = recolored.clouds[epoch].labels == shelf.index

    share = (
        side(recolored, epoch, fused=True)[shelf_points] == PointState.CHANGED
    ).mean()

    assert share >= min_share


@pytest.mark.parametrize("epoch", list(Epoch))
def test_the_recolored_shelf_has_a_dominant_candidate_on_it(
    recolored: Run, epoch: Epoch
) -> None:
    shelf = shelf_of(recolored.pair)
    cloud = recolored.clouds[epoch]

    candidate, share = dominant_of(recolored, epoch, shelf, fused=True)

    label, purity = majority(cloud, candidate.point_indices)
    mine = cloud.points[
        changed_points_of(cloud, side(recolored, epoch, True), shelf)
    ]
    error = float(
        np.linalg.norm(candidate.box.center - shelf.box(epoch).center)
    )
    assert candidate.epoch == epoch
    assert label == shelf.index
    assert purity >= MIN_PURITY
    assert share >= MIN_DOMINANT_SHARE
    assert candidate.box.contains(mine).mean() >= MIN_DOMINANT_SHARE
    assert error <= MAX_CENTER_ERROR
