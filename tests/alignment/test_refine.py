import numpy as np
import pytest
from numpy.typing import NDArray
from rooms import Box, sample_room, sample_room_labelled, static_point_count
from rotations import perturbation, pose_error

from spatial_change.alignment import refine as refine_module
from spatial_change.alignment.refine import (
    RefineConfig,
    Refinement,
    refine_alignment,
)
from spatial_change.domain import SE3
from spatial_change.geometry import IcpConfig, IcpResult, TargetIndex

DENSITY = 400.0
CHANGE_SHARE = 0.2
MOVED_BEFORE = Box(2.0, 1.2, (0.6, 0.6, 0.5))
MOVED_AFTER = Box(4.0, 2.0, (0.6, 0.6, 0.5))
REMOVED = Box(4.5, 1.0, (0.5, 0.5, 0.9))
ADDED = Box(3.0, 3.6, (0.7, 0.5, 0.6))
# Of the after points, the moved and the added box each hold half the
# changed share.
BOX_POINTS = round(
    static_point_count(DENSITY) * CHANGE_SHARE / 2 / (1 - CHANGE_SHARE)
)

ROTATION_TOL_DEG = 0.3
TRANSLATION_TOL = 0.01
EXACT_TRANSLATION_TOL = 1e-6
CHANGE_SHARE_TOL = 0.01
MIN_AFTER_POINTS = 5000
MIN_RECALL = 0.9
MIN_KEPT = 0.9
# Excluding the changed points must cut the plain error to a quarter.
EXCLUDE_GAIN = 0.25
PLAIN = RefineConfig(icp=IcpConfig(trim_fraction=0.0), max_rounds=1)
LOOPED = RefineConfig(icp=IcpConfig(trim_fraction=0.0))
SMALL_DENSITY = 20.0

Epochs = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], SE3]


def _perturbation() -> SE3:
    # Tilted, not axis-aligned; 8 cm of translation.
    return perturbation((1.0, 2.0, 2.0), 4.0, (0.05, -0.04, 0.045))


def _epochs() -> Epochs:
    before = sample_room(
        np.random.default_rng(10), [MOVED_BEFORE, REMOVED], BOX_POINTS, DENSITY
    )
    after, changed = sample_room_labelled(
        np.random.default_rng(11), [MOVED_AFTER, ADDED], BOX_POINTS, DENSITY
    )
    shift = _perturbation()
    return before, shift.apply(after), changed, shift.inverse()


def _meets_tolerance(refined: Refinement, truth: SE3) -> bool:
    rot_deg, trans = pose_error(refined.transform, truth)
    return rot_deg < ROTATION_TOL_DEG and trans < TRANSLATION_TOL


@pytest.fixture(scope="module")
def epochs() -> Epochs:
    return _epochs()


@pytest.fixture(scope="module")
def aware(epochs: Epochs) -> Refinement:
    before, after, _, _ = epochs
    return refine_alignment(after, before)


@pytest.fixture(scope="module")
def aware_one_round(epochs: Epochs) -> Refinement:
    before, after, _, _ = epochs
    return refine_alignment(after, before, config=RefineConfig(max_rounds=1))


@pytest.fixture(scope="module")
def aware_excluding(epochs: Epochs) -> Refinement:
    before, after, changed, _ = epochs
    return refine_alignment(after, before, exclude=changed)


@pytest.fixture(scope="module")
def plain(epochs: Epochs) -> Refinement:
    before, after, _, _ = epochs
    return refine_alignment(after, before, config=PLAIN)


@pytest.fixture(scope="module")
def plain_looped(epochs: Epochs) -> Refinement:
    before, after, _, _ = epochs
    return refine_alignment(after, before, config=LOOPED)


@pytest.fixture(scope="module")
def plain_excluding(epochs: Epochs) -> Refinement:
    before, after, changed, _ = epochs
    return refine_alignment(after, before, config=PLAIN, exclude=changed)


def test_changed_share_of_the_after_cloud_is_twenty_percent(
    epochs: Epochs,
) -> None:
    _, after, changed, _ = epochs

    assert changed.mean() == pytest.approx(CHANGE_SHARE, abs=CHANGE_SHARE_TOL)
    assert len(after) > MIN_AFTER_POINTS


def test_recovers_the_alignment_with_twenty_percent_changed_points(
    epochs: Epochs, aware: Refinement
) -> None:
    _, _, changed, truth = epochs

    assert _meets_tolerance(aware, truth)
    recall = (~aware.static_mask[changed]).mean()
    kept = aware.static_mask[~changed].mean()
    assert recall > MIN_RECALL
    assert kept > MIN_KEPT
    assert aware.rounds >= 2
    assert aware.icp.converged


def test_a_plain_run_misses_the_tolerance_on_the_same_data(
    epochs: Epochs, plain: Refinement, aware: Refinement
) -> None:
    truth = epochs[3]

    assert not _meets_tolerance(plain, truth)
    rot_deg, trans = pose_error(plain.transform, truth)
    aware_rot, aware_trans = pose_error(aware.transform, truth)
    assert aware_rot < rot_deg
    assert aware_trans < trans


def test_the_change_loop_alone_meets_the_tolerance(
    epochs: Epochs, plain: Refinement, plain_looped: Refinement
) -> None:
    truth = epochs[3]

    assert not _meets_tolerance(plain, truth)
    assert _meets_tolerance(plain_looped, truth)
    assert plain_looped.rounds > 1


def test_exclude_alone_meets_the_tolerance(
    epochs: Epochs, plain: Refinement, plain_excluding: Refinement
) -> None:
    truth = epochs[3]

    assert _meets_tolerance(plain_excluding, truth)
    plain_rot, plain_trans = pose_error(plain.transform, truth)
    rot, trans = pose_error(plain_excluding.transform, truth)
    assert rot < EXCLUDE_GAIN * plain_rot
    assert trans < EXCLUDE_GAIN * plain_trans


def test_excluded_points_never_become_static(
    epochs: Epochs, aware_excluding: Refinement
) -> None:
    changed = epochs[2]

    assert not aware_excluding.static_mask[changed].any()
    assert aware_excluding.static_mask[~changed].mean() > MIN_KEPT


def test_the_full_target_index_is_built_once(
    epochs: Epochs, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after, _, _ = epochs
    builds: list[int] = []
    original = TargetIndex.__init__

    def counting(self: TargetIndex, target: NDArray[np.float64]) -> None:
        builds.append(len(target))
        original(self, target)

    monkeypatch.setattr(TargetIndex, "__init__", counting)

    refined = refine_alignment(after, before)

    assert refined.rounds >= 2
    assert builds.count(len(before)) == 1


def test_identical_clouds_stop_after_one_round() -> None:
    cloud = sample_room(np.random.default_rng(12), density=DENSITY)
    truth = _perturbation()

    refined = refine_alignment(truth.apply_inverse(cloud), cloud)

    assert refined.rounds == 1
    assert refined.static_mask.all()
    assert pose_error(refined.transform, truth)[1] < EXACT_TRANSLATION_TOL


def test_round_cap_is_respected(aware_one_round: Refinement) -> None:
    assert aware_one_round.rounds == 1


def _spy_on_icp(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, SE3, IcpResult]]:
    calls: list[tuple[int, SE3, IcpResult]] = []
    original = refine_module.trimmed_icp

    def spy(
        source: NDArray[np.float64],
        target: TargetIndex,
        init: SE3,
        config: IcpConfig,
    ) -> IcpResult:
        result = original(source, target, init, config)
        calls.append((len(source), init, result))
        return result

    monkeypatch.setattr(refine_module, "trimmed_icp", spy)
    return calls


def test_a_coarse_pass_seeds_the_first_full_round(
    epochs: Epochs, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after, _, _ = epochs
    calls = _spy_on_icp(monkeypatch)
    stride = RefineConfig().coarse_stride

    refine_alignment(after, before)

    coarse, first_full = calls[0], calls[1]
    assert coarse[0] == len(after[::stride])
    assert first_full[0] > coarse[0]
    assert first_full[1] is coarse[2].transform


def test_coarse_stride_one_disables_the_coarse_pass(
    epochs: Epochs, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after, _, _ = epochs
    calls = _spy_on_icp(monkeypatch)

    refined = refine_alignment(
        after, before, config=RefineConfig(coarse_stride=1)
    )

    assert len(calls) == refined.rounds
    assert calls[0][0] == len(after)


def test_small_inputs_skip_the_coarse_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = sample_room(np.random.default_rng(13), density=SMALL_DENSITY)
    truth = _perturbation()
    calls = _spy_on_icp(monkeypatch)

    refined = refine_alignment(truth.apply_inverse(cloud), cloud)

    assert len(cloud) // RefineConfig().coarse_stride < 1000
    assert calls[0][0] == len(cloud)
    assert refined.rounds == len(calls)


def test_invalid_inputs_raise(epochs: Epochs) -> None:
    before, after, _, _ = epochs

    with pytest.raises(ValueError, match="exclude"):
        refine_alignment(after, before, exclude=np.zeros(5, dtype=np.bool_))
    with pytest.raises(ValueError, match="exclude"):
        refine_alignment(after, before, exclude=np.zeros(len(after)))
    with pytest.raises(ValueError, match="at least 3"):
        refine_alignment(
            after, before, exclude=np.ones(len(after), dtype=np.bool_)
        )


def test_a_collinear_static_set_raises() -> None:
    line = np.zeros((50, 3))
    line[:, 0] = np.linspace(0.0, 3.0, 50)

    with pytest.raises(ValueError, match="collinear"):
        refine_alignment(line, line)


def test_clouds_too_far_apart_for_the_gate_raise(epochs: Epochs) -> None:
    before, after, _, _ = epochs

    with pytest.raises(ValueError, match="usable pairs"):
        refine_alignment(after + 1000.0, before)


@pytest.mark.parametrize("shape", [(30,), (30, 2)])
def test_malformed_clouds_raise_value_error(
    epochs: Epochs, shape: tuple[int, ...]
) -> None:
    before, after, _, _ = epochs
    bad = np.zeros(shape)

    with pytest.raises(ValueError, match="target"):
        refine_alignment(after, bad)
    with pytest.raises(ValueError, match="source"):
        refine_alignment(bad, before)


@pytest.mark.parametrize(
    "kwargs",
    [{"change_threshold": 0.0}, {"max_rounds": 0}, {"coarse_stride": 0}],
)
def test_refine_config_validates(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RefineConfig(**kwargs)  # type: ignore[arg-type]
