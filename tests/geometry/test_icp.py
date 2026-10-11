import numpy as np
import pytest
from numpy.typing import NDArray
from rooms import Box, sample_room
from rotations import perturbation, pose_error

from spatial_change.domain import SE3
from spatial_change.geometry import (
    IcpConfig,
    IcpResult,
    TargetIndex,
    trimmed_icp,
)


def _perturbation(rng: np.random.Generator) -> SE3:
    return perturbation(rng.normal(size=3), 4.0, (0.05, -0.04, 0.03))


@pytest.fixture(scope="module")
def room() -> NDArray[np.float64]:
    return sample_room(np.random.default_rng(0), density=60.0)


def test_recovers_known_transform_on_identical_clouds(
    room: NDArray[np.float64],
) -> None:
    truth = _perturbation(np.random.default_rng(1))  # target_from_source
    source = truth.apply_inverse(room)

    result = trimmed_icp(source, room)

    rot_deg, trans = pose_error(result.transform, truth)
    assert rot_deg < 1e-4
    assert trans < 1e-6
    assert result.converged
    assert result.rmse < 1e-6
    assert result.inlier_fraction == pytest.approx(1.0)
    assert 1 < result.iterations <= 50


def test_init_is_honoured(room: NDArray[np.float64]) -> None:
    truth = _perturbation(np.random.default_rng(2))
    source = truth.apply_inverse(room)

    result = trimmed_icp(source, room, init=truth)

    assert result.iterations <= 2
    assert pose_error(result.transform, truth)[1] < 1e-9


def test_recovers_from_independent_samplings() -> None:
    rng = np.random.default_rng(3)
    target = sample_room(rng, density=100.0)
    truth = _perturbation(rng)
    source = truth.apply_inverse(sample_room(rng, density=100.0))

    result = trimmed_icp(source, target)

    rot_deg, trans = pose_error(result.transform, truth)
    assert rot_deg < 0.2
    assert trans < 0.01
    assert result.converged


def test_gate_discards_far_outliers(room: NDArray[np.float64]) -> None:
    truth = _perturbation(np.random.default_rng(4))
    source = truth.apply_inverse(room)
    far = np.random.default_rng(5).uniform(20.0, 30.0, (200, 3))
    config = IcpConfig(trim_fraction=0.0)

    result = trimmed_icp(np.vstack([source, far]), room, config=config)

    assert pose_error(result.transform, truth)[1] < 1e-6
    assert result.inlier_fraction == pytest.approx(
        len(source) / (len(source) + 200)
    )


def test_trimming_rejects_near_outliers(room: NDArray[np.float64]) -> None:
    rng = np.random.default_rng(6)
    truth = _perturbation(rng)
    source = truth.apply_inverse(room)
    bent = source.copy()
    shifted = rng.random(len(bent)) < 0.1
    bent[shifted] += [0.0, 0.0, 0.18]  # inside the gate, biases a plain fit

    trimmed = trimmed_icp(bent, room, config=IcpConfig(trim_fraction=0.2))
    plain = trimmed_icp(bent, room, config=IcpConfig(trim_fraction=0.0))

    assert pose_error(trimmed.transform, truth)[1] < 0.003
    assert (
        pose_error(plain.transform, truth)[1]
        > 2 * pose_error(trimmed.transform, truth)[1]
    )


def test_a_subset_of_the_source_can_be_registered(
    room: NDArray[np.float64],
) -> None:
    rng = np.random.default_rng(7)
    truth = _perturbation(rng)
    source = truth.apply_inverse(room)
    corrupted = rng.random(len(source)) < 0.3
    source[corrupted] += [0.0, 0.0, 0.2]
    config = IcpConfig(trim_fraction=0.0)

    subset = trimmed_icp(source[~corrupted], room, config=config)
    everything = trimmed_icp(source, room, config=config)

    assert pose_error(subset.transform, truth)[1] < 1e-6
    assert pose_error(everything.transform, truth)[1] > 0.01
    assert subset.inlier_fraction == pytest.approx(1.0)


def test_iteration_cap_reports_not_converged(room: NDArray[np.float64]) -> None:
    truth = _perturbation(np.random.default_rng(8))
    source = truth.apply_inverse(room)

    result = trimmed_icp(source, room, config=IcpConfig(max_iterations=1))

    assert not result.converged
    assert result.iterations == 1


def test_error_cases(room: NDArray[np.float64]) -> None:
    some = room[:100]
    with pytest.raises(ValueError, match="shape"):
        trimmed_icp(some[:, :2], room)
    with pytest.raises(ValueError, match="shape"):
        trimmed_icp(some, room[:, :2])
    with pytest.raises(ValueError, match="source.*at least 3"):
        trimmed_icp(some[:2], room)
    with pytest.raises(ValueError, match="target.*at least 3"):
        trimmed_icp(some, room[:2])
    with pytest.raises(ValueError, match="usable pairs"):
        trimmed_icp(some + 100.0, room)
    with pytest.raises(ValueError, match="source.*finite"):
        trimmed_icp(np.full((10, 3), np.nan), room)
    with pytest.raises(ValueError, match="target.*finite"):
        trimmed_icp(some, np.full((10, 3), np.nan))


def test_target_index_validates_its_input(room: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="shape"):
        TargetIndex(room[:, :2])
    with pytest.raises(ValueError, match="at least 3"):
        TargetIndex(room[:2])
    with pytest.raises(ValueError, match="finite"):
        TargetIndex(np.full((10, 3), np.nan))


def test_a_prebuilt_index_gives_the_same_result_without_rebuilding(
    room: NDArray[np.float64], monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = _perturbation(np.random.default_rng(9))
    source = truth.apply_inverse(room)
    index = TargetIndex(room)
    builds: list[int] = []
    original = TargetIndex.__init__

    def counting(self: TargetIndex, target: NDArray[np.float64]) -> None:
        builds.append(len(target))
        original(self, target)

    monkeypatch.setattr(TargetIndex, "__init__", counting)

    reused = trimmed_icp(source, index)
    again = trimmed_icp(source[::2], index)
    fresh = trimmed_icp(source, room)

    assert isinstance(reused, IcpResult)
    assert builds == [len(room)]
    np.testing.assert_array_equal(
        reused.transform.matrix, fresh.transform.matrix
    )
    assert again.converged


def test_a_converged_run_stops_before_the_iteration_cap() -> None:
    rng = np.random.default_rng(1)
    static = Box(2.0, 1.0, (0.5, 0.5, 0.5))
    target = sample_room(rng, [static], 800, density=200.0)
    truth = _perturbation(rng)
    moved = Box(4.0, 2.0, (0.5, 0.5, 0.5))
    source = truth.apply_inverse(sample_room(rng, [moved], 800, density=200.0))

    result = trimmed_icp(source, target)

    assert result.converged
    assert result.iterations < IcpConfig().max_iterations


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"trim_fraction": -0.1},
        {"trim_fraction": 1.0},
        {"max_correspondence_distance": 0.0},
        {"tolerance": 0.0},
    ],
)
def test_icp_config_validates(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        IcpConfig(**kwargs)  # type: ignore[arg-type]
