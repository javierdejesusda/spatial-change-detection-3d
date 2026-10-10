import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from rotations import random_rotation

from spatial_change.domain import Sim3
from spatial_change.geometry import umeyama


def _random_sim3(rng: np.random.Generator) -> Sim3:
    return Sim3(
        float(rng.uniform(0.1, 10.0)),
        random_rotation(rng),
        rng.uniform(-10.0, 10.0, 3),
    )


def _rotation_error_deg(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@given(st.integers(0, 2**32 - 1), st.integers(3, 50))
def test_recovers_exact_sim3(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    truth = _random_sim3(rng)
    src = rng.normal(size=(n, 3))

    estimate = umeyama(src, truth.apply(src))

    assert estimate.scale == pytest.approx(truth.scale, rel=1e-6)
    np.testing.assert_allclose(estimate.rotation, truth.rotation, atol=1e-6)
    np.testing.assert_allclose(
        estimate.translation, truth.translation, atol=1e-6
    )


@given(st.integers(0, 2**32 - 1))
def test_recovers_rigid_transform_without_scale(seed: int) -> None:
    rng = np.random.default_rng(seed)
    truth = _random_sim3(rng)
    rigid = Sim3(1.0, truth.rotation, truth.translation)
    src = rng.normal(size=(20, 3))

    estimate = umeyama(src, rigid.apply(src), with_scale=False)

    assert estimate.scale == 1.0
    np.testing.assert_allclose(estimate.rotation, rigid.rotation, atol=1e-6)
    np.testing.assert_allclose(
        estimate.translation, rigid.translation, atol=1e-6
    )


def test_without_scale_returns_unit_scale_for_scaled_data() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))

    estimate = umeyama(src, 3.0 * src, with_scale=False)

    assert estimate.scale == 1.0
    np.testing.assert_allclose(estimate.rotation, np.eye(3), atol=1e-9)


def test_recovers_sim3_under_noise() -> None:
    rng = np.random.default_rng(7)
    truth = _random_sim3(rng)
    src = rng.normal(size=(500, 3))
    dst = truth.apply(src) + rng.normal(scale=1e-3, size=(500, 3))

    estimate = umeyama(src, dst)

    assert _rotation_error_deg(estimate.rotation, truth.rotation) < 0.5
    assert abs(estimate.scale / truth.scale - 1.0) < 0.01


def test_handles_reflection_in_svd() -> None:
    # Mirrored correspondences: the best proper rotation must still have
    # determinant +1 instead of returning the reflection.
    rng = np.random.default_rng(3)
    src = rng.normal(size=(40, 3))
    dst = src * np.array([1.0, 1.0, -1.0])

    estimate = umeyama(src, dst)

    # The scale must still be the least-squares optimum for the returned
    # rotation, which requires the sign correction in the scale as well.
    src_c, dst_c = src - src.mean(axis=0), dst - dst.mean(axis=0)
    optimal = np.sum(dst_c * (src_c @ estimate.rotation.T)) / np.sum(src_c**2)
    assert np.linalg.det(estimate.rotation) == pytest.approx(1.0)
    assert estimate.scale == pytest.approx(optimal, rel=1e-9)
    np.testing.assert_allclose(
        estimate.apply(src.mean(axis=0)[None])[0], dst.mean(axis=0), atol=1e-9
    )


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (np.zeros((2, 3)), np.zeros((2, 3))),
        (np.zeros((4, 3)), np.zeros((5, 3))),
        (np.zeros((4, 2)), np.zeros((4, 2))),
        (np.ones((5, 3)), np.arange(15.0).reshape(5, 3)),
        (
            np.outer(np.arange(6.0), [1.0, 2.0, 3.0]),
            np.random.default_rng(0).normal(size=(6, 3)),
        ),
        (
            np.random.default_rng(0).normal(size=(6, 3)),
            np.zeros((6, 3)),
        ),
        (
            np.random.default_rng(0).normal(size=(6, 3)),
            np.full((6, 3), np.nan),
        ),
    ],
    ids=[
        "too-few",
        "length-mismatch",
        "wrong-columns",
        "coincident-src",
        "collinear-src",
        "coincident-dst",
        "non-finite",
    ],
)
def test_degenerate_input_raises(
    src: NDArray[np.float64], dst: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError):
        umeyama(src, dst)
