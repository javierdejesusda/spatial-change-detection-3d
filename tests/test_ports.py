from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.domain import SE3, Intrinsics
from spatial_change.ports import (
    FeatureEncoder,
    Labeler,
    MaskSegmenter,
    ReconstructionResult,
    Reconstructor,
    Unloadable,
)

_INTRINSICS = Intrinsics(fx=1.0, fy=1.0, cx=1.5, cy=1.0, width=4, height=3)


def _result(n: int) -> ReconstructionResult:
    return ReconstructionResult(
        intrinsics=[_INTRINSICS] * n,
        world_from_camera=[SE3.identity()] * n,
        depth=[np.ones((3, 4), np.float32)] * n,
        confidence=[np.ones((3, 4), np.float32)] * n,
    )


class FakeReconstructor:
    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        return _result(len(images))

    def unload(self) -> None:
        pass


class FakeMaskSegmenter:
    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        return np.zeros((len(boxes_xyxy), *image.shape[:2]), bool)

    def unload(self) -> None:
        pass


class FakeFeatureEncoder:
    @property
    def patch_size(self) -> int:
        return 14

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        h, w = image.shape[0] // 14, image.shape[1] // 14
        return np.zeros((h, w, 8), np.float32)

    def unload(self) -> None:
        pass


class FakeLabeler:
    def label(
        self,
        crops: Sequence[NDArray[np.uint8]],
        vocabulary: Sequence[str],
    ) -> list[tuple[str, float]]:
        return [(vocabulary[0], 1.0) for _ in crops]

    def unload(self) -> None:
        pass


@pytest.mark.parametrize(
    ("fake", "port"),
    [
        (FakeReconstructor(), Reconstructor),
        (FakeMaskSegmenter(), MaskSegmenter),
        (FakeFeatureEncoder(), FeatureEncoder),
        (FakeLabeler(), Labeler),
        (FakeLabeler(), Unloadable),
    ],
)
def test_fakes_satisfy_ports(fake: object, port: type) -> None:
    assert isinstance(fake, port)


def test_object_without_unload_is_not_a_port() -> None:
    class NoUnload:
        def label(
            self,
            crops: Sequence[NDArray[np.uint8]],
            vocabulary: Sequence[str],
        ) -> list[tuple[str, float]]:
            return []

    assert not isinstance(NoUnload(), Labeler)
    assert not isinstance(NoUnload(), Unloadable)


def test_reconstruction_result_accepts_equal_lengths() -> None:
    result = FakeReconstructor().reconstruct([np.zeros((3, 4, 3), np.uint8)])

    assert len(result.depth) == 1


def test_reconstruction_result_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        ReconstructionResult(
            intrinsics=[_INTRINSICS],
            world_from_camera=[SE3.identity()] * 2,
            depth=[np.ones((3, 4), np.float32)],
            confidence=[np.ones((3, 4), np.float32)],
        )
