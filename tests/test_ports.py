from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.domain import SE3, Intrinsics
from spatial_change.ports import (
    ChangeCrops,
    Describer,
    FeatureEncoder,
    Labeler,
    MaskSegmenter,
    ReconstructionResult,
    Reconstructor,
    Unloadable,
    VisionLanguageModel,
    check_vocabulary,
)
from spatial_change.semantics.relations import ChangeFact

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
    def loaded_model_id(self) -> str | None:
        return None

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


@pytest.mark.parametrize(
    "vocabulary", [[], ["red", "red"], ["red", ""], ["red", 3]]
)
def test_check_vocabulary_rejects_an_invalid_vocabulary(
    vocabulary: list[object],
) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        check_vocabulary(vocabulary)  # type: ignore[arg-type]


def test_check_vocabulary_accepts_unique_names() -> None:
    check_vocabulary(["red", "green"])


class FakeDescriber:
    def describe(self, fact: ChangeFact, crops: ChangeCrops) -> str:
        return fact.label

    def unload(self) -> None:
        pass


class FakeVisionLanguageModel:
    def generate(self, images: Sequence[NDArray[np.uint8]], prompt: str) -> str:
        return prompt

    def unload(self) -> None:
        pass


@pytest.mark.parametrize(
    ("fake", "port"),
    [
        (FakeDescriber(), Describer),
        (FakeDescriber(), Unloadable),
        (FakeVisionLanguageModel(), VisionLanguageModel),
        (FakeVisionLanguageModel(), Unloadable),
    ],
)
def test_describer_fakes_satisfy_ports(fake: object, port: type) -> None:
    assert isinstance(fake, port)


def test_a_describer_is_not_a_vision_language_model() -> None:
    assert not isinstance(FakeDescriber(), VisionLanguageModel)


def _crop(height: int = 4, width: int = 5) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), np.uint8)


def test_change_crops_default_to_no_crops() -> None:
    crops = ChangeCrops()

    assert crops.before == ()
    assert crops.after == ()


def test_change_crops_keep_the_crops_in_order() -> None:
    first, second = _crop(2, 3), _crop(6, 7)

    crops = ChangeCrops(before=(first, second), after=(second,))

    assert crops.before[0] is first
    assert crops.before[1] is second
    assert crops.after[0] is second


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 5), np.uint8),
        np.zeros((4, 5, 4), np.uint8),
        np.zeros((4, 5, 3), np.float32),
        np.zeros((0, 5, 3), np.uint8),
        np.zeros((4, 0, 3), np.uint8),
    ],
)
@pytest.mark.parametrize("side", ["before", "after"])
def test_change_crops_reject_invalid_images(
    bad: NDArray[np.uint8], side: str
) -> None:
    with pytest.raises(ValueError, match=side):
        ChangeCrops(**{side: (bad,)})
