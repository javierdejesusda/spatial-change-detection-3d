from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from spatial_change.ports import Labeler, ReconstructionResult
from spatial_change.reconstruction.fake import FakeReconstructor


class ShortReconstructor(FakeReconstructor):
    """Drops the last view of every result."""

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        return super().reconstruct(images[:-1])


class ScriptedLabeler:
    """A labeler that records its calls and answers from a script.

    It wraps ``inner`` when given, gives every crop ``fixed`` when given,
    and otherwise replays ``results`` in order.

    Attributes:
        calls: The crops and the vocabulary of every call.
    """

    def __init__(
        self,
        results: Sequence[tuple[str, float]] = (),
        fixed: tuple[str, float] | None = None,
        inner: Labeler | None = None,
    ) -> None:
        self._results = list(results)
        self._fixed = fixed
        self._inner = inner
        self.calls: list[tuple[list[NDArray[np.uint8]], list[str]]] = []

    @property
    def vocabularies(self) -> list[list[str]]:
        """The vocabulary of every call."""
        return [vocabulary for _, vocabulary in self.calls]

    @property
    def crop_counts(self) -> list[int]:
        """The number of crops of every call."""
        return [len(crops) for crops, _ in self.calls]

    def label(
        self, crops: Sequence[NDArray[np.uint8]], vocabulary: Sequence[str]
    ) -> list[tuple[str, float]]:
        self.calls.append((list(crops), list(vocabulary)))
        if self._inner is not None:
            return self._inner.label(crops, vocabulary)
        if self._fixed is not None:
            return [self._fixed for _ in crops]
        out = self._results[: len(crops)]
        self._results = self._results[len(crops) :]
        return out

    def unload(self) -> None:
        pass
