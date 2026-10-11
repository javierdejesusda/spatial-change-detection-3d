from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from spatial_change.ports import ReconstructionResult
from spatial_change.reconstruction.fake import FakeReconstructor


class ShortReconstructor(FakeReconstructor):
    """Drops the last view of every result."""

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        return super().reconstruct(images[:-1])
