import numpy as np
import pytest
import torch
from visibility_scenes import scene_clouds

from spatial_change.visibility import VisibilityConfig, vote_points

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_and_cuda_give_identical_counts() -> None:
    pair, before, _ = scene_clouds(0)
    frames = pair.after.frames

    cpu = vote_points(before.points, frames, VisibilityConfig(device="cpu"))
    cuda = vote_points(before.points, frames, VisibilityConfig(device="cuda"))

    np.testing.assert_array_equal(cuda.counts, cpu.counts)
    np.testing.assert_array_equal(cuda.states, cpu.states)
