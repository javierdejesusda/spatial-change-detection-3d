import subprocess
import sys

import spatial_change.reconstruction as reconstruction


def test_package_exports_the_public_names() -> None:
    assert set(reconstruction.__all__) == {
        "ChunkAlignment",
        "DensifiedEpoch",
        "DensifyConfig",
        "FakeReconstructor",
        "JointReconstruction",
        "Pi3XConfig",
        "Pi3XReconstructor",
        "densify_epoch",
        "fuse_frames",
        "reconstruct_joint",
    }
    for name in reconstruction.__all__:
        assert hasattr(reconstruction, name)


def test_importing_the_package_does_not_load_torch_or_pi3() -> None:
    code = (
        "import sys\n"
        "import spatial_change.reconstruction\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'pi3' not in sys.modules\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
