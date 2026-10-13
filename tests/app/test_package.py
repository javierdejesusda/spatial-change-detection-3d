import subprocess
import sys

CODE = """
import importlib
import sys

heavy = ('gradio', 'gradio_rerun', 'torch', 'transformers', 'open3d', 'cv2')
for module in ('spatial_change.cli', 'spatial_change.app.handlers'):
    importlib.import_module(module)
    loaded = [name for name in heavy if name in sys.modules]
    assert not loaded, (module, loaded)
"""


def test_importing_does_not_load_the_web_or_heavy_stacks() -> None:
    subprocess.run([sys.executable, "-I", "-c", CODE], check=True)
