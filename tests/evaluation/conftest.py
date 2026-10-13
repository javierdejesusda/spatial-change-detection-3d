import pytest
from pipeline_scenes import TEMPLATE, run_pair

from spatial_change.pipeline import PipelineResult


@pytest.fixture(scope="session")
def template_run(tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    """One run of the first synthetic scene pair, shared by the session."""
    result, _ = run_pair(tmp_path_factory.mktemp("template-run"), TEMPLATE)
    return result
