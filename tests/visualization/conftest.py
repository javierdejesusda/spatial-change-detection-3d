import pytest
from rrd_reading import Rrd
from visualization_scenes import record


@pytest.fixture(scope="module")
def rrd(tmp_path_factory: pytest.TempPathFactory) -> Rrd:
    return record(tmp_path_factory.mktemp("recording"))
