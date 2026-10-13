import math

import pytest

from spatial_change.domain.report import EvidenceCounts
from spatial_change.visibility import PointState, VisibilityConfig, Vote


def test_vote_values_are_the_count_columns() -> None:
    assert [v.value for v in Vote] == [0, 1, 2, 3]
    assert [v.name.lower() for v in Vote] == list(EvidenceCounts.model_fields)


def test_point_state_codes() -> None:
    assert PointState.UNKNOWN == 0
    assert PointState.PRESENT == 1
    assert PointState.CHANGED == 2


def test_defaults() -> None:
    config = VisibilityConfig()

    assert config.abs_tolerance == 0.05
    assert config.rel_tolerance == 0.02
    assert config.confidence_slack == 1.0
    assert config.min_confidence == 0.1
    assert config.neighbourhood == 1
    assert config.min_free_votes == 1
    assert config.min_consistent_votes == 1
    assert config.batch_size == 262_144
    assert config.device == "cpu"
    assert config.veto is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"abs_tolerance": 0.0},
        {"abs_tolerance": -0.1},
        {"abs_tolerance": math.nan},
        {"abs_tolerance": math.inf},
        {"rel_tolerance": -0.01},
        {"rel_tolerance": math.nan},
        {"rel_tolerance": math.inf},
        {"confidence_slack": -1.0},
        {"confidence_slack": math.nan},
        {"confidence_slack": math.inf},
        {"min_confidence": -0.1},
        {"min_confidence": 1.1},
        {"min_confidence": math.nan},
        {"neighbourhood": -1},
        {"min_free_votes": 0},
        {"min_consistent_votes": 0},
        {"batch_size": 0},
        {"device": ""},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be"):
        VisibilityConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rel_tolerance": 0.0},
        {"confidence_slack": 0.0},
        {"min_confidence": 0.0},
        {"min_confidence": 1.0},
        {"neighbourhood": 0},
    ],
)
def test_boundary_config_is_accepted(kwargs: dict[str, object]) -> None:
    VisibilityConfig(**kwargs)  # type: ignore[arg-type]
