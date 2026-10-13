from pathlib import Path
from typing import NamedTuple

import pytest
from pipeline_scenes import TEMPLATE, run_pair
from visibility_scenes import scene_clouds

from spatial_change.evaluation.synthetic import (
    SyntheticScore,
    pool_scores,
    score_result,
)
from spatial_change.pipeline import RunConfig

SEEDS = range(11)
ABLATION_SEEDS = SEEDS[:3]
MIN_F1 = 0.9
NO_VETO = RunConfig.model_validate(
    {**TEMPLATE.model_dump(), "stages": {"visibility": {"veto": False}}}
)


class Outcome(NamedTuple):
    """The score of one seed and its changes that rest on no free vote."""

    score: SyntheticScore
    no_free_vote: int


def summary(seeds: int, score: SyntheticScore) -> str:
    return (
        f"synthetic suite: seeds {seeds}, tp {score.tp}, fp {score.fp}, "
        f"fn {score.fn}, precision {score.precision:.3f}, "
        f"recall {score.recall:.3f}, f1 {score.f1:.3f}, "
        f"fp_occluded {score.fp_occluded}, "
        f"fp_unobserved {score.fp_unobserved}"
    )


def run_suite(
    seeds: range, config: RunConfig, root: Path
) -> dict[int, Outcome]:
    outcomes = {}
    for seed in seeds:
        result, _ = run_pair(root / f"seed{seed}", config, seed=seed)
        outcomes[seed] = Outcome(
            score_result(result, scene_clouds(seed)[0]),
            sum(c.evidence.free == 0 for c in result.report.changes),
        )
    return outcomes


def pooled(outcomes: dict[int, Outcome], seeds: range) -> SyntheticScore:
    return pool_scores(outcomes[s].score for s in seeds)


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Outcome]:
    return run_suite(SEEDS, TEMPLATE, tmp_path_factory.mktemp("suite"))


@pytest.fixture(scope="module")
def without_veto(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[int, Outcome]:
    return run_suite(
        ABLATION_SEEDS, NO_VETO, tmp_path_factory.mktemp("ablated")
    )


@pytest.mark.slow
def test_the_changes_are_found_and_nothing_hidden_or_unseen_is_reported(
    suite: dict[int, Outcome],
) -> None:
    total = pooled(suite, SEEDS)
    print(summary(len(SEEDS), total))

    assert total.f1 >= MIN_F1
    assert total.fp_occluded == 0
    assert total.fp_unobserved == 0


@pytest.mark.slow
def test_without_the_veto_changes_rest_on_votes_that_are_not_free(
    suite: dict[int, Outcome], without_veto: dict[int, Outcome]
) -> None:
    kept = pooled(suite, ABLATION_SEEDS)
    dropped = pooled(without_veto, ABLATION_SEEDS)
    print(summary(len(ABLATION_SEEDS), dropped).replace("suite", "no veto"))

    assert kept.fp == 0
    assert sum(suite[s].no_free_vote for s in ABLATION_SEEDS) == 0
    assert dropped.fp > 0
    assert dropped.f1 < MIN_F1
    assert sum(o.no_free_vote for o in without_veto.values()) > 0
