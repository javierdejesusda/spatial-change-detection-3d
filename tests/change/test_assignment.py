import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from spatial_change.change.assignment import (
    gated_assignment,
    linear_assignment,
)


def _brute_force(cost: NDArray[np.float64]) -> float:
    rows, cols = cost.shape
    if rows <= cols:
        return min(
            sum(cost[r, c] for r, c in enumerate(choice))
            for choice in itertools.permutations(range(cols), rows)
        )
    return min(
        sum(cost[r, c] for c, r in enumerate(choice))
        for choice in itertools.permutations(range(rows), cols)
    )


@st.composite
def _matrices(draw: st.DrawFn) -> NDArray[np.float64]:
    rows = draw(st.integers(1, 5))
    cols = draw(st.integers(1, 5))
    values = draw(
        st.lists(
            st.floats(-10.0, 10.0),
            min_size=rows * cols,
            max_size=rows * cols,
        )
    )
    return np.array(values).reshape(rows, cols)


@settings(max_examples=200, deadline=None)
@given(_matrices())
def test_matches_the_brute_force_optimum(cost: NDArray[np.float64]) -> None:
    rows, cols = linear_assignment(cost)

    assert len(rows) == min(cost.shape)
    assert len(set(rows)) == len(rows)
    assert len(set(cols)) == len(cols)
    assert list(rows) == sorted(rows)
    assert cost[rows, cols].sum() == pytest.approx(_brute_force(cost), abs=1e-9)


def test_prefers_the_cheapest_total_over_the_greedy_choice() -> None:
    cost = np.array([[1.0, 2.0], [2.0, 10.0]])

    rows, cols = linear_assignment(cost)

    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 0]


def test_a_wide_matrix_leaves_columns_unassigned() -> None:
    cost = np.array([[5.0, 1.0, 9.0], [5.0, 9.0, 1.0]])

    rows, cols = linear_assignment(cost)

    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 2]


def test_a_tall_matrix_leaves_rows_unassigned() -> None:
    cost = np.array([[5.0, 9.0], [1.0, 9.0], [9.0, 1.0]])

    rows, cols = linear_assignment(cost)

    assert rows.tolist() == [1, 2]
    assert cols.tolist() == [0, 1]


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
def test_an_empty_matrix_gives_no_pairs(shape: tuple[int, int]) -> None:
    rows, cols = linear_assignment(np.zeros(shape))

    assert rows.tolist() == []
    assert cols.tolist() == []


def test_ties_resolve_deterministically() -> None:
    cost = np.ones((3, 3))

    first = linear_assignment(cost)
    second = linear_assignment(cost.copy())

    assert first[0].tolist() == second[0].tolist() == [0, 1, 2]
    assert first[1].tolist() == second[1].tolist()
    assert sorted(first[1].tolist()) == [0, 1, 2]


def test_large_costs_do_not_break_the_optimum() -> None:
    cost = np.array([[1e9, 2e9], [3e9, 1e9]])

    rows, cols = linear_assignment(cost)

    assert cols.tolist() == [0, 1]


@pytest.mark.parametrize(
    "cost",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 2, 2)),
        np.array([[1.0, np.nan], [1.0, 1.0]]),
        np.array([[1.0, np.inf], [1.0, 1.0]]),
    ],
)
def test_rejects_invalid_input(cost: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        linear_assignment(cost)


@pytest.mark.parametrize(
    ("cost", "rows", "cols"),
    [
        pytest.param(
            [[0.1, 0.9], [0.8, 0.2]], [0, 1], [0, 1], id="cheap-entries"
        ),
        pytest.param([[0.3, 0.9], [0.8, 0.7]], [0], [0], id="over-the-limit"),
        pytest.param([[0.5]], [], [], id="limit-is-exclusive"),
        # Matching the cheap pair (0, 0) is optimal even though it leaves the
        # second row, whose only partner is too costly, unmatched.
        pytest.param(
            [[0.1, 0.45], [0.2, 0.8]], [0], [0], id="no-forced-pairing"
        ),
    ],
)
def test_gated_assignment_pairs_only_the_entries_under_the_limit(
    cost: list[list[float]], rows: list[int], cols: list[int]
) -> None:
    paired_rows, paired_cols = gated_assignment(np.array(cost), max_cost=0.5)

    assert paired_rows.tolist() == rows
    assert paired_cols.tolist() == cols


def test_gated_assignment_handles_rectangles_and_empty_sides() -> None:
    wide = np.array([[0.4, 0.1, 0.9]])
    rows, cols = gated_assignment(wide, max_cost=0.5)
    assert (rows.tolist(), cols.tolist()) == ([0], [1])

    tall = np.array([[0.9], [0.2], [0.3]])
    rows, cols = gated_assignment(tall, max_cost=0.5)
    assert (rows.tolist(), cols.tolist()) == ([1], [0])

    rows, cols = gated_assignment(np.zeros((0, 4)), max_cost=0.5)
    assert rows.tolist() == [] and cols.tolist() == []


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
def test_gated_assignment_rejects_a_bad_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="max_cost"):
        gated_assignment(np.zeros((1, 1)), max_cost=limit)


@settings(max_examples=100, deadline=None)
@given(_matrices(), st.floats(0.5, 20.0))
def test_gated_assignment_is_optimal_with_the_option_to_skip(
    cost: NDArray[np.float64], limit: float
) -> None:
    rows, cols = gated_assignment(cost, limit)

    assert len(set(rows)) == len(rows)
    assert len(set(cols)) == len(cols)
    assert np.all(cost[rows, cols] < limit)
    skipped = (cost.shape[0] - len(rows)) + (cost.shape[1] - len(cols))
    paid = cost[rows, cols].sum() + skipped * limit / 2.0
    assert paid == pytest.approx(_brute_force_gated(cost, limit), abs=1e-9)


def _brute_force_gated(cost: NDArray[np.float64], limit: float) -> float:
    """Minimizes matched cost plus half the limit per unmatched side."""
    rows, cols = cost.shape
    best = (rows + cols) * limit / 2.0
    for size in range(1, min(rows, cols) + 1):
        for chosen_rows in itertools.combinations(range(rows), size):
            for chosen_cols in itertools.permutations(range(cols), size):
                matched = sum(
                    cost[r, c]
                    for r, c in zip(chosen_rows, chosen_cols, strict=True)
                )
                skipped = (rows - size) + (cols - size)
                best = min(best, matched + skipped * limit / 2.0)
    return float(best)
