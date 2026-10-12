"""Minimum-cost assignment between two sets, by the Hungarian method."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _solve_wide(
    cost: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Assigns every row of a matrix with at most as many rows as columns.

    Kuhn-Munkres with row and column potentials, one augmenting path per
    row, O(R * R * C).

    Args:
        cost: ``(R, C)`` finite costs with ``R <= C``.

    Returns:
        ``(R,)`` column chosen for each row.
    """
    rows, cols = cost.shape
    row_potential = np.zeros(rows + 1)
    col_potential = np.zeros(cols + 1)
    # Columns are 1-based; index 0 is the virtual column of the path start.
    owner = np.zeros(cols + 1, np.intp)
    way = np.zeros(cols + 1, np.intp)
    for row in range(1, rows + 1):
        owner[0] = row
        column = 0
        slack = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, np.bool_)
        while True:
            used[column] = True
            current = owner[column]
            reduced = (
                cost[current - 1] - row_potential[current] - col_potential[1:]
            )
            free = ~used[1:]
            better = free & (reduced < slack[1:])
            slack[1:][better] = reduced[better]
            way[1:][better] = column
            candidates = np.where(free, slack[1:], np.inf)
            nearest = int(np.argmin(candidates)) + 1
            delta = candidates[nearest - 1]
            row_potential[owner[used]] += delta
            col_potential[used] -= delta
            slack[~used] -= delta
            column = nearest
            if owner[column] == 0:
                break
        while column:
            previous = way[column]
            owner[column] = owner[previous]
            column = previous
    chosen = np.zeros(rows, np.intp)
    for column in range(1, cols + 1):
        if owner[column]:
            chosen[owner[column] - 1] = column - 1
    return chosen


def _as_cost(cost: ArrayLike) -> NDArray[np.float64]:
    """Validates a cost matrix.

    Args:
        cost: The candidate matrix.

    Returns:
        The matrix as a float64 array.

    Raises:
        ValueError: If it is not 2-D or has a value that is not finite.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"cost must be 2-D, got {matrix.ndim} dimensions")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cost must be finite")
    return matrix


def linear_assignment(
    cost: ArrayLike,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pairs rows with columns at the least total cost.

    A rectangular matrix pairs ``min(R, C)`` rows with distinct columns and
    leaves the rest of the longer side unpaired. Equal costs resolve the
    same way on every run.

    Args:
        cost: ``(R, C)`` matrix of finite costs; either side may be empty.

    Returns:
        The paired rows, ascending, and their columns.

    Raises:
        ValueError: If ``cost`` is not a 2-D array or has a value that is
            not finite.
    """
    matrix = _as_cost(cost)
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return np.empty(0, np.intp), np.empty(0, np.intp)
    if rows <= cols:
        return np.arange(rows), _solve_wide(matrix)
    paired_cols = np.arange(cols)
    paired_rows = _solve_wide(matrix.T)
    order = np.argsort(paired_rows)
    return paired_rows[order], paired_cols[order]


def gated_assignment(
    cost: ArrayLike, max_cost: float
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pairs rows with columns, leaving out the pairs that cost too much.

    The matrix is padded with a dummy column for every row and a dummy row
    for every column. Taking a dummy costs ``max_cost / 2`` and the dummies
    pair among themselves for free, so leaving both sides of a pair
    unmatched costs ``max_cost``. A pair is therefore made only when it
    costs less than ``max_cost``, and the pairs made are those of the least
    total cost with that option open, which is not always the cheapest
    pairing of the rows and columns that could be paired.

    Args:
        cost: ``(R, C)`` matrix of finite costs; either side may be empty.
        max_cost: Cost at and above which a pair is never made, positive and
            finite.

    Returns:
        The paired rows, ascending, and their columns.

    Raises:
        ValueError: If ``cost`` is not a 2-D array or has a value that is
            not finite, or ``max_cost`` is not positive and finite.
    """
    matrix = _as_cost(cost)
    if not 0.0 < max_cost < math.inf:
        raise ValueError("max_cost must be positive and finite")
    rows, cols = matrix.shape
    padded = np.zeros((rows + cols, rows + cols))
    padded[:rows, :cols] = matrix
    padded[:rows, cols:] = max_cost / 2.0
    padded[rows:, :cols] = max_cost / 2.0
    paired_rows, paired_cols = linear_assignment(padded)
    real = (paired_rows < rows) & (paired_cols < cols)
    paired_rows, paired_cols = paired_rows[real], paired_cols[real]
    cheap = matrix[paired_rows, paired_cols] < max_cost
    return paired_rows[cheap], paired_cols[cheap]
