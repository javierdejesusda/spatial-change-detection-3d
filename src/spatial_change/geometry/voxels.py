"""Voxel grid keys and the connected components of occupied voxels."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Voxels the linear key of the grid can index, as a float to compare spans
# that are too large for an int64.
_MAX_VOXELS = 2.0**63


def voxel_cells(
    points: NDArray[np.float64], voxel_size: float
) -> NDArray[np.int64]:
    """Gives the grid key of the voxel of every point.

    Args:
        points: World points of shape ``(N, 3)``, finite and within the
            range of an int64 key.
        voxel_size: Voxel edge.

    Returns:
        Array of shape ``(N, 3)``.
    """
    return np.floor(points / voxel_size).astype(np.int64)


def _forward_offsets(radius: int) -> NDArray[np.int64]:
    """Lists the neighbour offsets that come after zero in key order.

    Args:
        radius: Largest absolute offset per axis.

    Returns:
        Array of shape ``(((2r + 1)^3 - 1) / 2, 3)``: one offset of every
        opposite pair, so each neighbouring pair is visited once.
    """
    span = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)[(2 * radius + 1) ** 3 // 2 + 1 :]


def lookup_keys(
    packed: NDArray[np.int64], keys: ArrayLike
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Finds linear voxel keys among the occupied ones.

    Args:
        packed: Sorted unique linear voxel keys, shape ``(V,)``, at least
            one.
        keys: Linear keys to find, shape ``(K,)``.

    Returns:
        The rows of ``packed`` and a mask of shape ``(K,)``. A row holds its
        key where the mask is true; elsewhere it is only a valid index.
    """
    slots = np.minimum(np.searchsorted(packed, keys), len(packed) - 1)
    return slots, packed[slots] == keys


def label_packed_voxels(
    packed: NDArray[np.int64], dims: tuple[int, ...], radius: int
) -> NDArray[np.int64]:
    """Labels occupied voxels by connected component.

    Every neighbouring pair is found by a binary search of the shifted
    key. The components then grow by hooking each edge's roots onto the
    smaller one and halving the trees by pointer jumping until no edge
    joins two trees.

    Args:
        packed: Sorted unique linear voxel keys, shape ``(V,)``.
        dims: Grid size the linear keys are packed with.
        radius: Neighbourhood radius in voxels.

    Returns:
        For every voxel the row of the first voxel of its component.
    """
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    heads, tails = [], []
    for offset in _forward_offsets(radius):
        slot, linked = lookup_keys(packed, packed + offset @ strides)
        heads.append(np.flatnonzero(linked))
        tails.append(slot[linked])
    head, tail = np.concatenate(heads), np.concatenate(tails)
    parent = np.arange(len(packed))
    while head.size:
        root_head, root_tail = parent[head], parent[tail]
        lower = np.minimum(root_head, root_tail)
        np.minimum.at(parent, root_head, lower)
        np.minimum.at(parent, root_tail, lower)
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
        open_edges = parent[head] != parent[tail]
        head, tail = head[open_edges], tail[open_edges]
    return parent


def voxel_grid(
    cells: NDArray[np.int64] | NDArray[np.float64], radius: int
) -> tuple[list[float], list[float], tuple[int, ...]]:
    """Sizes the grid that holds voxel keys and their neighbours.

    The arithmetic is done on Python numbers, exact for int64 keys and
    overflowing to infinity for float keys, so no span wraps around.

    Args:
        cells: Voxel keys of shape ``(N, 3)``, at least one row.
        radius: Neighbourhood radius in voxels, kept free around the keys.

    Returns:
        The key of the grid origin, the grid size per axis, and the same
        size as ints.

    Raises:
        ValueError: If the grid has more voxels than a 64-bit key can
            index.
    """
    origin = [low - radius for low in cells.min(axis=0).tolist()]
    spans = [
        high + radius - low + 1
        for high, low in zip(cells.max(axis=0).tolist(), origin, strict=True)
    ]
    # Negated so that a NaN span, which no comparison accepts, is rejected.
    if not math.prod(spans) < _MAX_VOXELS:
        raise ValueError("the points span too many voxels; increase voxel_size")
    return origin, spans, tuple(int(span) for span in spans)


def label_voxel_cells(
    cells: NDArray[np.int64], radius: int
) -> NDArray[np.int64]:
    """Labels voxel keys by connected component.

    Two keys are linked when they differ by at most ``radius`` on every
    axis.

    Args:
        cells: Voxel keys of shape ``(N, 3)``; repeated keys are allowed.
        radius: Neighbourhood radius in voxels, at least 1.

    Returns:
        Array of shape ``(N,)``: equal labels for keys of one component,
        and for repeats of one key.

    Raises:
        ValueError: If the keys span more voxels than a 64-bit key can
            index.
    """
    if len(cells) == 0:
        return np.empty(0, np.int64)
    origin, _, dims = voxel_grid(cells, radius)
    packed, inverse = np.unique(
        np.ravel_multi_index((cells - np.array(origin, np.int64)).T, dims),
        return_inverse=True,
    )
    return label_packed_voxels(packed, dims, radius)[inverse.reshape(-1)]
