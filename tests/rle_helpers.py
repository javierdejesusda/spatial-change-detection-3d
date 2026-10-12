"""Compressed COCO RLE encoder used to build hand-made test masks."""

import numpy as np
from numpy.typing import NDArray


def run_lengths(mask: NDArray[np.bool_]) -> list[int]:
    """Column-major run lengths, starting with a (possibly empty) 0 run."""
    flat = mask.T.reshape(-1)
    runs: list[int] = []
    current = False
    count = 0
    for value in flat:
        if bool(value) == current:
            count += 1
        else:
            runs.append(count)
            current = bool(value)
            count = 1
    runs.append(count)
    return runs


def encode_counts(runs: list[int]) -> str:
    """Encodes run lengths with the compressed COCO string scheme."""
    out = []
    for i, run in enumerate(runs):
        x = run - runs[i - 2] if i > 2 else run
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if c & 0x10 else x != 0
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def encode_mask(mask: NDArray[np.bool_]) -> dict[str, object]:
    """Encodes a mask as a compressed RLE dict with ``size = [h, w]``."""
    height, width = mask.shape
    return {
        "size": [height, width],
        "counts": encode_counts(run_lengths(mask)),
    }
