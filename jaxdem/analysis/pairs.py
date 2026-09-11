# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Convert bins into flat index-pairs.

The JAX engine operates on a flat list of pairs (t0, t1) and a `bin_id` per pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bins import BinSpec


@dataclass(frozen=True)
class Pairs:
    """Flat representation of bin tuples, suitable for JAX execution.

    Attributes:
        pair_i: shape (P,) int array
        pair_j: shape (P,) int array
        bin_id: shape (P,) int array in [0, B)
        counts_per_bin: shape (B,) int array (number of tuples per bin)

    """

    pair_i: np.ndarray
    pair_j: np.ndarray
    bin_id: np.ndarray
    counts_per_bin: np.ndarray


def build_pairs(binspec: BinSpec, *, max_pairs: int | None = None) -> Pairs:
    """Build (pair_i, pair_j, bin_id) arrays from a BinSpec.

    The returned representation necessarily uses memory proportional to the
    number of pairs. Arrays are allocated once, avoiding an additional list of
    per-bin arrays and concatenation copies for large analyses. Set
    ``max_pairs`` to reject work beyond an explicit host-allocation budget.
    """
    if max_pairs is not None and (
        isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 0
    ):
        raise ValueError("max_pairs must be a nonnegative integer or None")
    B = binspec.num_bins()
    counts = np.fromiter((binspec.weight(b) for b in range(B)), dtype=int, count=B)
    total = int(counts.sum())
    if max_pairs is not None and total > max_pairs:
        raise ValueError(
            f"BinSpec emits an estimated {total} pairs, exceeding max_pairs={max_pairs}"
        )
    pair_i = np.empty((total,), dtype=int)
    pair_j = np.empty((total,), dtype=int)
    bin_id = np.empty((total,), dtype=int)
    cursor = 0
    actual_counts = np.zeros((B,), dtype=int)
    for b in range(B):
        for idxs in binspec.iter_tuples(b):
            if not idxs:
                continue
            if max_pairs is not None and cursor >= max_pairs:
                raise ValueError(f"BinSpec emits more than max_pairs={max_pairs} pairs")
            if cursor >= pair_i.size:
                new_size = max(1, 2 * pair_i.size)
                pair_i.resize(new_size, refcheck=False)
                pair_j.resize(new_size, refcheck=False)
                bin_id.resize(new_size, refcheck=False)
            pair_i[cursor] = idxs[0]
            pair_j[cursor] = idxs[-1]
            bin_id[cursor] = b
            cursor += 1
            actual_counts[b] += 1
    if cursor != pair_i.size:
        pair_i = pair_i[:cursor]
        pair_j = pair_j[:cursor]
        bin_id = bin_id[:cursor]
    counts = actual_counts
    return Pairs(pair_i=pair_i, pair_j=pair_j, bin_id=bin_id, counts_per_bin=counts)


def flatten_pairs(binspec: BinSpec) -> Pairs:
    """Deprecated alias for build_pairs()."""
    import warnings

    warnings.warn(
        "jaxdem.analysis.flatten_pairs is deprecated; use build_pairs instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_pairs(binspec)


FlatPairs = Pairs  # backwards-compatible alias
