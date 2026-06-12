# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Helpers for converting bins into flat index-pairs.

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


def build_pairs(binspec: BinSpec) -> Pairs:
    """Build (pair_i, pair_j, bin_id) arrays from a BinSpec."""
    B = binspec.num_bins()
    pair_i_chunks: list[np.ndarray] = []
    pair_j_chunks: list[np.ndarray] = []
    bin_id_chunks: list[np.ndarray] = []
    counts = np.zeros((B,), dtype=np.int64)

    for b in range(B):
        tuples = [idxs for idxs in binspec.iter_tuples(b) if idxs]
        cnt = len(tuples)
        counts[b] = cnt
        if cnt == 0:
            continue
        arr = np.asarray(tuples, dtype=np.int32)
        pair_i_chunks.append(arr[:, 0])
        pair_j_chunks.append(arr[:, -1])
        bin_id_chunks.append(np.full((cnt,), b, dtype=np.int32))

    if pair_i_chunks:
        pair_i = np.concatenate(pair_i_chunks)
        pair_j = np.concatenate(pair_j_chunks)
        bin_id = np.concatenate(bin_id_chunks)
    else:
        pair_i = np.empty((0,), dtype=np.int32)
        pair_j = np.empty((0,), dtype=np.int32)
        bin_id = np.empty((0,), dtype=np.int32)
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
