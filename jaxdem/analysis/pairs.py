# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
from __future__ import annotations

"""Helpers for converting bins into flat index-pairs.

The JAX engine operates on a flat list of pairs (t0, t1) and a `bin_id` per pair.
"""

from dataclasses import dataclass
from typing import List, Tuple

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
    pair_i_list: List[int] = []
    pair_j_list: List[int] = []
    bin_id_list: List[int] = []
    counts = np.zeros((B,), dtype=np.int64)

    for b in range(B):
        cnt = 0
        for idxs in binspec.iter_tuples(b):
            if not idxs:
                continue
            i = int(idxs[0])
            j = int(idxs[-1])
            pair_i_list.append(i)
            pair_j_list.append(j)
            bin_id_list.append(int(b))
            cnt += 1
        counts[b] = cnt

    pair_i = np.asarray(pair_i_list, dtype=np.int32)
    pair_j = np.asarray(pair_j_list, dtype=np.int32)
    bin_id = np.asarray(bin_id_list, dtype=np.int32)
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
