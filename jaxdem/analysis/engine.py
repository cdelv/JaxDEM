# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
# JAX binned accumulation engine (vmap + segment_sum).
#
# Minimal, high-performance path:
# - Precompute (pair_i, pair_j, bin_id) once on host from a BinSpec.
# - vmap a *pure* kernel over pairs.
# - Reduce into bins with `jax.ops.segment_sum`.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util
from jax import ops

from .bins import BinSpec
from .pairs import Pairs, build_pairs

PyTree = Any


@dataclass(frozen=True)
class Binned:
    """Binned accumulation output.

    Attributes:
        sums: pytree with each leaf shaped (B, ...)
        counts: array shape (B,) float32
        mean: pytree with each leaf shaped (B, ...)
        pairs: flattened pair representation used for the run (host arrays)
    """

    sums: PyTree
    counts: jnp.ndarray
    mean: PyTree
    pairs: Pairs


def evaluate_binned(
    kernel: Any,
    arrays: Mapping[str, Any],
    binspec: BinSpec,
    *,
    kernel_kwargs: Optional[Dict[str, Any]] = None,
    jit: bool = True,
) -> Binned:
    """Run a kernel over bins and average in JAX.

    Args:
        kernel: pure function called as `kernel(arrays, t0, t1, **kernel_kwargs)`.
        arrays: mapping of field name -> array with leading time axis, e.g.
            pos: (T,N,d) or (T,S,N,d)
        binspec: bin specification (host-side); defines which indices to use.
        kernel_kwargs: passed to kernel.
        jit: whether to jit the core compute.
    """

    kernel_kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)

    # Flatten binspec once on host
    pairs = build_pairs(binspec)
    B = int(binspec.num_bins())

    # Convert indices to device arrays
    pair_i = jnp.asarray(pairs.pair_i, dtype=jnp.int32)
    pair_j = jnp.asarray(pairs.pair_j, dtype=jnp.int32)
    bin_id = jnp.asarray(pairs.bin_id, dtype=jnp.int32)

    # Convert everything to a JAX pytree.
    # JAX treats dicts as pytrees with static keys (sorted).
    arrays_tree = {str(k): jnp.asarray(v) for k, v in arrays.items()}

    if pair_i.size == 0:
        # No pairs -> produce empty bins
        ones = jnp.zeros((0,), dtype=jnp.float32)
        counts = ops.segment_sum(ones, jnp.zeros((0,), dtype=jnp.int32), num_segments=B)
        # We cannot infer leaf shapes without running kernel; return empty dict.
        return Binned(sums={}, counts=counts, mean={}, pairs=pairs)

    def compute(
        pair_i: jnp.ndarray,
        pair_j: jnp.ndarray,
        bin_id: jnp.ndarray,
        arrays_tree: Mapping[str, jnp.ndarray],
    ) -> Tuple[PyTree, jnp.ndarray, PyTree]:
        def per_pair(i: jnp.ndarray, j: jnp.ndarray) -> PyTree:
            return kernel(arrays_tree, i, j, **kernel_kwargs)

        vals = jax.vmap(per_pair, in_axes=(0, 0))(pair_i, pair_j)

        ones = jnp.ones((bin_id.shape[0],), dtype=jnp.float32)
        counts = ops.segment_sum(ones, bin_id, num_segments=B)  # (B,)

        def segsum(v: jnp.ndarray) -> jnp.ndarray:
            return ops.segment_sum(v, bin_id, num_segments=B)

        sums = tree_util.tree_map(segsum, vals)

        def mean_leaf(s: jnp.ndarray) -> jnp.ndarray:
            denom = jnp.maximum(counts, 1.0)
            reshape = (B,) + (1,) * (s.ndim - 1)
            return s / denom.reshape(reshape)

        mean = tree_util.tree_map(mean_leaf, sums)

        # NaN out empty bins
        def mask_empty(m: jnp.ndarray) -> jnp.ndarray:
            empty = counts == 0
            # broadcast to leaf shape
            reshape = (B,) + (1,) * (m.ndim - 1)
            return jnp.where(empty.reshape(reshape), jnp.nan, m)

        mean = tree_util.tree_map(mask_empty, mean)
        return sums, counts, mean

    fn = jax.jit(compute) if jit else compute
    sums, counts, mean = fn(pair_i, pair_j, bin_id, arrays_tree)
    return Binned(sums=sums, counts=counts, mean=mean, pairs=pairs)


def run_binned_jax(*args: Any, **kwargs: Any) -> Binned:
    """Deprecated alias for evaluate_binned()."""

    import warnings

    warnings.warn(
        "jaxdem.analysis.run_binned_jax is deprecated; use evaluate_binned instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return evaluate_binned(*args, **kwargs)


BinnedResult = Binned  # backwards-compatible alias
