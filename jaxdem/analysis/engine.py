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
from typing import Any
from collections.abc import Mapping
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax import ops

from .bins import BinSpec
from .pairs import Pairs, build_pairs

PyTree = Any

# Cache of jitted compute closures keyed by (path, kernel, kwargs digest, B[, chunk]).
# Without this, every `evaluate_binned` call wraps a fresh closure in `jax.jit`
# and recompiles even for identical kernels/shapes.
_JIT_CACHE_MAXSIZE = 16
_JIT_CACHE: OrderedDict[Any, Any] = OrderedDict()


def clear_jit_cache() -> None:
    """Clear analysis-owned compiled-function references.

    This does not clear JAX's process-wide compilation caches.
    """
    _JIT_CACHE.clear()


def _kwargs_cache_key(kernel_kwargs: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Build a hashable digest of kernel kwargs, or None if not possible."""
    items: list[Any] = []
    for k in sorted(kernel_kwargs):
        v = kernel_kwargs[k]
        if v is None or isinstance(v, (bool, int, float, str)):
            items.append((k, type(v).__name__, v))
        else:
            # Array contents can be large and copying them into a cache key is
            # more expensive than compiling an uncached closure.
            return None
    return tuple(items)


def _cached_jit(fn: Any, cache_key: Any) -> Any:
    """Return a jitted version of `fn`. Reuse a cached jitted function when possible."""
    if cache_key is None:
        return jax.jit(fn)
    try:
        jitted = _JIT_CACHE.get(cache_key)
        if jitted is None:
            jitted = jax.jit(fn)
            _JIT_CACHE[cache_key] = jitted
            if len(_JIT_CACHE) > _JIT_CACHE_MAXSIZE:
                _JIT_CACHE.popitem(last=False)
        else:
            _JIT_CACHE.move_to_end(cache_key)
        return jitted
    except TypeError:  # unhashable component (e.g. exotic kernel object)
        return jax.jit(fn)


@dataclass(frozen=True)
class Binned:
    """Binned accumulation output.

    Attributes:
        sums: pytree with each leaf shaped (B, ...)
        counts: integer array shape (B,)
        mean: pytree with each leaf shaped (B, ...)
        pairs: flattened pair representation used for the run (host arrays)

    """

    sums: PyTree
    counts: jnp.ndarray
    mean: PyTree
    pairs: Pairs


def _compute_mean_and_mask(sums: PyTree, counts: jnp.ndarray, B: int) -> PyTree:
    """Compute per-bin mean from sums/counts and NaN-mask empty bins."""

    def mean_leaf(s: jnp.ndarray) -> jnp.ndarray:
        denom = jnp.maximum(counts, 1).astype(jnp.promote_types(s.dtype, float))
        reshape = (B,) + (1,) * (s.ndim - 1)
        return s / denom.reshape(reshape)

    mean = tree_util.tree_map(mean_leaf, sums)

    def mask_empty(m: jnp.ndarray) -> jnp.ndarray:
        empty = counts == 0
        reshape = (B,) + (1,) * (m.ndim - 1)
        return jnp.where(empty.reshape(reshape), jnp.nan, m)

    return tree_util.tree_map(mask_empty, mean)


def evaluate_binned(
    kernel: Any,
    arrays: Mapping[str, Any],
    binspec: BinSpec,
    *,
    kernel_kwargs: dict[str, Any] | None = None,
    jit: bool = True,
    chunk_size: int | None = None,
    max_pairs: int | None = None,
) -> Binned:
    """Run a kernel over bins and average the results in JAX.

    Args:
        kernel: pure function called as `kernel(arrays, t0, t1, **kernel_kwargs)`.
        arrays: mapping of field name -> array with leading time axis, e.g.
            pos: (T,N,d) or (T,S,N,d)
        binspec: bin specification built on the host. Defines which indices to use.
        kernel_kwargs: extra keyword arguments for the kernel.
        jit: whether to jit the core compute.
        chunk_size: optional maximum number of pairs transferred to the device
            and evaluated per chunk. With the
            default *None*, one ``jax.vmap`` call processes all pairs. Set a
            positive integer to process pairs in chunks with
            repeated compiled calls. Chunking keeps peak device pair storage
            proportional to *chunk_size*. The returned ``pairs`` arrays still
            retain one host index triple per pair.
        max_pairs: optional bound on the number of host pair-index triples.
            The default ``None`` retains all pairs without a host-memory bound.
            The estimate is checked before allocation and the bound is also
            enforced while enumerating custom bin specifications.

    """
    kernel_kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)

    # Flatten binspec once on host
    pairs = build_pairs(binspec, max_pairs=max_pairs)
    B = int(binspec.num_bins())
    P = int(pairs.pair_i.shape[0])

    # Convert everything to a JAX pytree.
    # JAX treats dicts as pytrees with static keys (sorted).
    arrays_tree = {str(k): jnp.asarray(v) for k, v in arrays.items()}

    if P == 0:
        # Infer the normal output tree without vmapping an empty axis. This is
        # well-defined when the input contains at least one frame, including a
        # custom BinSpec whose bins happen to be empty.
        sample = kernel(arrays_tree, jnp.array(0), jnp.array(0), **kernel_kwargs)
        ones = jnp.zeros((0,), dtype=int)
        counts = ops.segment_sum(ones, jnp.zeros((0,), dtype=int), num_segments=B)
        sums = tree_util.tree_map(
            lambda v: jnp.zeros((B, *jnp.asarray(v).shape), jnp.asarray(v).dtype),
            sample,
        )
        return Binned(
            sums=sums,
            counts=counts,
            mean=_compute_mean_and_mask(sums, counts, B),
            pairs=pairs,
        )

    # ------------------------------------------------------------------
    # Decide between the single-shot path and the chunked path.
    # ------------------------------------------------------------------
    if chunk_size is None or chunk_size >= P:
        # ---- Original single-shot path (unchanged) --------------------
        pair_i = jnp.asarray(pairs.pair_i, dtype=int)
        pair_j = jnp.asarray(pairs.pair_j, dtype=int)
        bin_id = jnp.asarray(pairs.bin_id, dtype=int)

        def compute(
            pair_i: jnp.ndarray,
            pair_j: jnp.ndarray,
            bin_id: jnp.ndarray,
            arrays_tree: Mapping[str, jnp.ndarray],
        ) -> tuple[PyTree, jnp.ndarray, PyTree]:
            def per_pair(i: jnp.ndarray, j: jnp.ndarray) -> PyTree:
                return kernel(arrays_tree, i, j, **kernel_kwargs)

            vals = jax.vmap(per_pair, in_axes=(0, 0))(pair_i, pair_j)

            # Accumulate counts in (at least) int32/int64: float32 accumulation
            # silently saturates past ~16.7M pairs per bin.
            ones = jnp.ones((bin_id.shape[0],), dtype=int)
            counts = ops.segment_sum(ones, bin_id, num_segments=B)  # (B,)

            def segsum(v: jnp.ndarray) -> jnp.ndarray:
                return ops.segment_sum(v, bin_id, num_segments=B)

            sums = tree_util.tree_map(segsum, vals)
            mean = _compute_mean_and_mask(sums, counts, B)
            return sums, counts, mean

        if jit:
            cache_key: tuple[Any, ...] | None = None
            kw_key = _kwargs_cache_key(kernel_kwargs)
            if kw_key is not None:
                cache_key = ("single", kernel, kw_key, B)
            fn = _cached_jit(compute, cache_key)
        else:
            fn = compute
        sums, counts, mean = fn(pair_i, pair_j, bin_id, arrays_tree)

    else:
        # ---- Chunked path via lax.scan --------------------------------
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")

        def compute_chunk(
            pair_i: jnp.ndarray,
            pair_j: jnp.ndarray,
            bin_id: jnp.ndarray,
            arrays_tree: Mapping[str, jnp.ndarray],
        ) -> tuple[PyTree, jnp.ndarray]:
            def per_pair(i: jnp.ndarray, j: jnp.ndarray) -> PyTree:
                return kernel(arrays_tree, i, j, **kernel_kwargs)

            # Evaluate the kernel on a single pair to infer the output
            # pytree structure and leaf shapes/dtypes for the accumulator.
            vals = jax.vmap(per_pair, in_axes=(0, 0))(pair_i, pair_j)
            counts = ops.segment_sum(
                jnp.ones((chunk_size,), dtype=int), bin_id, num_segments=B + 1
            )
            sums = tree_util.tree_map(
                lambda v: ops.segment_sum(v, bin_id, num_segments=B + 1), vals
            )
            return sums, counts

        if jit:
            cache_key = None
            kw_key = _kwargs_cache_key(kernel_kwargs)
            if kw_key is not None:
                cache_key = ("chunked", kernel, kw_key, B, int(chunk_size))
            fn_chunked: Any = _cached_jit(compute_chunk, cache_key)
        else:
            fn_chunked = compute_chunk

        sums = None
        counts = jnp.zeros((B + 1,), dtype=int)
        for start in range(0, P, chunk_size):
            stop = min(start + chunk_size, P)
            valid = stop - start
            pi_host = np.zeros(chunk_size, dtype=int)
            pj_host = np.zeros(chunk_size, dtype=int)
            bi_host = np.full(chunk_size, B, dtype=int)
            pi_host[:valid] = pairs.pair_i[start:stop]
            pj_host[:valid] = pairs.pair_j[start:stop]
            bi_host[:valid] = pairs.bin_id[start:stop]
            chunk_sums, chunk_counts = fn_chunked(
                jnp.asarray(pi_host),
                jnp.asarray(pj_host),
                jnp.asarray(bi_host),
                arrays_tree,
            )
            sums = (
                chunk_sums
                if sums is None
                else tree_util.tree_map(jnp.add, sums, chunk_sums)
            )
            counts = counts + chunk_counts

        sums = tree_util.tree_map(lambda s: s[:B], sums)
        counts = counts[:B]
        mean = _compute_mean_and_mask(sums, counts, B)

    return Binned(sums=sums, counts=counts, mean=mean, pairs=pairs)
