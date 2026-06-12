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
_JIT_CACHE: dict[Any, Any] = {}


def _kwargs_cache_key(kernel_kwargs: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Build a hashable digest of kernel kwargs, or None if not possible."""
    items: list[Any] = []
    for k in sorted(kernel_kwargs):
        v = kernel_kwargs[k]
        if v is None or isinstance(v, (bool, int, float, str)):
            items.append((k, type(v).__name__, v))
        else:
            try:
                arr = np.asarray(v)
                items.append((k, arr.dtype.str, arr.shape, arr.tobytes()))
            except Exception:
                return None
    return tuple(items)


def _cached_jit(fn: Any, cache_key: Any) -> Any:
    """Return a jitted version of `fn`, reusing a cached one when possible."""
    if cache_key is None:
        return jax.jit(fn)
    try:
        jitted = _JIT_CACHE.get(cache_key)
        if jitted is None:
            jitted = jax.jit(fn)
            _JIT_CACHE[cache_key] = jitted
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
        denom = jnp.maximum(counts, 1).astype(jnp.promote_types(s.dtype, jnp.float32))
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
) -> Binned:
    """Run a kernel over bins and average in JAX.

    Args:
        kernel: pure function called as `kernel(arrays, t0, t1, **kernel_kwargs)`.
        arrays: mapping of field name -> array with leading time axis, e.g.
            pos: (T,N,d) or (T,S,N,d)
        binspec: bin specification (host-side); defines which indices to use.
        kernel_kwargs: passed to kernel.
        jit: whether to jit the core compute.
        chunk_size: optional maximum number of pairs to evaluate per chunk.
            When *None* (the default) all pairs are processed in a single
            ``jax.vmap`` call — identical to the previous behaviour.  Set to
            a positive integer to process pairs in chunks via
            ``jax.lax.scan``, which keeps peak device memory proportional to
            *chunk_size* rather than the total number of pairs.

    """
    kernel_kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)

    # Flatten binspec once on host
    pairs = build_pairs(binspec)
    B = int(binspec.num_bins())
    P = int(pairs.pair_i.shape[0])

    # Convert everything to a JAX pytree.
    # JAX treats dicts as pytrees with static keys (sorted).
    arrays_tree = {str(k): jnp.asarray(v) for k, v in arrays.items()}

    if P == 0:
        # No pairs -> produce empty bins
        ones = jnp.zeros((0,), dtype=jnp.int64)
        counts = ops.segment_sum(ones, jnp.zeros((0,), dtype=jnp.int32), num_segments=B)
        # We cannot infer leaf shapes without running kernel; return empty dict.
        return Binned(sums={}, counts=counts, mean={}, pairs=pairs)

    # ------------------------------------------------------------------
    # Decide between the single-shot path and the chunked path.
    # ------------------------------------------------------------------
    if chunk_size is None or chunk_size >= P:
        # ---- Original single-shot path (unchanged) --------------------
        pair_i = jnp.asarray(pairs.pair_i, dtype=jnp.int32)
        pair_j = jnp.asarray(pairs.pair_j, dtype=jnp.int32)
        bin_id = jnp.asarray(pairs.bin_id, dtype=jnp.int32)

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
            ones = jnp.ones((bin_id.shape[0],), dtype=jnp.int64)
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

        n_chunks = -(-P // chunk_size)  # ceil division
        pad_total = n_chunks * chunk_size

        # Pad on host.  Overflow pairs use index 0 (valid, cheap to
        # evaluate) but are directed to dummy bin *B* which is discarded.
        pi_host = np.zeros(pad_total, dtype=np.int32)
        pj_host = np.zeros(pad_total, dtype=np.int32)
        bi_host = np.full(pad_total, B, dtype=np.int32)
        pi_host[:P] = pairs.pair_i
        pj_host[:P] = pairs.pair_j
        bi_host[:P] = pairs.bin_id

        # Transfer to device as (n_chunks, chunk_size)
        pair_i_c = jnp.asarray(pi_host.reshape(n_chunks, chunk_size))
        pair_j_c = jnp.asarray(pj_host.reshape(n_chunks, chunk_size))
        bin_id_c = jnp.asarray(bi_host.reshape(n_chunks, chunk_size))

        def compute_chunked(
            pair_i_c: jnp.ndarray,
            pair_j_c: jnp.ndarray,
            bin_id_c: jnp.ndarray,
            arrays_tree: Mapping[str, jnp.ndarray],
        ) -> tuple[PyTree, jnp.ndarray, PyTree]:
            def per_pair(i: jnp.ndarray, j: jnp.ndarray) -> PyTree:
                return kernel(arrays_tree, i, j, **kernel_kwargs)

            # Evaluate the kernel on a single pair to infer the output
            # pytree structure and leaf shapes/dtypes for the accumulator.
            _sample = per_pair(pair_i_c[0, 0], pair_j_c[0, 0])
            init_sums = tree_util.tree_map(
                lambda v: jnp.zeros((B + 1, *v.shape), v.dtype), _sample
            )
            init_counts = jnp.zeros((B + 1,), dtype=jnp.int64)

            def scan_body(
                carry: tuple[PyTree, jnp.ndarray],
                xs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            ) -> tuple[tuple[PyTree, jnp.ndarray], None]:
                sums_acc, counts_acc = carry
                pi, pj, bi = xs

                vals = jax.vmap(per_pair, in_axes=(0, 0))(pi, pj)

                ones = jnp.ones((chunk_size,), dtype=jnp.int64)
                chunk_counts = ops.segment_sum(ones, bi, num_segments=B + 1)
                chunk_sums = tree_util.tree_map(
                    lambda v: ops.segment_sum(v, bi, num_segments=B + 1),
                    vals,
                )

                new_sums = tree_util.tree_map(jnp.add, sums_acc, chunk_sums)
                new_counts = counts_acc + chunk_counts
                return (new_sums, new_counts), None

            (sums_full, counts_full), _ = jax.lax.scan(
                scan_body,
                (init_sums, init_counts),
                (pair_i_c, pair_j_c, bin_id_c),
            )

            # Discard the dummy bin (index B)
            sums = tree_util.tree_map(lambda s: s[:B], sums_full)
            counts = counts_full[:B]

            mean = _compute_mean_and_mask(sums, counts, B)
            return sums, counts, mean

        if jit:
            cache_key = None
            kw_key = _kwargs_cache_key(kernel_kwargs)
            if kw_key is not None:
                cache_key = ("chunked", kernel, kw_key, B, int(chunk_size))
            fn_chunked: Any = _cached_jit(compute_chunked, cache_key)
        else:
            fn_chunked = compute_chunked
        sums, counts, mean = fn_chunked(pair_i_c, pair_j_c, bin_id_c, arrays_tree)

    return Binned(sums=sums, counts=counts, mean=mean, pairs=pairs)
