# SPDX-License-Identifier: BSD-3-Clause
"""Shared cell lookup with scalar and batch-wide branch selection."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


def _cell_lookup_dispatch(dense_lookup: Any, sparse_lookup: Any) -> Any:
    """Select one lookup method across all explicit batch axes."""

    @custom_vmap
    def dispatch(use_dense: Any, hashes: Any, queries: Any) -> Any:
        return jax.lax.cond(
            jnp.all(use_dense), dense_lookup, sparse_lookup, use_dense, hashes, queries
        )

    @dispatch.def_vmap
    def mapped(axis_size: int, in_batched: Any, *args: Any) -> Any:
        axes = tuple(0 if batched else None for batched in in_batched)

        def dense_batch(*values: Any) -> Any:
            return jax.vmap(dense_lookup, in_axes=axes, axis_size=axis_size)(*values)

        def sparse_batch(*values: Any) -> Any:
            return jax.vmap(sparse_lookup, in_axes=axes, axis_size=axis_size)(*values)

        # Another dispatcher lets an outer vmap collect the next explicit axis.
        return _cell_lookup_dispatch(dense_batch, sparse_batch)(*args), True

    return dispatch


@partial(jax.jit, inline=True, static_argnames=("mapped",))
def _lookup_cell_starts(
    sorted_hashes: jax.Array, queries: jax.Array, *, mapped: bool
) -> jax.Array:
    """Return the first sorted index for each queried cell hash.

    Dense, modest hash ranges use a temporary direct table. Sparse ranges use
    ``searchsorted`` instead. Missing hashes return the
    insertion point in the sparse branch and ``N`` in the dense branch; grid
    traversals verify the hash at that candidate before accepting the cell.
    """
    n_items = sorted_hashes.shape[0]
    if n_items == 0:
        if mapped:
            return jnp.zeros(queries.shape, dtype=jnp.int32)
        return jnp.zeros_like(queries, dtype=jnp.int32)

    capacity = 8 * n_items
    if capacity > 2**31 - 1:
        return jnp.searchsorted(
            sorted_hashes, queries, side="left", method="scan_unrolled"
        )

    if mapped:
        hash_capacity = jnp.asarray(capacity, dtype=sorted_hashes.dtype)
    else:
        # Scalar/manual mesh axes retain the variance of both inputs.
        prototype = jnp.zeros_like(sorted_hashes, shape=()) + jnp.zeros_like(
            queries, shape=()
        )
        hash_capacity = jnp.full_like(prototype, capacity, shape=())
    use_dense = sorted_hashes[-1] < hash_capacity

    def dense_lookup(
        original_dense: jax.Array, hashes: jax.Array, queries: jax.Array
    ) -> jax.Array:
        if mapped:
            hash_capacity = jnp.asarray(capacity, dtype=hashes.dtype)
            table = jnp.full((capacity,), n_items, dtype=jnp.int32)
        else:
            prototype = jnp.zeros_like(hashes, shape=()) + jnp.zeros_like(
                queries, shape=()
            )
            hash_capacity = jnp.full_like(prototype, capacity, shape=())
            table = jnp.full_like(
                prototype, n_items, shape=(capacity,), dtype=jnp.int32
            )
        stored_in_range = hashes < hash_capacity
        stored_indices = jnp.where(stored_in_range, hashes, hash_capacity).astype(
            jnp.int32
        )
        table = table.at[stored_indices].min(
            jax.lax.iota(jnp.int32, n_items), mode="drop"
        )

        query_in_range = queries < hash_capacity
        query_indices = jnp.where(query_in_range, queries, hash_capacity).astype(
            jnp.int32
        )
        return table.at[query_indices].get(mode="fill", fill_value=n_items)

    def sparse_lookup(
        original_dense: jax.Array, hashes: jax.Array, queries: jax.Array
    ) -> jax.Array:
        if not mapped:
            hashes = hashes + jnp.zeros_like(queries, shape=hashes.shape)
            queries = queries + jnp.zeros_like(hashes, shape=queries.shape)
        starts = jnp.searchsorted(hashes, queries, side="left", method="scan_unrolled")
        if not mapped:
            return starts
        present = (starts < n_items) & (
            hashes[jnp.minimum(starts, n_items - 1)] == queries
        )
        # Dense rows retain N for absent hashes even in a mixed sparse batch.
        return jnp.where(original_dense & ~present, n_items, starts)

    if mapped:
        return _cell_lookup_dispatch(dense_lookup, sparse_lookup)(
            use_dense, sorted_hashes, queries
        )
    return jax.lax.cond(
        use_dense,
        partial(dense_lookup, use_dense),
        partial(sparse_lookup, use_dense),
        sorted_hashes,
        queries,
    )


@jax.jit(inline=True)
def _cell_starts(sorted_hashes: jax.Array, queries: jax.Array) -> jax.Array:
    """Select dense or sparse lookup independently for each environment."""
    return _lookup_cell_starts(sorted_hashes, queries, mapped=False)


@custom_vmap
def _static_cell_starts(sorted_hashes: jax.Array, queries: jax.Array) -> jax.Array:
    """Keep scalar lookup and select one lookup branch for a mapped batch."""
    return _cell_starts(sorted_hashes, queries)


@_static_cell_starts.def_vmap
def _cell_starts_mapped(
    axis_size: int, in_batched: Any, sorted_hashes: jax.Array, queries: jax.Array
) -> tuple[jax.Array, bool]:
    """Use the batch dispatcher only when lookup arguments are mapped."""
    axes = tuple(0 if batched else None for batched in in_batched)
    result = jax.vmap(
        partial(_lookup_cell_starts, mapped=True), in_axes=axes, axis_size=axis_size
    )(sorted_hashes, queries)
    return result, True


_static_cell_starts = jax.jit(_static_cell_starts, inline=True)  # type: ignore[assignment]
