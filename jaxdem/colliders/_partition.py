# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared spatial-partitioning helpers for the grid-based colliders.

These helpers centralize grid-parameter computation and cell-hash dtype for
``cell_list.py``, ``multi_cell_list.py``, and ``neighbor_list.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

# Bound query-axis vectorization during neighbor-list construction.
# Zero-capacity queries retain vmap: batched lax.map cannot reshape zero-width
# outputs, and those queries have no neighbor buffers to bound.
NEIGHBOR_QUERY_BATCH_SIZE = 65_536

# Bound force/energy traversal workspaces and the longest-lane loop per batch.
PAIR_TRAVERSAL_BATCH_SIZE = 16_384


@jax.jit(inline=True)
def _cell_starts(sorted_hashes: jax.Array, queries: jax.Array) -> jax.Array:
    """Return the first sorted index for each queried cell hash.

    Dense, modest hash ranges use a temporary direct table. Sparse ranges use
    ``searchsorted`` instead. Missing hashes return the
    insertion point in the sparse branch and ``N`` in the dense branch; grid
    traversals verify the hash at that candidate before accepting the cell.
    """
    n_items = sorted_hashes.shape[0]
    if n_items == 0:
        return jnp.zeros(queries.shape, dtype=jnp.int32)

    capacity = 8 * n_items
    if capacity > 2**31 - 1:
        return jnp.searchsorted(
            sorted_hashes, queries, side="left", method="scan_unrolled"
        )

    hash_capacity = jnp.asarray(capacity, dtype=sorted_hashes.dtype)
    use_dense = sorted_hashes[-1] < hash_capacity

    def dense_lookup() -> jax.Array:
        table = jnp.full((capacity,), n_items, dtype=jnp.int32)
        stored_in_range = sorted_hashes < hash_capacity
        stored_indices = jnp.where(
            stored_in_range, sorted_hashes, hash_capacity
        ).astype(jnp.int32)
        table = table.at[stored_indices].min(
            jax.lax.iota(jnp.int32, n_items), mode="drop"
        )

        query_in_range = queries < hash_capacity
        query_indices = jnp.where(query_in_range, queries, hash_capacity).astype(
            jnp.int32
        )
        return table.at[query_indices].get(mode="fill", fill_value=n_items)

    def sparse_lookup() -> jax.Array:
        return jnp.searchsorted(
            sorted_hashes, queries, side="left", method="scan_unrolled"
        )

    return jax.lax.cond(use_dense, dense_lookup, sparse_lookup)


@jax.jit(inline=True)
def _force_pair_fn(
    acc: tuple[jax.Array, jax.Array],
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    valid: jax.Array,
    system: "System",
) -> tuple[jax.Array, jax.Array]:
    history = system.force_model.init_history(jnp.shape(j), pos.shape[-1])
    f, t, _ = system.force_model.force(
        i, j, pos, state, system, history, advance_history=False
    )
    f = jnp.where((valid > 0)[..., None], f, 0.0)
    t = jnp.where((valid > 0)[..., None], t, 0.0)
    return acc[0] + f, acc[1] + t


@jax.jit(inline=True)
def _energy_pair_fn(
    acc: jax.Array,
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    valid: jax.Array,
    system: "System",
) -> jax.Array:
    e = system.force_model.energy(i, j, pos, state, system)
    e = jnp.where(valid > 0, e, 0.0)
    return acc + 0.5 * e


@jax.jit(inline=True, static_argnames=("periodic",))
def _grid_params(
    box_size: jax.Array, cell_size: jax.Array, periodic: bool
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute grid dimensions, hashing strides, and the effective cell size.

    Parameters
    ----------
    box_size : jax.Array
        Physical extents of the domain per axis.
    cell_size : jax.Array
        Requested (scalar) grid cell size.
    periodic : bool
        Whether the domain is periodic. Periodic grids are floored so an
        integer number of cells tiles the box, and the cell size is inflated
        to stay commensurate. Non-periodic grids are ceiled.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(grid_dims, grid_strides, cell_size, hash_overflow)`` where
        ``cell_size`` is the effective per-axis-uniform cell size of the grid
        and ``hash_overflow`` is a scalar boolean. ``hash_overflow`` is True
        when the total number of grid cells exceeds the range of the
        cell-hash dtype.
    """
    # Cell hashes are unsigned so the full native JAX integer range is
    # available. Particle indices remain signed because -1 is their sentinel.
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    hash_bits = 64 if jax.config.jax_enable_x64 else 32
    hash_limit = 1 << hash_bits
    max_hash = jnp.asarray(hash_limit - 1, dtype=hash_dtype)
    dims_float = (
        jnp.floor(box_size / cell_size) if periodic else jnp.ceil(box_size / cell_size)
    )
    dims_float = jnp.maximum(dims_float, 1)
    dims_valid = jnp.all(jnp.isfinite(dims_float)) & jnp.all(
        dims_float < jnp.asarray(float(hash_limit), dtype=box_size.dtype)
    )
    safe_dims_float = jnp.where(dims_valid, dims_float, 1)
    grid_dims = safe_dims_float.astype(hash_dtype)
    if periodic:
        cell_size = box_size / grid_dims

    def step(
        carry: tuple[jax.Array, jax.Array], dim: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        product, overflow = carry
        can_multiply = dim <= max_hash // product
        # Select a safe factor before multiplication. ``where(product * dim)``
        # would still evaluate an overflowing multiply on accelerator backends.
        safe_dim = jnp.where(can_multiply, dim, jnp.asarray(1, dim.dtype))
        next_product = product * safe_dim
        return (next_product, overflow | ~can_multiply), product

    (_, product_overflow), grid_strides = jax.lax.scan(
        step,
        (jnp.asarray(1, hash_dtype), ~dims_valid),
        grid_dims,
    )
    hash_overflow = product_overflow

    return grid_dims, grid_strides, cell_size, hash_overflow
