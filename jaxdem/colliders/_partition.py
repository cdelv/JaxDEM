# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared spatial-partitioning helpers for the grid-based colliders.

These helpers centralize grid-parameter computation, cell-hash dtype,
stencil deduplication, and the prefix-sum packing of per-stencil-cell
neighbor buffers that ``cell_list.py``, ``multi_cell_list.py``, and
``neighbor_list.py`` share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


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


@jax.jit(inline=True, static_argnames=("max_neighbors",))
def _pack_stencil_lists(
    all_n_lists: jax.Array,
    all_counts: jax.Array,
    max_neighbors: int,
) -> tuple[jax.Array, jax.Array]:
    """Pack per-stencil-cell neighbor buffers into one row per particle.

    Parameters
    ----------
    all_n_lists : jax.Array
        Per-stencil-cell neighbor buffers, shape ``(N, M, local_capacity)``,
        padded with ``-1``.
    all_counts : jax.Array
        Number of valid entries per stencil cell, shape ``(N, M)``.
    max_neighbors : int
        Static output row width.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(neighbor_list, count_overflow)``. ``neighbor_list`` has shape
        ``(N, max_neighbors)`` and is padded with ``-1``. ``count_overflow``
        is True when any particle has more than ``max_neighbors`` neighbors
        across its stencil cells.
    """
    n_rows = all_n_lists.shape[0]
    local_capacity = all_n_lists.shape[-1]

    # Vectorized prefix-sum packing
    row_offsets = jnp.cumsum(all_counts, axis=-1) - all_counts
    local_iota = jnp.arange(local_capacity)
    target_indices = row_offsets[:, :, None] + local_iota[None, None, :]
    valid_mask = local_iota[None, None, :] < all_counts[:, :, None]

    safe_indices = jnp.where(
        valid_mask.reshape(n_rows, -1),
        target_indices.reshape(n_rows, -1),
        max_neighbors,
    )

    packed = jnp.full((n_rows, max_neighbors), -1, dtype=all_n_lists.dtype)
    row_idx = jnp.arange(n_rows)[:, None]
    packed = packed.at[row_idx, safe_indices].set(
        all_n_lists.reshape(n_rows, -1), mode="drop"
    )

    count_overflow = jnp.any(jnp.sum(all_counts, axis=-1) > max_neighbors)
    return packed, count_overflow
