# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared spatial-partitioning helpers for the grid-based colliders.

These helpers centralize the grid-parameter computation, cell hashing dtype,
stencil deduplication, and the prefix-sum packing of per-stencil-cell
neighbor buffers that were previously duplicated across ``cell_list.py``,
``multi_cell_list.py`` and ``neighbor_list.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

#: A per-pair accumulation kernel used by the generic cell traversals:
#: ``pair_fn(acc, i, j, pos, state, valid) -> new_acc`` where ``acc`` is an
#: arbitrary pytree accumulator and ``valid`` is the interaction mask already
#: combining clump/bond exclusions (and, for multi-cell lists, the canonical
#: hash dedup).
PairKernel = Callable[[Any, jax.Array, jax.Array, jax.Array, "State", jax.Array], Any]


def _force_pair_kernel(system: "System") -> PairKernel:
    """Pair kernel accumulating masked ``(force, torque)`` contributions."""

    def pair_fn(
        acc: tuple[jax.Array, jax.Array],
        i: jax.Array,
        j: jax.Array,
        pos: jax.Array,
        state: "State",
        valid: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        f, t = system.force_model.force(i, j, pos, state, system)
        mask = valid if valid.ndim == 0 else valid[:, None]
        return acc[0] + f * mask, acc[1] + t * mask

    return pair_fn


def _force_init(dim: int) -> tuple[jax.Array, jax.Array]:
    """Zero accumulator matching :func:`_force_pair_kernel`."""
    return (
        jnp.zeros(dim, dtype=float),
        jnp.zeros(1 if dim == 2 else 3, dtype=float),
    )


def _energy_pair_kernel(system: "System") -> PairKernel:
    """Pair kernel accumulating half the masked pair potential energy.

    The half compensates for visiting each (i, j) pair from both sides.
    """

    def pair_fn(
        acc: jax.Array,
        i: jax.Array,
        j: jax.Array,
        pos: jax.Array,
        state: "State",
        valid: jax.Array,
    ) -> jax.Array:
        e = system.force_model.energy(i, j, pos, state, system)
        return acc + 0.5 * e * valid

    return pair_fn


def _energy_init() -> jax.Array:
    """Zero accumulator matching :func:`_energy_pair_kernel`."""
    return jnp.asarray(0.0, dtype=float)


def _hash_dtype() -> jnp.dtype:
    """Integer dtype used for cell hashes.

    Cell hashes are products/sums of grid coordinates and strides; for grids
    with more than ``2**31`` cells they overflow int32. We use int64 whenever
    x64 is enabled. When x64 is disabled, hashes are int32 and
    :func:`_grid_params` reports a ``hash_overflow`` flag if the total number
    of grid cells exceeds the representable range.
    """
    return jnp.int64 if getattr(jax.config, "jax_enable_x64", False) else jnp.int32


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
        integer number of cells tiles the box (and the cell size is inflated
        to remain commensurate); non-periodic grids are ceiled.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(grid_dims, grid_strides, cell_size, hash_overflow)`` where
        ``cell_size`` is the effective per-axis-uniform cell size actually
        used by the grid and ``hash_overflow`` is a scalar boolean that is
        True when the total number of grid cells exceeds the range
        representable by the cell-hash dtype (see :func:`_hash_dtype`).
    """
    dtype = _hash_dtype()
    if periodic:
        grid_dims = jnp.floor(box_size / cell_size).astype(dtype)
        # Floor at one cell per axis so boxes smaller than a cell do not
        # produce zero-sized grids (division by zero / empty hashing).
        grid_dims = jnp.maximum(grid_dims, 1)
        cell_size = box_size / grid_dims
    else:
        grid_dims = jnp.ceil(box_size / cell_size).astype(dtype)
        grid_dims = jnp.maximum(grid_dims, 1)

    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=dtype), jnp.cumprod(grid_dims[:-1])]
    )

    # Overflow guard: total cell count must be representable by the hash dtype.
    total_cells = jnp.prod(grid_dims.astype(float))
    hash_overflow = total_cells > float(np.iinfo(dtype).max)

    return grid_dims, grid_strides, cell_size, hash_overflow





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
        ``(neighbor_list, count_overflow)`` where ``neighbor_list`` has shape
        ``(N, max_neighbors)`` (padded with ``-1``) and ``count_overflow`` is
        True if any particle accumulated more than ``max_neighbors``
        neighbors across its stencil cells.
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
