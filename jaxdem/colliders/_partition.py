# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared spatial-partitioning helpers for the grid-based colliders.

These helpers centralize grid-parameter computation and cell-hash dtype for
``cell_list.py``, ``multi_cell_list.py``, and ``neighbor_list.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

# Bound query-axis vectorization during neighbor-list construction.
# Zero-capacity queries retain vmap: batched lax.map cannot reshape zero-width
# outputs, and those queries have no neighbor buffers to bound.
NEIGHBOR_QUERY_BATCH_SIZE = 65_536

# Bound force/energy traversal workspaces and the longest-lane loop per batch.
PAIR_TRAVERSAL_BATCH_SIZE = 16_384


#: Number of candidate particles each while-loop iteration of the pair
#: traversals visits. A vmapped ``lax.while_loop`` runs until the longest
#: lane finishes. Visiting several candidates reduces loop iterations at
#: the price of at most ``PAIR_UNROLL - 1`` extra masked pair evaluations
#: per lane.
PAIR_UNROLL = 4


@jax.jit
@partial(jax.named_call, name="cell_list._get_spatial_partition")
def _get_spatial_partition(
    pos: jax.Array,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    iota: jax.Array,
) -> tuple[jax.Array, ...]:
    """Compute the spatial hashing and partitioning of the cell list.

    Returns
    -------
    tuple[jax.Array, ...]
        ``(perm, p_cell_hash, neighbor_cell_hashes, hash_overflow)``.
    """
    grid_dims, grid_strides, cell_size, hash_overflow = _grid_params(
        system.domain.box_size, cell_size, system.domain.periodic
    )

    hash_dtype = grid_strides.dtype
    hash_sentinel = jnp.bitwise_not(jnp.asarray(0, hash_dtype))
    pos = system.domain._shift(pos, system)
    if system.domain.periodic:
        p_cell_coords = jnp.floor(
            (((pos - system.domain.anchor) / system.domain.box_size) % 1) * grid_dims
        ).astype(hash_dtype)
    else:
        p_cell_coords = jnp.floor((pos - system.domain.anchor) / cell_size).astype(
            hash_dtype
        )

    p_cell_hash = jnp.dot(p_cell_coords, grid_strides)

    p_cell_hash, perm = jax.lax.sort([p_cell_hash, iota], num_keys=1)

    # Note: we compute neighbor coords using original (unsorted) p_cell_coords
    base_coords = p_cell_coords[:, None, :]
    offsets = neighbor_mask[None, :, :]
    nonnegative = offsets >= 0
    offset_magnitude = jnp.where(nonnegative, offsets, -offsets).astype(hash_dtype)

    def wrapped_offset_coords() -> tuple[jax.Array, jax.Array]:
        """Unsigned modular coordinates plus signed periodic image counts."""
        dims = grid_dims[None, None, :]
        quotient = offset_magnitude // dims
        remainder = offset_magnitude % dims
        positive_carry = base_coords >= (dims - remainder)
        positive_coords = jnp.where(
            (remainder == 0) | ~positive_carry,
            base_coords + remainder,
            base_coords - (dims - remainder),
        )
        negative_borrow = remainder > base_coords
        negative_coords = jnp.where(
            negative_borrow,
            dims - (remainder - base_coords),
            base_coords - remainder,
        )
        coords = jnp.where(nonnegative, positive_coords, negative_coords)
        image = jnp.where(
            nonnegative,
            quotient.astype(offsets.dtype) + positive_carry.astype(offsets.dtype),
            -quotient.astype(offsets.dtype) - negative_borrow.astype(offsets.dtype),
        )
        return coords, image

    if system.domain.periodic:
        neighbor_cell_coords, periodic_image = wrapped_offset_coords()

        neighbor_cell_coords = system.domain.neighbor_cells(
            neighbor_cell_coords, periodic_image, grid_dims, cell_size
        )
        neighbor_cell_hashes = jnp.dot(
            neighbor_cell_coords.astype(hash_dtype), grid_strides
        )
    else:
        positive_oob = offset_magnitude >= grid_dims - base_coords
        negative_oob = offset_magnitude > base_coords
        out_of_bounds = jnp.any(
            jnp.where(nonnegative, positive_oob, negative_oob), axis=-1
        )
        neighbor_cell_coords = jnp.where(
            nonnegative, base_coords + offset_magnitude, base_coords - offset_magnitude
        )
        neighbor_cell_hashes = jnp.dot(neighbor_cell_coords, grid_strides)
        neighbor_cell_hashes = jnp.where(
            out_of_bounds, hash_sentinel, neighbor_cell_hashes
        )

    return (
        perm,
        p_cell_hash,
        neighbor_cell_hashes,
        hash_overflow,
    )


@jax.jit(inline=True)
@partial(jax.named_call, name="cell_list._dedup_stencil_hashes")
def _dedup_stencil_hashes(stencil_hashes: jax.Array) -> jax.Array:
    """Deduplicate one particle's stencil hashes using the unsigned sentinel."""
    indices = jnp.arange(stencil_hashes.size)
    is_duplicate = jnp.zeros(stencil_hashes.shape, dtype=bool)
    # Small static slices fuse with their reductions instead of materializing
    # an (N, stencil, stencil) temporary for a large shear-periodic stencil.
    for start in range(0, stencil_hashes.size, 16):
        end = min(start + 16, stencil_hashes.size)
        mask = (indices[start:end, None] < indices[None, :]) & (
            stencil_hashes[start:end, None] == stencil_hashes[None, :]
        )
        is_duplicate |= jnp.any(mask, axis=0)
    sentinel = jnp.bitwise_not(jnp.asarray(0, stencil_hashes.dtype))
    return jnp.where(is_duplicate, sentinel, stencil_hashes)


def _make_direct_row_body(
    sorted_hashes: jax.Array,
    n_db: int | jax.Array,
    local_capacity: int,
    candidate_valid: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Build one neighbor row directly while walking its stencil in order.

    The returned function carries one ``local_capacity``-wide row across all
    stencil cells. This avoids materializing per-cell rows of shape
    ``(stencil_size, local_capacity)`` before packing.

    Parameters
    ----------
    sorted_hashes : jax.Array
        Cell hashes of the database points, sorted ascending.
    n_db : int | jax.Array
        Number of database points.
    local_capacity : int
        Static size of the output neighbor row.
    candidate_valid : Callable[[jax.Array], jax.Array]
        Boolean predicate evaluated on each candidate database index.
    """

    @jax.jit(inline=True)
    def row_body(
        stencil: jax.Array, cell_starts: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        j_arr = jax.lax.iota(dtype=int, size=PAIR_UNROLL)
        init_carry = (
            jnp.array(0, dtype=int),
            cell_starts[0],
            jnp.array(0, dtype=int),
            jnp.full((local_capacity,), -1, dtype=int),
            jnp.array(False),  # overflow flag
        )

        def cond_fun(
            val: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> bool:
            stencil_idx, _, c, _, _ = val
            in_stencil = stencil_idx < stencil.shape[0]
            has_space = c < local_capacity + 1
            return cast(bool, in_stencil * has_space)

        def body_fun(
            val: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            stencil_idx, k, c, nl, overflow = val
            target_cell_hash = stencil[stencil_idx]
            actual_k_arr = k + j_arr
            safe_k_arr = jnp.minimum(actual_k_arr, jnp.maximum(1, n_db) - 1)
            in_cell_arr = (actual_k_arr < n_db) * (
                sorted_hashes[safe_k_arr] == target_cell_hash
            )
            valid_arr = jax.vmap(candidate_valid)(safe_k_arr) * in_cell_arr

            valid_counts = valid_arr.astype(c.dtype)
            num_valid = jnp.sum(valid_counts)

            cumsum = jnp.cumsum(valid_counts) - valid_counts
            write_idx_arr = c + cumsum
            write_idx_arr = jnp.where(valid_arr, write_idx_arr, local_capacity)

            nl = nl.at[write_idx_arr].set(safe_k_arr, mode="drop")
            c = c + num_valid
            overflow = overflow | (c > local_capacity)

            next_k = k + PAIR_UNROLL
            safe_next_k = jnp.minimum(next_k, jnp.maximum(1, n_db) - 1)
            remains_in_cell = (next_k < n_db) & (
                sorted_hashes[safe_next_k] == target_cell_hash
            )
            next_stencil_idx = jnp.where(remains_in_cell, stencil_idx, stencil_idx + 1)
            safe_stencil_idx = jnp.minimum(next_stencil_idx, stencil.shape[0] - 1)
            next_k = jnp.where(remains_in_cell, next_k, cell_starts[safe_stencil_idx])
            return next_stencil_idx, next_k, c, nl, overflow

        _, _, _, neighbor_row, row_overflow = jax.lax.while_loop(
            cond_fun, body_fun, init_carry
        )
        return neighbor_row, row_overflow

    return row_body


@jax.jit(inline=True)
@partial(jax.named_call, name="multi_cell_list._loose_cell_aabbs")
def _loose_cell_aabbs(
    member_min: jax.Array,
    member_max: jax.Array,
    sorted_hash: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Per-loose-cell axis-aligned bounding box, broadcast to each member.

    The hash-sorted particles of a loose cell form a contiguous run. The
    function therefore computes the cell's expandable AABB (the union of
    every member's box ``[member_min, member_max]``) as a segmented min/max
    reduction over those runs. It returns the result as
    ``(cell_center, cell_half_extent)`` indexed by sorted particle.
    """
    N = member_min.shape[0]
    # Dense, contiguous segment id per sorted particle (0-based cell rank).
    seg_start = jnp.concatenate(
        [jnp.array([True]), sorted_hash[1:] != sorted_hash[:-1]]
    )
    seg_id = jnp.cumsum(seg_start) - 1

    cell_min = jax.ops.segment_min(member_min, seg_id, num_segments=N)[seg_id]
    cell_max = jax.ops.segment_max(member_max, seg_id, num_segments=N)[seg_id]

    cell_center = 0.5 * (cell_min + cell_max)
    cell_half = 0.5 * (cell_max - cell_min)
    return cell_center, cell_half


@jax.jit(inline=True)
def _sorted_cell_run_ends(hashes: jax.Array) -> jax.Array:
    """Return each sorted member's exclusive cell boundary."""
    n = hashes.shape[0]
    if n == 0:
        return jnp.zeros_like(hashes, dtype=int)
    last = jnp.concatenate((hashes[1:] != hashes[:-1], jnp.ones((1,), bool)))
    ends = jnp.where(last, jnp.arange(1, n + 1), n)
    return jax.lax.associative_scan(jnp.minimum, ends, reverse=True)


@jax.jit(inline=True, static_argnames=("quasistatic",))
def _force_pair_fn(
    acc: tuple[jax.Array, jax.Array],
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    valid: jax.Array,
    system: "System",
    quasistatic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    history = system.force_model.init_history(jnp.shape(j), pos.shape[-1])
    f, t, _ = system.force_model.force(
        i,
        j,
        pos,
        state,
        system,
        history,
        advance_history=False,
        quasistatic=quasistatic,
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
    history = system.force_model.init_history(jnp.shape(j), pos.shape[-1])
    e = system.force_model.energy(i, j, pos, state, system, history)
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
        # Derive the scalar from grid_dims so it carries the same manual-axis
        # variance under shard_map. A literal one is replicated whereas the
        # products in the scan can differ between independently sharded worlds.
        (jnp.ones_like(grid_dims[0]), ~dims_valid),
        grid_dims,
    )
    hash_overflow = product_overflow

    return grid_dims, grid_strides, cell_size, hash_overflow


def _capacity_offsets(counts: jax.Array, capacity: int) -> tuple[jax.Array, jax.Array]:
    """Saturate at capacity + 1 so even severe overflows cannot wrap integers."""
    limit = capacity + 1
    totals = jax.lax.associative_scan(
        lambda a, b: a + jnp.minimum(b, limit - a), jnp.minimum(counts, limit)
    )
    overflow = totals[-1] > capacity
    return (
        jnp.concatenate(
            (jnp.zeros(1, dtype=counts.dtype), jnp.minimum(totals, capacity))
        ),
        overflow,
    )
