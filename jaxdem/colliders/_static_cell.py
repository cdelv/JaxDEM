# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared cell traversal with a fixed-capacity pool of tiles.

Cell membership already occupies exactly N slots in the sorted partition. This
module pools traversal work instead: each tile visits up to ``PAIR_UNROLL``
members of one queried cell. Prefix sums replace occupancy-dependent loops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

from ..utils.linalg import cross, norm2
from . import valid_interaction_mask
from ._cell_lookup import _static_cell_starts as _cell_starts
from ._partition import (
    PAIR_UNROLL,
    _can_use_sorted_order,
    _capacity_offsets,
    _dedup_periodic_stencil,
    _effective_cell_size,
    _energy_pair_fn,
    _force_only_pair_fn,
    _force_pair_fn,
    _get_spatial_partition_grid,
    _index_limit,
    _loose_cell_aabbs,
    _neighbor_hashes_from_coords,
    _permute_state,
    _saturated_prefix_sum,
    _shared_max,
    _sorted_cell_run_ends,
    _vec_sum,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

_TILE_BATCH_SIZE = 262_144


class _StaticCellCollider(Protocol):
    """Cell geometry and static traversal settings shared by both colliders."""

    neighbor_mask: jax.Array
    cell_size: jax.Array
    tile_capacity: int
    clump_aware: bool
    _prune_cells: bool


class TileQueue(NamedTuple):
    """Fixed-capacity traversal descriptors and allocation diagnostics.

    ``sources == N_query`` identifies unused slots. ``required_tiles`` is exact when
    the pool fits, and saturates at ``tile_capacity + 1`` on overflow. Overflow
    includes hash overflow. Results from an overflowing traversal are invalid.

    ``run_ends`` has one entry per sorted database particle. A tile gets its
    cell end from the entry at its cursor.
    Optional exclusion metadata maps eligible ranks back to that same order.
    """

    perm: jax.Array
    sources: jax.Array
    cursors: jax.Array
    run_ends: jax.Array
    required_tiles: jax.Array
    overflow: jax.Array
    exclusion_starts: jax.Array | None = None
    clump_ends: jax.Array | None = None
    exclusion_gaps: jax.Array | None = None
    run_starts: jax.Array | None = None


class _ClumpIndex(NamedTuple):
    """Clump-grouped primary ranks, signed gaps, and N + 1 group offsets."""

    ranks: jax.Array
    gaps: jax.Array
    offsets: jax.Array


@jax.jit
def _build_clump_index(perm: jax.Array, clump_id: jax.Array) -> _ClumpIndex:
    """Group primary sorted ranks by clump, with stable order inside each group."""
    n = perm.size
    if n == 0:
        empty = jnp.empty_like(perm)
        return _ClumpIndex(empty, empty, jnp.zeros_like(perm, shape=(1,)))
    labels = clump_id[perm]
    _, ranks = jax.lax.sort(
        (labels, jnp.arange(n, dtype=perm.dtype)), num_keys=1, is_stable=True
    )
    counts = jnp.zeros_like(perm).at[labels].add(1)
    offsets = jnp.concatenate((jnp.zeros_like(perm, shape=(1,)), jnp.cumsum(counts)))
    return _ClumpIndex(ranks, ranks - jnp.arange(n, dtype=ranks.dtype), offsets)


@partial(jax.jit, static_argnames=("right",))
def _bounded_search(
    values: jax.Array,
    target: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    *,
    right: bool,
) -> jax.Array:
    """Search sorted slices with fixed depth and broadcast bounds."""
    target, lo, hi = jnp.broadcast_arrays(target, lo, hi)
    n = values.size
    if n == 0:
        return lo

    def step(
        _: int, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        lower, upper = bounds
        mid = lower + (upper - lower) // 2
        value = values[jnp.minimum(mid, n - 1)]
        open_range = lower < upper
        less = value <= target if right else value < target
        return (
            jnp.where(open_range & less, mid + 1, lower),
            jnp.where(open_range & ~less, mid, upper),
        )

    bounds = step(0, (lo, hi))
    return jax.lax.fori_loop(1, n.bit_length(), step, bounds)[0]


@jax.jit
def _clump_exclusion_ranges(
    index: _ClumpIndex,
    clump_id: jax.Array,
    starts: jax.Array,
    ends: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Find each query clump's auxiliary slice inside each target cell."""
    labels = jnp.broadcast_to(clump_id[..., None], starts.shape)
    if index.ranks.size == 0:
        empty = jnp.zeros_like(starts, dtype=index.offsets.dtype)
        return empty, empty
    lower, upper = index.offsets[labels], index.offsets[labels + 1]
    a = _bounded_search(index.ranks, starts, lower, upper, right=False)
    b = _bounded_search(index.ranks, ends, lower, upper, right=False)
    return a, b


@partial(jax.jit, static_argnames=("return_run_ends", "prune_cells", "sort_queries"))
def _partition_tile_counts(
    pos: jax.Array,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    query_pos: jax.Array | None = None,
    *,
    return_run_ends: bool = False,
    prune_cells: bool = False,
    member_radii: jax.Array | None = None,
    query_radii: Any = 0.0,
    clump_id: jax.Array | None = None,
    sort_queries: bool = False,
) -> tuple[jax.Array, ...]:
    """Return permutation, starts, ends, tile counts, and hash overflow.

    Starts, ends, and counts have shape ``(N_query, stencil_size)``. Missing and
    duplicated periodic cells contribute zero tiles. This helper lets tests
    and benchmarks measure required work without allocating the tile queue.
    ``return_run_ends=True`` adds the compact run ends as the sixth result.

    ``prune_cells=True`` rejects loose-cell boxes before tile allocation.
    ``member_radii`` expands the member boxes. Otherwise they contain centers.
    ``query_radii`` supplies a scalar or one half-width per query.
    Shear-periodic domains skip this rejection.
    """
    n = pos.shape[0]
    if n == 0:
        n_query = n if query_pos is None else query_pos.shape[0]
        empty = jnp.zeros((n_query, neighbor_mask.shape[0]), dtype=int)
        result = jnp.empty((0,), dtype=int), empty, empty, empty, jnp.asarray(False)
        return result + (jnp.empty((0,), dtype=int),) if return_run_ends else result
    (
        perm,
        hashes,
        p_coords,
        grid_dims,
        grid_strides,
        eff_cell_size,
        hash_overflow,
    ) = _get_spatial_partition_grid(pos, system, cell_size, jnp.arange(n))
    if query_pos is not None:
        (
            _,
            _,
            q_coords,
            q_dims,
            q_strides,
            q_cell_size,
            query_overflow,
        ) = _get_spatial_partition_grid(
            query_pos, system, cell_size, jnp.arange(query_pos.shape[0])
        )
        stencil = _neighbor_hashes_from_coords(
            q_coords, q_dims, q_strides, q_cell_size, neighbor_mask, system.domain
        )
        hash_overflow |= query_overflow
        stencil = _dedup_periodic_stencil(stencil, q_dims, neighbor_mask, system.domain)
    else:
        coords_for_stencil = p_coords[perm] if sort_queries else p_coords
        stencil = _neighbor_hashes_from_coords(
            coords_for_stencil,
            grid_dims,
            grid_strides,
            eff_cell_size,
            neighbor_mask,
            system.domain,
        )
        stencil = _dedup_periodic_stencil(
            stencil, grid_dims, neighbor_mask, system.domain
        )
    starts = _cell_starts(hashes, stencil)

    run_ends = _sorted_cell_run_ends(hashes)
    safe_starts = jnp.minimum(starts, n - 1)
    present = (starts < n) & (hashes[safe_starts] == stencil)
    ends = jnp.where(present, run_ends[safe_starts], starts)
    if prune_cells and not hasattr(system.domain, "gamma"):
        members = pos[perm]
        if member_radii is None:
            member_min = member_max = members
        else:
            radii = member_radii[perm, None]
            member_min, member_max = members - radii, members + radii
        centers, half_extents = _loose_cell_aabbs(member_min, member_max, hashes)
        queries = (members if sort_queries else pos) if query_pos is None else query_pos
        dr = system.domain.displacement(
            queries[:, None, :], centers[safe_starts], system
        )
        query_half = jnp.asarray(query_radii)
        if query_half.ndim > 0:
            if sort_queries and query_pos is None:
                query_half = query_half[perm]
            query_half = query_half[:, None, None]
        overlap = jnp.all(
            jnp.abs(dr) <= query_half + half_extents[safe_starts], axis=-1
        )
        # Empty runs allocate no tiles, including after clump exclusion.
        ends = jnp.where(overlap, ends, starts)
    occupancy = ends - starts
    if clump_id is not None:
        index = _build_clump_index(perm, clump_id)
        q_clump = clump_id[perm] if (sort_queries and query_pos is None) else clump_id
        own_starts, own_ends = _clump_exclusion_ranges(index, q_clump, starts, ends)
        occupancy = occupancy - (own_ends - own_starts)
    counts = (occupancy + PAIR_UNROLL - 1) // PAIR_UNROLL
    result = perm, starts, ends, counts, hash_overflow
    return result + (run_ends,) if return_run_ends else result


@partial(jax.jit, static_argnames=("tile_capacity", "prune_cells", "sort_queries"))
def _build_tile_queue(
    pos: jax.Array,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    tile_capacity: int,
    query_pos: jax.Array | None = None,
    *,
    clump_id: jax.Array | None = None,
    prune_cells: bool = False,
    member_radii: jax.Array | None = None,
    query_radii: Any = 0.0,
    sort_queries: bool = False,
) -> TileQueue:
    """Allocate globally pooled four-candidate tiles without occupancy loops."""
    n = pos.shape[0] if query_pos is None else query_pos.shape[0]
    if n == 0 or pos.shape[0] == 0:
        empty = jnp.zeros((tile_capacity,), dtype=int)
        return TileQueue(
            jnp.arange(pos.shape[0]),
            jnp.full((tile_capacity,), n, dtype=int),
            empty,
            jnp.zeros((pos.shape[0],), dtype=int),
            jnp.asarray(0, dtype=int),
            jnp.asarray(False),
        )
    perm, starts, ends, counts, hash_overflow, run_ends = _partition_tile_counts(
        pos,
        system,
        cell_size,
        neighbor_mask,
        query_pos,
        return_run_ends=True,
        prune_cells=prune_cells,
        member_radii=member_radii,
        query_radii=query_radii,
        sort_queries=sort_queries,
    )
    exclusion_starts = clump_ends = exclusion_gaps = run_starts = None
    if clump_id is not None:
        if query_pos is not None or clump_id.shape != (pos.shape[0],):
            raise ValueError("Clump exclusions require matching state particles")
        index = _build_clump_index(perm, clump_id)
        own_starts, own_ends = _clump_exclusion_ranges(index, clump_id, starts, ends)
        eligible = ends - starts - (own_ends - own_starts)
        counts = (eligible + PAIR_UNROLL - 1) // PAIR_UNROLL
        clump_ends = index.offsets[clump_id + 1]
        exclusion_gaps = index.gaps
        members = jnp.arange(pos.shape[0])
        boundaries = jnp.concatenate(
            (jnp.ones((1,), bool), run_ends[1:] != run_ends[:-1])
        )
        run_starts = jax.lax.cummax(jnp.where(boundaries, members, 0), axis=0)
    flat_counts = counts.reshape(-1)
    # Saturating arithmetic prevents even N*N work from wrapping signed indices.
    limit = tile_capacity + 1
    totals = _saturated_prefix_sum(flat_counts, limit)
    required = totals[-1]
    overflow = (required > tile_capacity) | hash_overflow
    if tile_capacity == 0:
        empty = jnp.empty((0,), dtype=int)
        return TileQueue(
            perm,
            empty,
            empty,
            run_ends,
            required,
            overflow,
            empty if clump_id is not None else None,
            clump_ends,
            exclusion_gaps,
            run_starts,
        )

    offsets = jnp.concatenate(
        (jnp.zeros((1,), dtype=int), jnp.minimum(totals, tile_capacity))
    )
    # Nonempty segments have unique starts below capacity. Other writes drop.
    marker_slots = jnp.where(
        (flat_counts > 0) & (offsets[:-1] < tile_capacity),
        offsets[:-1],
        tile_capacity,
    )
    markers = jnp.zeros((tile_capacity,), dtype=int)
    markers = markers.at[marker_slots].set(
        jnp.arange(flat_counts.size), unique_indices=True, mode="drop"
    )
    segments = jax.lax.cummax(markers, axis=0)
    slots = jnp.arange(tile_capacity)
    active = slots < required
    ordinal = jnp.where(active, slots - offsets[segments], 0)
    sources = jnp.where(active, segments // counts.shape[1], n)
    cursors = jnp.where(active, starts.reshape(-1)[segments] + PAIR_UNROLL * ordinal, 0)
    if clump_id is not None:
        exclusion_starts = jnp.where(active, own_starts.reshape(-1)[segments], 0)
    return TileQueue(
        perm,
        sources,
        cursors,
        run_ends,
        required,
        overflow,
        exclusion_starts,
        clump_ends,
        exclusion_gaps,
        run_starts,
    )


def _tile_member_indices(
    queue: TileQueue,
    slots: jax.Array,
    cursors: jax.Array,
    indices: jax.Array,
) -> jax.Array:
    """Decode four original sorted ranks, optionally skipping the source clump."""
    if queue.exclusion_starts is None:
        return indices
    assert queue.run_starts is not None
    assert queue.clump_ends is not None
    assert queue.exclusion_gaps is not None

    n = queue.perm.size
    sources = jnp.minimum(queue.sources[slots], n - 1)
    cell_starts = queue.run_starts[jnp.minimum(cursors, n - 1)][:, None]
    lo = queue.exclusion_starts[slots, None]
    eligible_rank = indices - cell_starts
    excluded_end = _bounded_search(
        queue.exclusion_gaps,
        cell_starts + eligible_rank - lo,
        lo,
        queue.clump_ends[sources, None],
        right=True,
    )
    return cell_starts + eligible_rank + excluded_end - lo


def _decode_tile_batch(
    queue: TileQueue,
    slots: jax.Array,
    n_query: int,
    n_db: int,
    *,
    sorted_neighbors: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decode one batch of pooled tile slots into ``(sources, safe_sources, neighbors, in_cell)``."""
    capacity = queue.sources.size
    safe_slots = jnp.minimum(slots, capacity - 1)
    sources = jnp.where(slots < capacity, queue.sources[safe_slots], n_query)
    safe_sources = jnp.minimum(sources, n_query - 1)
    cursors = queue.cursors[safe_slots]
    ends = queue.run_ends[jnp.minimum(cursors, n_db - 1)]
    indices = _tile_member_indices(
        queue, safe_slots, cursors, cursors[:, None] + jnp.arange(PAIR_UNROLL)
    )
    safe_indices = jnp.minimum(indices, n_db - 1)
    neighbors = safe_indices if sorted_neighbors else queue.perm[safe_indices]
    # `ends` is always <= n_db, so `indices < ends[:, None]` implies `indices < n_db`.
    in_cell = (sources[:, None] < n_query) & (indices < ends[:, None])
    return sources, safe_sources, neighbors, in_cell


def _scan_tile_batches(
    body: Callable[..., Any],
    initial: Any,
    capacity: int,
    required_tiles: jax.Array | None = None,
) -> Any:
    """Run ``body(carry, slots)`` over bounded batches of ``[0, capacity)``."""
    batch_size = min(capacity, _TILE_BATCH_SIZE)
    padded = (capacity + batch_size - 1) // batch_size * batch_size
    batches = jnp.arange(padded).reshape(-1, batch_size)
    num_batches = batches.shape[0]
    if num_batches == 1:
        return body(initial, batches[0])[0]
    # Join manual-axis variance from inputs into the scan carry via a prototype step.
    prototype = body(initial, batches[0])[0]
    initial = jax.tree.map(
        lambda old, new: old + jnp.zeros_like(new), initial, prototype
    )
    if required_tiles is None:
        return jax.lax.scan(body, initial, batches)[0]

    # Skip trailing empty tile batches via a differentiable power-of-2 prefix switch.
    active_batches = _shared_max(
        jnp.minimum((required_tiles + batch_size - 1) // batch_size, num_batches)
    )
    prefixes: list[int] = []
    p = 1
    while p < num_batches:
        prefixes.append(p)
        p = min(2 * p, num_batches)
    prefixes.append(num_batches)
    if len(prefixes) == 1:
        return jax.lax.scan(body, initial, batches)[0]
    branch_idx = sum((active_batches > prefix).astype(int) for prefix in prefixes[:-1])

    def run_prefix(count: int, carry: Any) -> Any:
        if count == 1:
            return body(carry, batches[0])[0]
        return jax.lax.scan(body, carry, batches[:count])[0]

    branches = tuple(partial(run_prefix, count) for count in prefixes)
    return jax.lax.switch(branch_idx, branches, initial)


def _grouped_add(acc: jax.Array, values: jax.Array, sources: jax.Array) -> jax.Array:
    """Accumulate sorted tiles with one write for each group of four equal sources.

    Source indices must be nondecreasing, with unused slots at index ``N``.
    Groups that cross a source boundary retain their individual writes.
    """
    n = acc.shape[0]
    width = 4
    padding = (-sources.shape[0]) % width
    sources = jnp.pad(sources, (0, padding), constant_values=n)
    values = jnp.pad(values, ((0, padding),) + ((0, 0),) * (values.ndim - 1))
    groups = sources.reshape(-1, width)
    grouped = values.reshape((-1, width) + values.shape[1:])
    combine = groups[:, 0] == groups[:, -1]
    broadcast = combine.reshape((-1, 1) + (1,) * (grouped.ndim - 2))
    first = (jnp.arange(width) == 0).reshape((1, width) + (1,) * (grouped.ndim - 2))
    totals = jnp.sum(grouped, axis=1, keepdims=True, dtype=values.dtype)
    grouped = jnp.where(broadcast & first, totals, grouped)
    writes = jnp.where(combine[:, None] & (jnp.arange(width) != 0), n, groups)
    # Dropped indices can occur between valid writes, so indices are not sorted.
    return acc.at[writes.reshape(-1)].add(
        grouped.reshape((-1,) + values.shape[1:]), mode="drop"
    )


def _traverse_tiles(
    state: State,
    system: System,
    queue: TileQueue,
    pair_fn: Callable[..., Any],
    initial: Any,
    *,
    sorted_order: bool = False,
) -> Any:
    """Evaluate fixed, bounded tile batches and accumulate directed results."""
    n, capacity = state.N, queue.sources.size
    totals = jax.tree.map(
        lambda a: jnp.zeros_like(a, shape=(n,) + a.shape, dtype=a.dtype), initial
    )
    if n == 0 or capacity == 0:
        return totals
    if sorted_order:
        pos = state.pos[queue.perm]
        eval_state: Any = _permute_state(state, queue.perm)
    else:
        pos = state.pos
        eval_state = state

    def batch(acc: Any, slots: jax.Array) -> tuple[Any, None]:
        sources, safe_sources, neighbors, in_cell = _decode_tile_batch(
            queue, slots, n, n, sorted_neighbors=sorted_order
        )
        if sorted_order:
            valid = in_cell & valid_interaction_mask(
                eval_state.clump_id[neighbors],
                eval_state.clump_id[safe_sources, None],
                eval_state.bond_id[neighbors],
                eval_state.orig_id[safe_sources, None],
                system.interact_same_bond_id,
            ).astype(bool)
        else:
            valid = in_cell & valid_interaction_mask(
                state.clump_id[neighbors],
                state.clump_id[safe_sources, None],
                state.bond_id[neighbors],
                safe_sources[:, None],
                system.interact_same_bond_id,
            ).astype(bool)
        values = jax.vmap(
            lambda i, j, mask: pair_fn(initial, i, j, pos, eval_state, mask)
        )(safe_sources, neighbors, valid)
        values = jax.tree.map(lambda a: _vec_sum(a, 1), values)
        acc = jax.tree.map(
            lambda a, v: _grouped_add(a, v, sources),
            acc,
            values,
        )
        return acc, None

    result = _scan_tile_batches(batch, totals, capacity, queue.required_tiles)
    if sorted_order:
        result = jax.tree.map(
            lambda a: jnp.empty_like(a).at[queue.perm].set(a, unique_indices=True),
            result,
        )
    return result


@partial(jax.jit, static_argnames=("capacity",))
def _search_neighbor_pairs(
    state: State,
    system: System,
    cutoff: Any,
    capacity: int | None,
    search_radii: jax.Array | None = None,
    skin: Any = 0.0,
) -> tuple[Any, ...]:
    """Count or pack directed Verlet pairs through the pooled tile queue.

    ``capacity=None`` returns exact row counts when ``overflow`` is false.
    Otherwise the return is ``(neighbors, row_offsets, overflow)``, with the
    same source/stencil/member ordering as the original cell-list builder.
    Each bounded tile batch filters candidates once and packs its accepted
    pairs into a shared pool. No full candidate-edge workspace is allocated.

    Tile, hash, or pair-capacity overflow invalidates the result. Counts from
    a truncated tile queue are incomplete and must not be used for sizing.
    """
    n = state.N
    pos = state.pos
    counts = jnp.zeros_like(pos, shape=(n,), dtype=int)
    if n == 0:
        if capacity is None:
            return counts, jnp.asarray(False)
        return (
            jnp.full_like(pos, -1, shape=(capacity,), dtype=int),
            jnp.zeros_like(pos, shape=(1,), dtype=int),
            jnp.asarray(False),
        )
    collider = cast(_StaticCellCollider, system.collider)
    system = system.domain.update_bounds(pos, system, padding=cutoff)
    cell_size = _effective_cell_size(collider, cutoff)

    def run_tile_queue() -> tuple[Any, ...]:
        queue = _build_tile_queue(
            pos,
            system,
            cell_size,
            collider.neighbor_mask,
            collider.tile_capacity,
            clump_id=state.clump_id if collider.clump_aware else None,
            prune_cells=collider._prune_cells,
            query_radii=cutoff,
        )
        tile_capacity = queue.sources.size
        if tile_capacity == 0:
            if capacity is None:
                return counts, queue.overflow
            offsets, pair_overflow = _capacity_offsets(counts, capacity)
            return (
                jnp.full_like(pos, -1, shape=(capacity,), dtype=int),
                offsets,
                queue.overflow | pair_overflow,
            )
        cutoff_sq = cutoff**2

        def candidates(slots: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            sources, safe_sources, neighbors, in_cell = _decode_tile_batch(
                queue, slots, n, n
            )
            dr = system.domain.displacement(
                pos[safe_sources, None, :], pos[neighbors], system
            )
            distance_sq = norm2(dr)
            within_reach = distance_sq <= cutoff_sq
            if search_radii is not None:
                pair_reach = (
                    search_radii[safe_sources, None] + search_radii[neighbors] + skin
                )
                within_reach = within_reach & (distance_sq <= pair_reach**2)
            valid = (
                in_cell
                & within_reach
                & valid_interaction_mask(
                    state.clump_id[neighbors],
                    state.clump_id[safe_sources, None],
                    state.bond_id[neighbors],
                    safe_sources[:, None],
                    system.interact_same_bond_id,
                ).astype(bool)
            )
            return sources, neighbors, valid

        def count_batch(
            cur_counts: jax.Array, slots: jax.Array
        ) -> tuple[jax.Array, None]:
            sources, _, valid = candidates(slots)
            cur_counts = cur_counts.at[sources].add(
                valid.sum(axis=1), mode="drop", indices_are_sorted=True
            )
            return cur_counts, None

        if capacity is None or capacity == 0:
            out_counts = _scan_tile_batches(
                count_batch, counts, tile_capacity, queue.required_tiles
            )
            if capacity is None:
                return out_counts, queue.overflow
            offsets, pair_overflow = _capacity_offsets(out_counts, capacity)
            return (
                jnp.full_like(pos, -1, shape=(capacity,), dtype=int),
                offsets,
                queue.overflow | pair_overflow,
            )

        def fill_batch(carry: Any, slots: jax.Array) -> tuple[Any, None]:
            cur_counts, neighbors, written = carry
            sources, destinations, valid = candidates(slots)
            cur_counts = cur_counts.at[sources].add(
                valid.sum(axis=1), mode="drop", indices_are_sorted=True
            )
            flat_valid = valid.reshape(-1)
            ranks = jnp.cumsum(flat_valid, dtype=cur_counts.dtype) - 1
            remaining = capacity - written
            writes = jnp.where(
                flat_valid & (ranks < remaining),
                written + jnp.minimum(ranks, remaining),
                capacity,
            )
            neighbors = neighbors.at[writes].set(destinations.reshape(-1), mode="drop")
            written = written + jnp.minimum(
                flat_valid.sum(dtype=cur_counts.dtype), remaining
            )
            return (cur_counts, neighbors, written), None

        out_counts, neighbors, _ = _scan_tile_batches(
            fill_batch,
            (
                counts,
                jnp.full_like(pos, -1, shape=(capacity,), dtype=int),
                jnp.zeros_like(pos, shape=(), dtype=int),
            ),
            tile_capacity,
            queue.required_tiles,
        )
        offsets, pair_overflow = _capacity_offsets(out_counts, capacity)
        return neighbors, offsets, queue.overflow | pair_overflow

    if collider.clump_aware:
        return run_tile_queue()

    # Fast dense sorted-grid compacted-stencil builder that preserves exact
    # per-replica static tile-capacity overflow checks without allocating TileQueue.
    from ._neighbor_cache import _dense_sorted_search_pairs

    return _dense_sorted_search_pairs(
        state,
        system,
        cell_size,
        collider.neighbor_mask,
        cutoff,
        capacity,
        search_radii,
        skin,
        prune_cells=collider._prune_cells,
        tile_capacity=collider.tile_capacity,
        fallback_fn=run_tile_queue,
    )


def _query_tile_rows(
    pos_a: jax.Array,
    pos_b: jax.Array,
    system: System,
    cutoff: Any,
    max_neighbors: int,
    state: State | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Keep each row's first neighbors through fixed tile batches."""
    n_a, n_b = pos_a.shape[0], pos_b.shape[0]
    if n_a == 0 or n_b == 0:
        return jnp.full((n_a, max_neighbors), -1, dtype=int), jnp.asarray(False)
    output_capacity = n_a * max_neighbors
    if output_capacity >= _index_limit():
        raise ValueError("neighbor output exceeds the particle-index integer range")
    collider = cast(_StaticCellCollider, system.collider)
    cell_size = _effective_cell_size(collider, cutoff)
    queue = _build_tile_queue(
        pos_b,
        system,
        cell_size,
        collider.neighbor_mask,
        collider.tile_capacity,
        None if state is not None else pos_a,
        clump_id=(
            state.clump_id if state is not None and collider.clump_aware else None
        ),
        prune_cells=collider._prune_cells,
        query_radii=cutoff,
    )
    capacity = queue.sources.size
    counts = jnp.zeros_like(pos_a, shape=(n_a,), dtype=int)
    neighbors = jnp.full_like(pos_a, -1, shape=(output_capacity,), dtype=int)
    if capacity == 0:
        return neighbors.reshape(n_a, max_neighbors), queue.overflow
    cutoff_sq = cutoff**2

    def batch(carry: Any, slots: jax.Array) -> tuple[Any, None]:
        counts, neighbors = carry
        sources, safe_sources, destinations, in_cell = _decode_tile_batch(
            queue, slots, n_a, n_b
        )
        dr = system.domain.displacement(
            pos_a[safe_sources, None, :], pos_b[destinations], system
        )
        valid = in_cell & (norm2(dr) <= cutoff_sq)
        if state is not None:
            valid &= valid_interaction_mask(
                state.clump_id[destinations],
                state.clump_id[safe_sources, None],
                state.bond_id[destinations],
                safe_sources[:, None],
                system.interact_same_bond_id,
            ).astype(bool)
        if max_neighbors > 0:
            flat_sources = jnp.broadcast_to(sources[:, None], valid.shape).reshape(-1)
            flat_valid = valid.reshape(-1)
            prefix = jnp.cumsum(flat_valid, dtype=counts.dtype)
            first = jnp.concatenate(
                (jnp.ones((1,), bool), flat_sources[1:] != flat_sources[:-1])
            )
            row_base = jax.lax.cummax(jnp.where(first, prefix - flat_valid, 0), axis=0)
            # Counts from earlier batches retain rows that cross batch boundaries.
            ranks = counts[jnp.minimum(flat_sources, n_a - 1)] + prefix - row_base - 1
            writes = jnp.where(
                flat_valid & (ranks < max_neighbors),
                flat_sources * max_neighbors + jnp.minimum(ranks, max_neighbors),
                output_capacity,
            )
            neighbors = neighbors.at[writes].set(
                destinations.reshape(-1), unique_indices=True, mode="drop"
            )
        counts = counts.at[sources].add(
            valid.sum(axis=1), mode="drop", indices_are_sorted=True
        )
        return (counts, neighbors), None

    counts, neighbors = _scan_tile_batches(batch, (counts, neighbors), capacity)
    return neighbors.reshape(n_a, max_neighbors), queue.overflow | jnp.any(
        counts > max_neighbors
    )


def _contact_tile_queue(
    state: State, system: System, *, sort_queries: bool = False
) -> tuple[System, TileQueue]:
    """Build the static tile queue for direct force or energy evaluation."""
    collider = cast(_StaticCellCollider, system.collider)
    search_radii = system.force_model.search_radii(state, system)
    search_radius = jnp.max(search_radii, initial=0.0)
    system = system.domain.update_bounds(state.pos, system, padding=search_radius)
    cell_size = _effective_cell_size(collider, 2.0 * search_radius)
    queue = _build_tile_queue(
        state.pos,
        system,
        cell_size,
        collider.neighbor_mask,
        collider.tile_capacity,
        clump_id=state.clump_id if collider.clump_aware else None,
        prune_cells=collider._prune_cells,
        member_radii=search_radii,
        query_radii=search_radii,
        sort_queries=sort_queries,
    )
    return system, queue


@jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
def _static_compute_force(
    state: State,
    system: System,
    *,
    advance_history: bool = True,
    quasistatic: bool = False,
) -> tuple[State, System]:
    """Evaluate directed contact forces over the fixed tile pool."""
    collider = cast(_StaticCellCollider, system.collider)
    use_sorted = _can_use_sorted_order(state, system) and not collider.clump_aware
    has_torque = system.force_model.has_pair_torque

    def run_tile_path(st_cur: State, sys_cur: System) -> tuple[Any, jax.Array]:
        sys_q, queue = _contact_tile_queue(st_cur, sys_cur, sort_queries=use_sorted)
        if not has_torque:
            out = _traverse_tiles(
                st_cur,
                sys_q,
                queue,
                partial(_force_only_pair_fn, system=sys_q, quasistatic=quasistatic),
                jnp.zeros_like(st_cur.force, shape=st_cur.force.shape[1:]),
                sorted_order=use_sorted,
            )
        else:
            out = _traverse_tiles(
                st_cur,
                sys_q,
                queue,
                partial(_force_pair_fn, system=sys_q, quasistatic=quasistatic),
                (
                    jnp.zeros_like(st_cur.force, shape=st_cur.force.shape[1:]),
                    jnp.zeros_like(st_cur.torque, shape=st_cur.torque.shape[1:]),
                ),
                sorted_order=use_sorted,
            )
        return out, queue.overflow

    if use_sorted and state.N > 0:
        from .cell_list import _dense_sorted_traverse

        search_radii = system.force_model.search_radii(state, system)
        search_radius = jnp.max(search_radii, initial=0.0)
        system = system.domain.update_bounds(state.pos, system, padding=search_radius)
        cell_size = _effective_cell_size(collider, 2.0 * search_radius)
        if not has_torque:
            pair_fn: Callable[..., Any] = partial(
                _force_only_pair_fn, system=system, quasistatic=quasistatic
            )
            init_acc: Any = jnp.zeros_like(state.force, shape=state.force.shape[1:])
        else:
            pair_fn = partial(_force_pair_fn, system=system, quasistatic=quasistatic)
            init_acc = (
                jnp.zeros_like(state.force, shape=state.force.shape[1:]),
                jnp.zeros_like(state.torque, shape=state.torque.shape[1:]),
            )
        out, overflow = _dense_sorted_traverse(
            state,
            system,
            cell_size,
            collider.neighbor_mask,
            pair_fn,
            init_acc,
            prune_cells=collider._prune_cells,
            tile_capacity=collider.tile_capacity,
            fallback_fn=run_tile_path,
        )
    else:
        system, queue = _contact_tile_queue(state, system, sort_queries=use_sorted)
        if not has_torque:
            out = _traverse_tiles(
                state,
                system,
                queue,
                partial(_force_only_pair_fn, system=system, quasistatic=quasistatic),
                jnp.zeros_like(state.force, shape=state.force.shape[1:]),
                sorted_order=use_sorted,
            )
        else:
            out = _traverse_tiles(
                state,
                system,
                queue,
                partial(_force_pair_fn, system=system, quasistatic=quasistatic),
                (
                    jnp.zeros_like(state.force, shape=state.force.shape[1:]),
                    jnp.zeros_like(state.torque, shape=state.torque.shape[1:]),
                ),
                sorted_order=use_sorted,
            )
        overflow = queue.overflow

    if not has_torque:
        state.force = out
        state.torque = (
            jnp.zeros_like(state.torque)
            if state.only_spheres
            else cross(state._pos_p_rot, out)
        )
    else:
        force, torque = out
        state.force = force
        state.torque = (
            torque if state.only_spheres else torque + cross(state._pos_p_rot, force)
        )
    system.collider.overflow = overflow
    return state, system


@jax.jit(inline=True)
def _static_compute_potential_energy(
    state: State, system: System
) -> tuple[State, System, jax.Array]:
    """Evaluate half-weighted directed energies over the fixed tile pool."""
    collider = cast(_StaticCellCollider, system.collider)
    use_sorted = _can_use_sorted_order(state, system) and not collider.clump_aware

    def run_tile_path(st_cur: State, sys_cur: System) -> tuple[jax.Array, jax.Array]:
        sys_q, queue = _contact_tile_queue(st_cur, sys_cur, sort_queries=use_sorted)
        e = _traverse_tiles(
            st_cur,
            sys_q,
            queue,
            partial(_energy_pair_fn, system=sys_q),
            jnp.zeros_like(st_cur.pos_c, shape=(), dtype=float),
            sorted_order=use_sorted,
        )
        return e, queue.overflow

    if use_sorted and state.N > 0:
        from .cell_list import _dense_sorted_traverse

        search_radii = system.force_model.search_radii(state, system)
        search_radius = jnp.max(search_radii, initial=0.0)
        system = system.domain.update_bounds(state.pos, system, padding=search_radius)
        cell_size = _effective_cell_size(collider, 2.0 * search_radius)
        energy, overflow = _dense_sorted_traverse(
            state,
            system,
            cell_size,
            collider.neighbor_mask,
            partial(_energy_pair_fn, system=system),
            jnp.zeros_like(state.pos_c, shape=(), dtype=float),
            prune_cells=collider._prune_cells,
            tile_capacity=collider.tile_capacity,
            fallback_fn=run_tile_path,
        )
    else:
        system, queue = _contact_tile_queue(state, system, sort_queries=use_sorted)
        energy = _traverse_tiles(
            state,
            system,
            queue,
            partial(_energy_pair_fn, system=system),
            jnp.zeros_like(state.pos_c, shape=(), dtype=float),
            sorted_order=use_sorted,
        )
        overflow = queue.overflow
    system.collider.overflow = overflow
    return state, system, energy.sum()


@jax.jit(static_argnames=("max_neighbors",), inline=True)
def _static_create_neighbor_list(
    state: State, system: System, cutoff: float, max_neighbors: int
) -> tuple[State, System, jax.Array, jax.Array]:
    """Return the first neighbors per row through the fixed tile pool."""
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if state.N == 0:
        return state, system, jnp.empty((0, max_neighbors), int), jnp.asarray(False)
    system = system.domain.update_bounds(state.pos, system, padding=cutoff)
    neighbors, overflow = _query_tile_rows(
        state.pos, state.pos, system, cutoff, max_neighbors, state
    )
    return state, system, neighbors, overflow


@jax.jit(static_argnames=("max_neighbors",), inline=True)
def _static_create_cross_neighbor_list(
    pos_a: jax.Array,
    pos_b: jax.Array,
    system: System,
    cutoff: float,
    max_neighbors: int,
) -> tuple[jax.Array, jax.Array]:
    """Query database positions with fixed tiles and independent row limits."""
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if pos_a.shape[0] == 0 or pos_b.shape[0] == 0:
        return jnp.full((pos_a.shape[0], max_neighbors), -1, int), jnp.asarray(False)
    system = system.domain.update_bounds(
        jnp.concatenate((pos_a, pos_b), axis=0), system, padding=cutoff
    )
    return _query_tile_rows(pos_a, pos_b, system, cutoff, max_neighbors)
