# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Fixed-capacity packed directed pairs for the Verlet cache.

Two search passes count neighbors and fill a shared pool without a per-particle
capacity. The static cell path instead filters and packs fixed tile batches
in one pass. CSR offsets identify each row. The static cell path evaluates
fixed pair batches. Other backends evaluate occupied CSR rows in bounded
batches. Small pools use a single flat pass to reduce launch overhead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from ..utils.linalg import cross, norm, norm2
from . import valid_interaction_mask
from ._cell_lookup import _cell_starts
from ._partition import (
    _build_dense_cell_table,
    _capacity_offsets,
    _compacted_stencil_decode,
    _dedup_periodic_stencil,
    _dense_grid_stats,
    _dense_stencil_prefix_batch,
    _effective_cell_size,
    _get_spatial_partition_grid,
    _loose_cell_aabbs,
    _neighbor_hashes_from_coords,
    _prepare_dense_grid,
    _shared_max,
    _sorted_cell_run_ends,
    _vec_sum,
    shared_bounded_loop,
)

_SEARCH_BATCH_SIZE = 65_536
# Large row batches minimize launch/sync overhead while bounding workspace at huge N.
_DEFAULT_ROW_BATCH_SIZE = 131_072
_ROW_BATCH_SIZE = {2: _DEFAULT_ROW_BATCH_SIZE, 3: _DEFAULT_ROW_BATCH_SIZE}
# Bound the force-law working set independently of the allocated pair pool.
_FIXED_PAIR_BATCH_SIZE = 262_144
_HISTORY_PAIR_BATCH_SIZE = 32_768
_HISTORY_PREFIX_SIZE = 4_096


@custom_vmap
def _shared_any(required: Any) -> Any:
    return required


@_shared_any.def_vmap
def _shared_any_mapped(axis_size: int, in_batched: Any, required: Any) -> Any:
    return _shared_any(jnp.any(required)), False


def _with_axes(current: Any, prototype: Any) -> Any:
    """Join manual-axis variance without changing values or executing the prototype."""
    return jax.tree.map(
        lambda value, proposed: jax.lax.select(jnp.asarray(True), value, proposed),
        current,
        prototype,
    )


def _prefix_capacities(
    capacity: int, minimum: int = _FIXED_PAIR_BATCH_SIZE
) -> tuple[int, ...]:
    """List fixed pair prefixes in doubling steps."""
    prefixes = []
    prefix = min(capacity, minimum)
    while prefix < capacity:
        prefixes.append(prefix)
        prefix = min(2 * prefix, capacity)
    return (*prefixes, capacity)


def _prefix_dispatch(
    evaluate: Any,
    minimum: int = _FIXED_PAIR_BATCH_SIZE,
    scalar_limit: int | None = None,
) -> Any:
    """Select one fixed prefix after the maximum over explicit batch axes."""

    @custom_vmap
    def dispatch(state: Any, system: Any) -> Any:
        capacity = system.collider.neighbor_list.shape[-1]
        if scalar_limit is not None and capacity <= scalar_limit:
            return evaluate(capacity, state, system)
        capacities = _prefix_capacities(capacity, minimum)
        if len(capacities) == 1:
            return evaluate(capacities[0], state, system)
        used = jnp.max(system.collider.row_offsets[..., -1], initial=0)
        index = sum(used > prefix for prefix in capacities[:-1])
        branches = tuple(partial(evaluate, prefix) for prefix in capacities)
        return jax.lax.switch(index, branches, state, system)

    @dispatch.def_vmap
    def mapped(axis_size: int, in_batched: Any, state: Any, system: Any) -> Any:
        axes = jax.tree.map(lambda batched: 0 if batched else None, in_batched)

        def evaluate_batch(prefix: int, states: Any, systems: Any) -> Any:
            return jax.vmap(
                partial(evaluate, prefix), in_axes=tuple(axes), axis_size=axis_size
            )(states, systems)

        # A fresh dispatcher lets an outer vmap collect another explicit axis.
        # Only the outermost call selects a prefix for the whole current batch.
        output = _prefix_dispatch(evaluate_batch, minimum)(state, system)
        return output, jax.tree.map(lambda _: True, output)

    return dispatch


def _uses_static_cache(col: Any) -> bool:
    """Select fixed traversal for the static cell backend."""
    return _uses_static_tiles(col.secondary_collider)


def _uses_static_tiles(collider: Any) -> bool:
    """Select static traversal when the cell backend has a tile budget."""
    return getattr(collider, "tile_capacity", None) is not None


def _uses_fixed_cache(col: Any, dim: int) -> bool:
    """Use fixed blocks for static backends and small history caches."""
    return _uses_static_cache(col) or (
        col.history.size > 0 and col.neighbor_list.size < 2 * dim * _ROW_BATCH_SIZE[dim]
    )


def _uses_flat_cache(col: Any, dim: int) -> bool:
    """Return the original flat traversal policy for output dtype compatibility."""
    capacity = col.neighbor_list.size
    if _uses_static_cache(col):
        return capacity <= _FIXED_PAIR_BATCH_SIZE
    return capacity < 2 * dim * _ROW_BATCH_SIZE[dim]


def _same_manual_axes(left: Any, right: Any) -> bool:
    """Keep independent force and torque axes when their reductions differ."""
    left_type, right_type = jax.typeof(left), jax.typeof(right)
    return getattr(
        left_type, "manual_axis_type", getattr(left_type, "vma", None)
    ) == getattr(right_type, "manual_axis_type", getattr(right_type, "vma", None))


def _fixed_pool_reduce(
    col: Any,
    initial: Any,
    evaluate: Any,
    history: Any = None,
    prefix_capacity: int | None = None,
    preserve_dtype: bool = True,
    join_forces: bool = False,
) -> Any:
    """Visit a fixed pair prefix in batches with fixed loop bounds.

    CSR rows can hold any fraction of the pool. The selected prefix covers all
    occupied slots without an occupancy bound for individual rows.
    The pair law only needs a fixed batch of temporary arrays at each step.
    """
    n = col.row_offsets.size - 1
    capacity = col.neighbor_list.size if prefix_capacity is None else prefix_capacity
    totals = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), initial)
    if n == 0 or capacity == 0:
        return totals, history
    sources = pair_sources(replace(col, neighbor_list=col.neighbor_list[:capacity]))
    width = min(
        capacity,
        _FIXED_PAIR_BATCH_SIZE if history is None else _HISTORY_PAIR_BATCH_SIZE,
    )
    full = capacity // width * width

    def batch(totals: Any, slots: Any, cast_totals: bool = False) -> Any:
        src = sources[slots]
        values, updated = evaluate(
            jnp.minimum(src, n - 1), col.neighbor_list[slots], slots
        )
        if cast_totals:
            totals = jax.tree.map(
                lambda total, value: total.astype(value.dtype), totals, values
            )
        # Padding has source N and drops outside the output. CSR source IDs
        # retain the same directed-pair order as the existing cache.
        if (
            join_forces
            and totals[0].dtype == totals[1].dtype
            and values[0].dtype == values[1].dtype
            and _same_manual_axes(totals[0], totals[1])
            and _same_manual_axes(values[0], values[1])
        ):
            force_width = totals[0].shape[-1]
            joined = (
                jnp.concatenate(totals, axis=-1)
                .at[src]
                .add(jnp.concatenate(values, axis=-1), mode="drop")
            )
            totals = (joined[..., :force_width], joined[..., force_width:])
        else:
            totals = jax.tree.map(
                lambda total, value: total.at[src].add(value, mode="drop"),
                totals,
                values,
            )
        return totals, updated

    # The first batch gives each carry the force law's manual-axis variance.
    # This also covers a shared state with different histories per replica.
    totals, first = batch(totals, jnp.arange(width), not preserve_dtype)
    if history is None:
        totals = jax.lax.fori_loop(
            1,
            full // width,
            lambda index, carry: batch(carry, index * width + jnp.arange(width))[0],
            totals,
        )
        if full < capacity:
            totals, _ = batch(totals, jnp.arange(full, capacity))
        return totals, None

    history_dtype = history.dtype if preserve_dtype else first.dtype

    def step(totals: Any, index: Any) -> Any:
        totals, updated = batch(totals, index * width + jnp.arange(width))
        return totals, updated.astype(history_dtype)

    # Each block writes disjoint slots. Carry only force totals through the loop.
    totals, following = jax.lax.scan(step, totals, jnp.arange(1, full // width))
    chunks = [
        first.astype(history_dtype),
        following.reshape((full - width,) + history.shape[1:]),
    ]
    if full < capacity:
        totals, tail = batch(totals, jnp.arange(full, capacity))
        chunks.append(tail.astype(history_dtype))
    if capacity < history.shape[0]:
        chunks.append(history[capacity:].astype(history_dtype))
    return totals, jnp.concatenate(chunks, axis=0)


def pair_sources(col: Any) -> jax.Array:
    """Decode CSR source IDs for operations that need a flat edge list."""
    capacity = col.neighbor_list.size
    n = col.row_offsets.shape[0] - 1
    if capacity == 0:
        return jnp.empty((0,), dtype=int)
    # Duplicate starts belong to empty rows; the last row at that offset wins.
    starts = (
        jnp.zeros(capacity + 1, dtype=int).at[col.row_offsets[:-1]].max(jnp.arange(n))
    )
    sources = jax.lax.cummax(starts, axis=0)[:-1]
    return jnp.where(jnp.arange(capacity) < col.row_offsets[-1], sources, n)


def build_pairs(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int,
    *,
    search_radii: Any | None = None,
    skin: Any = 0.0,
) -> tuple[Any, ...]:
    """Build pairs within a scalar search bound and optional per-pair reaches."""
    from ._static_cell import _search_neighbor_pairs

    if _uses_static_tiles(system.collider):
        return _search_neighbor_pairs(
            state, system, cutoff, capacity, search_radii, skin
        )
    return _search_pairs(state, system, cutoff, capacity, search_radii, skin)


@jax.jit
def count_pairs(
    state: Any,
    system: Any,
    cutoff: Any,
    *,
    search_radii: Any | None = None,
    skin: Any = 0.0,
) -> tuple[Any, Any]:
    """Return directed row counts and search overflow without allocating pairs."""
    from ._static_cell import _search_neighbor_pairs

    if _uses_static_tiles(system.collider):
        counts, overflow = _search_neighbor_pairs(
            state, system, cutoff, None, search_radii, skin
        )
        return counts, overflow
    counts, overflow = _search_pairs(state, system, cutoff, None, search_radii, skin)
    return counts, overflow


def _search_pair_candidates(
    data: Any, rows: Any, cursor: Any, end: Any, *, need_dst: bool = True
) -> tuple[Any, Any]:
    (
        pos,
        pos_s,
        perm,
        clumps,
        clumps_s,
        bonds_s,
        system,
        cutoff_sq,
        search_radii,
        search_radii_s,
        skin,
    ) = data
    n = perm.shape[0]
    indices = cursor[..., None] + jnp.arange(4)
    safe = jnp.minimum(indices, n - 1)
    src = jnp.minimum(rows, n - 1)
    valid = (rows[:, None, None] < n) & (indices < end[..., None])
    delta = system.domain.displacement(pos[src, None, None, :], pos_s[safe], system)
    distance_sq = norm2(delta)
    valid &= distance_sq <= cutoff_sq
    if search_radii is not None:
        reach = search_radii[src, None, None] + search_radii_s[safe] + skin
        valid &= distance_sq <= reach**2
    valid &= valid_interaction_mask(
        clumps_s[safe],
        clumps[src, None, None],
        bonds_s[safe],
        src[:, None, None],
        system.interact_same_bond_id,
    ).astype(bool)
    dst = perm[safe] if need_dst else safe
    return dst, valid


def _count_search_step(index: Any, counts: Any, operands: Any) -> Any:
    data, rows, begin, end = operands
    _, valid = _search_pair_candidates(
        data, rows, begin + 4 * index, end, need_dst=False
    )
    return counts + valid.sum(axis=-1)


def _fill_search_step(index: Any, carry: Any, operands: Any) -> Any:
    data, rows, begin, end, slot_ends = operands
    slots, neighbors = carry
    dst, valid = _search_pair_candidates(
        data, rows, begin + 4 * index, end, need_dst=True
    )
    ranks = jnp.cumsum(valid, axis=-1) - 1
    remaining = slot_ends - slots
    writes = slots[..., None] + jnp.minimum(ranks, remaining[..., None])
    capacity = neighbors.shape[0]
    writes = jnp.where(valid & (ranks < remaining[..., None]), writes, capacity)
    neighbors = neighbors.at[writes].set(dst, mode="drop")
    return slots + jnp.minimum(valid.sum(axis=-1), remaining), neighbors


_count_search_loop = shared_bounded_loop(_count_search_step, first_step=True)
_fill_search_loop = shared_bounded_loop(_fill_search_step)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _SearchState:
    """Particle data read by the discrete neighbor search."""

    pos: Any
    clump_id: Any
    bond_id: Any
    only_spheres: bool = jax.tree.static(default=False)

    @property
    def N(self) -> int:
        return int(self.pos.shape[0])


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _SearchCells:
    """Cell geometry read by the discrete neighbor search."""

    cell_size: Any
    neighbor_mask: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _SearchSystem:
    """Domain, cell geometry, and exclusions read by the neighbor search."""

    domain: Any
    collider: Any
    interact_same_bond_id: Any


def _compact_search_inputs(state: Any, system: Any) -> tuple[Any, Any] | None:
    """Keep built-in search dependencies; preserve custom-domain system access."""
    from ..domains.free import FreeDomain
    from ..domains.lees_edwards import LeesEdwardsDomain
    from ..domains.periodic import PeriodicDomain
    from ..domains.reflect import ReflectDomain
    from ..domains.reflect_sphere import ReflectSphereDomain
    from .naive import NaiveSimulator

    domain = system.domain
    if type(domain) not in (
        FreeDomain,
        PeriodicDomain,
        LeesEdwardsDomain,
        ReflectDomain,
        ReflectSphereDomain,
    ):
        return None
    collider = system.collider
    if hasattr(collider, "neighbor_mask"):
        collider = _SearchCells(collider.cell_size, collider.neighbor_mask)
    elif type(collider) is NaiveSimulator:
        collider = replace(
            collider, history=jnp.zeros((0, 0, 0)), overflow=jnp.asarray(False)
        )
    else:
        return None

    # Keep the built-in domain methods. Search does not read rate or cache
    # fields, so local constants avoid retaining their input array arguments.
    fields = {"inv_box_size": jnp.zeros(domain.box_size.shape, domain.box_size.dtype)}
    if type(collider) is NaiveSimulator:
        fields["anchor"] = jnp.zeros(domain.anchor.shape, domain.anchor.dtype)
        if not domain.periodic:
            fields["box_size"] = jnp.ones(domain.box_size.shape, domain.box_size.dtype)
    if type(domain) is FreeDomain:
        fields.update(
            box_size=jnp.ones(domain.box_size.shape, domain.box_size.dtype),
            anchor=jnp.zeros(domain.anchor.shape, domain.anchor.dtype),
        )
    if type(domain) in (PeriodicDomain, LeesEdwardsDomain):
        fields["box_rate"] = jnp.zeros(domain.box_rate.shape, domain.box_rate.dtype)
    if type(domain) is LeesEdwardsDomain:
        for name in ("gamma_dot", "alpha_axis", "beta_axis"):
            value = getattr(domain, name)
            fields[name] = jnp.zeros(value.shape, value.dtype)
    if type(domain) in (ReflectDomain, ReflectSphereDomain):
        value = domain.restitution_coefficient
        fields["restitution_coefficient"] = jnp.zeros(value.shape, value.dtype)
    return (
        _SearchState(
            state.pos,
            state.clump_id,
            state.bond_id,
            only_spheres=bool(getattr(state, "only_spheres", False)),
        ),
        _SearchSystem(
            replace(domain, **fields), collider, system.interact_same_bond_id
        ),
    )


def _search_pairs(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
) -> tuple[Any, ...]:
    """Keep scalar search and share traversal for explicit environment batches."""
    compact = _compact_search_inputs(state, system)
    if compact is None:
        # Custom domains can read any full-system field through their methods.
        return _search_pairs_scalar(state, system, cutoff, capacity, search_radii, skin)
    state, system = compact
    # Search outputs are discrete. The later force evaluation reads the original
    # differentiable state and system, outside this custom batching boundary.
    state, system, cutoff, search_radii, skin = jax.tree.map(
        jax.lax.stop_gradient, (state, system, cutoff, search_radii, skin)
    )

    @custom_vmap
    def dispatch(
        state: Any, system: Any, cutoff: Any, search_radii: Any, skin: Any
    ) -> Any:
        return _search_pairs_scalar(state, system, cutoff, capacity, search_radii, skin)

    @dispatch.def_vmap
    def mapped(
        axis_size: int,
        in_batched: Any,
        state: Any,
        system: Any,
        cutoff: Any,
        search_radii: Any,
        skin: Any,
    ) -> Any:
        axes = jax.tree.map(lambda batched: 0 if batched else None, in_batched)
        result = jax.vmap(
            lambda state, system, cutoff, search_radii, skin: _search_pairs_batched(
                state, system, cutoff, capacity, search_radii, skin
            ),
            in_axes=tuple(axes),
            axis_size=axis_size,
        )(state, system, cutoff, search_radii, skin)
        return result, jax.tree.map(lambda _: True, result)

    return dispatch(state, system, cutoff, search_radii, skin)


def _prepare_search_partition(
    state: Any, system: Any, cutoff: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Return ``(system, perm, starts, ends, hash_overflow)`` for cell or naive search."""
    n = state.N
    collider = system.collider
    rows = jnp.arange(n)
    pos = state.pos
    if hasattr(collider, "neighbor_mask"):
        system = system.domain.update_bounds(pos, system, padding=cutoff)
        cell_size = _effective_cell_size(collider, cutoff)
        (
            perm,
            hashes,
            p_coords,
            grid_dims,
            grid_strides,
            eff_cell_size,
            hash_overflow,
        ) = _get_spatial_partition_grid(pos, system, cell_size, rows)
        stencil = _neighbor_hashes_from_coords(
            p_coords,
            grid_dims,
            grid_strides,
            eff_cell_size,
            collider.neighbor_mask,
            system.domain,
        )
        stencil = _dedup_periodic_stencil(
            stencil, grid_dims, collider.neighbor_mask, system.domain
        )
        starts = _cell_starts(hashes, stencil)
        safe = jnp.minimum(starts, n - 1)
        present = (starts < n) & (hashes[safe] == stencil)
        ends = jnp.where(present, _sorted_cell_run_ends(hashes)[safe], starts)
    else:
        # The all-pairs backend uses one cell, retaining O(N^2) search cost.
        from .naive import NaiveSimulator

        if not isinstance(collider, NaiveSimulator):
            raise ValueError("NeighborList requires CellList, MultiCellList, or naive")
        perm = rows
        starts = jnp.zeros((n, 1), dtype=int)
        ends = jnp.full((n, 1), n, dtype=int)
        hash_overflow = jnp.asarray(False)
    return system, perm, starts, ends, hash_overflow


def _search_pairs_naive(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
) -> tuple[Any, ...]:
    """Chunked all-pairs neighbor-list builder for the naive secondary backend."""
    n = state.N
    pos = state.pos
    cutoff_sq = cutoff**2
    cols = jnp.arange(n)
    row_chunk = min(n, 256)
    padded_n = (n + row_chunk - 1) // row_chunk * row_chunk
    batches = jnp.arange(padded_n).reshape(-1, row_chunk)

    def pair_mask(rows: Any) -> Any:
        safe_rows = jnp.minimum(rows, n - 1)
        delta = system.domain.displacement(
            pos[safe_rows, None, :], pos[None, :, :], system
        )
        dist_sq = norm2(delta)
        valid = (rows[:, None] < n) & (dist_sq <= cutoff_sq)
        if search_radii is not None:
            reach = search_radii[safe_rows, None] + search_radii[None, :] + skin
            valid &= dist_sq <= reach**2
        valid &= valid_interaction_mask(
            state.clump_id[None, :],
            state.clump_id[safe_rows, None],
            state.bond_id[None, :, :],
            safe_rows[:, None],
            system.interact_same_bond_id,
        ).astype(bool)
        return valid

    counts = jax.lax.map(lambda rows: pair_mask(rows).sum(axis=1), batches).reshape(-1)[
        :n
    ]
    if capacity is None:
        return counts, jnp.asarray(False)
    offsets, overflow = _capacity_offsets(counts, capacity)
    neighbors = jnp.full_like(counts, -1, shape=(capacity,), dtype=int)
    if capacity == 0:
        return neighbors, offsets, overflow

    def fill_chunk(neighbors: Any, rows: Any) -> tuple[Any, None]:
        safe_rows = jnp.minimum(rows, n - 1)
        valid = pair_mask(rows)
        base = offsets[safe_rows]
        budget = jnp.where(rows < n, offsets[safe_rows + 1] - base, 0)
        ranks = jnp.cumsum(valid, axis=1) - 1
        writes = jnp.where(
            valid & (ranks < budget[:, None]),
            base[:, None] + jnp.minimum(ranks, budget[:, None]),
            capacity,
        )
        dst = jnp.broadcast_to(cols[None, :], valid.shape)
        neighbors = neighbors.at[writes].set(dst, mode="drop")
        return neighbors, None

    neighbors, _ = jax.lax.scan(fill_chunk, neighbors, batches)
    return neighbors, offsets, overflow


def _csr_slot_intervals(
    offsets: Any, counts: Any, rows: Any, safe_rows: Any, n: int
) -> tuple[Any, Any]:
    """Disjoint per-cell write intervals ``(slots, slot_ends)`` in a particle's pooled row."""
    base = offsets[safe_rows, None]
    budget = jnp.where(rows < n, offsets[safe_rows + 1] - offsets[safe_rows], 0)
    slot_ends = base + jnp.minimum(counts[safe_rows].cumsum(axis=1), budget[:, None])
    slots = jnp.concatenate((base, slot_ends[:, :-1]), axis=1)
    return slots, slot_ends


def _dense_sorted_search_pairs(
    state: Any,
    system: Any,
    cell_size: Any,
    neighbor_mask: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
    *,
    prune_cells: bool = False,
    tile_capacity: int | None = None,
    fallback_fn: Any,
) -> tuple[Any, ...]:
    """Fast 2-pass dense sorted-grid outer-scan neighbor-list builder.

    Avoids ``(N, M)`` stencil/start/end buffers, binary search, ``while_loop``
    host syncs, and atomic scatter-adds. Falls back cleanly to ``fallback_fn``
    when the grid is sparse, wraps onto itself, or has ``max_occ > 8``.
    """
    n = state.N
    pos = state.pos
    iota = jax.lax.iota(dtype=int, size=n)
    meta, use_dense = _prepare_dense_grid(
        pos, system, cell_size, neighbor_mask, iota, prune_cells=prune_cells
    )
    perm = meta.perm
    p_cell_hash = meta.p_cell_hash
    p_cell_coords = meta.p_cell_coords
    grid_dims = meta.grid_dims
    grid_strides = meta.grid_strides
    eff_cell_size = meta.eff_cell_size
    hash_overflow = meta.hash_overflow
    cap_dense = meta.cap_dense
    periodic = meta.periodic
    has_shear = meta.has_shear
    do_prune = meta.do_prune
    shear_r = meta.shear_r
    use_compact_shear = meta.use_compact_shear

    def run_dense(cap: int) -> tuple[Any, ...]:
        pos_s = pos[perm]
        coords_s = p_cell_coords[perm].astype(int)
        clumps_s = state.clump_id[perm]
        bonds_s = state.bond_id[perm]
        search_radii_s = None if search_radii is None else search_radii[perm]
        dims_i = grid_dims.astype(int)
        strides_i = grid_strides.astype(int)
        offsets_mask = neighbor_mask.astype(int)
        cutoff_sq = cutoff**2
        interact_same = system.interact_same_bond_id

        cell_counts, cell_offsets = _build_dense_cell_table(p_cell_hash, cap)

        if do_prune:
            cell_center, cell_half = _loose_cell_aabbs(pos_s, pos_s, p_cell_hash)
        else:
            cell_center = cell_half = None

        chunk_size = min(n, 65_536)

        def run_pure_dense_build() -> tuple[Any, ...]:
            def stencil_prefix_batch(
                b_iota: Any, b_coords: Any
            ) -> tuple[Any, Any, Any, Any]:
                return _dense_stencil_prefix_batch(
                    b_iota,
                    b_coords,
                    n=n,
                    pos_s=pos_s,
                    dims_i=dims_i,
                    strides_i=strides_i,
                    eff_cell_size=eff_cell_size,
                    grid_dims=grid_dims,
                    grid_strides=grid_strides,
                    neighbor_mask=neighbor_mask,
                    offsets_mask=offsets_mask,
                    cell_offsets=cell_offsets,
                    system=system,
                    cap=cap,
                    periodic=periodic,
                    has_shear=has_shear,
                    use_compact_shear=use_compact_shear,
                    shear_r=shear_r,
                    do_prune=do_prune,
                    cell_center=cell_center,
                    cell_half=cell_half,
                    prune_rad=cutoff if do_prune else None,
                )

            (
                pre_starts,
                _pre_lens,
                pre_pref,
                pre_totals,
                max_total,
                tile_overflow,
            ) = _dense_grid_stats(
                n,
                chunk_size,
                iota,
                coords_s,
                stencil_prefix_batch,
                cell_counts=cell_counts,
                dims_i=dims_i,
                strides_i=strides_i,
                offsets_mask=offsets_mask,
                periodic=periodic,
                has_shear=has_shear,
                do_prune=do_prune,
                tile_capacity=tile_capacity,
                hash_overflow=hash_overflow,
            )

            def eval_candidates(
                b_iota: Any,
                starts_m: Any,
                pref_m: Any,
                totals: Any,
                slots: Any,
            ) -> tuple[Any, Any]:
                def one(s: Any, st_m: Any, pr_m: Any, tot: Any) -> tuple[Any, Any]:
                    k = _compacted_stencil_decode(st_m, pr_m, slots)
                    sk = jnp.minimum(k, n - 1)
                    dr = system.domain.displacement(pos_s[s], pos_s[sk], system)
                    dist_sq = norm2(dr)
                    valid = (slots < tot) & (dist_sq <= cutoff_sq)
                    if search_radii_s is not None:
                        reach = search_radii_s[s] + search_radii_s[sk] + skin
                        valid = valid & (dist_sq <= reach * reach)
                    valid = valid & valid_interaction_mask(
                        clumps_s[sk],
                        clumps_s[s],
                        bonds_s[sk],
                        perm[s],
                        interact_same,
                    ).astype(bool)
                    return perm[sk], valid

                return jax.vmap(one)(b_iota, starts_m, pref_m, totals)

            def build_width(width: int) -> tuple[Any, ...]:
                if width == 0:
                    counts = jnp.zeros_like(cell_counts, shape=(n,), dtype=int)
                    if capacity is None:
                        return counts, tile_overflow
                    offsets, pair_overflow = _capacity_offsets(counts, capacity)
                    return (
                        jnp.full_like(counts, -1, shape=(capacity,), dtype=int),
                        offsets,
                        tile_overflow | pair_overflow,
                    )

                slots = jax.lax.iota(dtype=int, size=width)
                if n <= chunk_size:
                    assert (
                        pre_starts is not None
                        and pre_pref is not None
                        and pre_totals is not None
                    )
                    dst_s, val_s = eval_candidates(
                        iota, pre_starts, pre_pref, pre_totals, slots
                    )
                    cnt_s = val_s.sum(axis=1, dtype=int)
                    counts = (
                        jnp.empty_like(cnt_s).at[perm].set(cnt_s, unique_indices=True)
                    )
                    if capacity is None:
                        return counts, tile_overflow
                    offsets, pair_overflow = _capacity_offsets(counts, capacity)
                    overflow = tile_overflow | pair_overflow
                    neighbors = jnp.full_like(counts, -1, shape=(capacity,), dtype=int)
                    if capacity == 0:
                        return neighbors, offsets, overflow
                    base_s = offsets[perm]
                    end_s = offsets[perm + 1]
                    ranks = jnp.cumsum(val_s, axis=1, dtype=int) - 1
                    writes = jnp.where(
                        val_s & (base_s[:, None] + ranks < end_s[:, None]),
                        base_s[:, None] + ranks,
                        capacity,
                    )
                    neighbors = neighbors.at[writes].set(
                        dst_s, unique_indices=True, mode="drop"
                    )
                    return neighbors, offsets, overflow

                full = n // chunk_size * chunk_size
                iota_g = iota[:full].reshape(-1, chunk_size)
                coords_g = coords_s[:full].reshape(-1, chunk_size, coords_s.shape[-1])

                def count_chunk(x: tuple[Any, Any]) -> Any:
                    bi, bc = x
                    st_c, _, pr_c, tot_c = stencil_prefix_batch(bi, bc)
                    _, val_c = eval_candidates(bi, st_c, pr_c, tot_c, slots)
                    return val_c.sum(axis=1, dtype=int)

                cnt_s = jax.lax.map(count_chunk, (iota_g, coords_g)).reshape(full)
                if full < n:
                    cnt_s = jnp.concatenate(
                        (cnt_s, count_chunk((iota[full:], coords_s[full:]))), axis=0
                    )
                counts = jnp.empty_like(cnt_s).at[perm].set(cnt_s, unique_indices=True)
                if capacity is None:
                    return counts, tile_overflow
                offsets, pair_overflow = _capacity_offsets(counts, capacity)
                overflow = tile_overflow | pair_overflow
                neighbors = jnp.full_like(counts, -1, shape=(capacity,), dtype=int)
                if capacity == 0:
                    return neighbors, offsets, overflow

                base_all = offsets[perm]
                end_all = offsets[perm + 1]

                def fill_chunk(
                    nbr: Any, x: tuple[Any, Any, Any, Any]
                ) -> tuple[Any, None]:
                    bi, bc, b_base, b_end = x
                    st_c, _, pr_c, tot_c = stencil_prefix_batch(bi, bc)
                    dst_c, val_c = eval_candidates(bi, st_c, pr_c, tot_c, slots)
                    ranks = jnp.cumsum(val_c, axis=1, dtype=int) - 1
                    writes = jnp.where(
                        val_c & (b_base[:, None] + ranks < b_end[:, None]),
                        b_base[:, None] + ranks,
                        capacity,
                    )
                    nbr = nbr.at[writes].set(dst_c, unique_indices=True, mode="drop")
                    return nbr, None

                neighbors, _ = jax.lax.scan(
                    fill_chunk,
                    neighbors,
                    (
                        iota_g,
                        coords_g,
                        base_all[:full].reshape(-1, chunk_size),
                        end_all[:full].reshape(-1, chunk_size),
                    ),
                )
                if full < n:
                    neighbors, _ = fill_chunk(
                        neighbors,
                        (iota[full:], coords_s[full:], base_all[full:], end_all[full:]),
                    )
                return neighbors, offsets, overflow

            widths = (0, 8, 16, 32, 48, 64, 96)
            branch_idx = sum((max_total > w).astype(int) for w in widths)
            branches = tuple(
                partial(lambda w, _: build_width(w), w) for w in widths
            ) + (lambda _: fallback_fn(),)
            return jax.lax.switch(branch_idx, branches, None)

        if shear_r >= 1 and (not has_shear or use_compact_shear):
            from ._pallas import try_pallas_build_pairs

            pallas_res = try_pallas_build_pairs(
                pos_s,
                coords_s,
                cell_offsets,
                dims_i,
                strides_i,
                eff_cell_size,
                perm,
                clumps_s,
                bonds_s,
                search_radii_s,
                system,
                cutoff,
                skin,
                interact_same,
                hash_overflow,
                shear_r,
                capacity,
                only_spheres=bool(state.only_spheres),
                do_prune=do_prune,
                cell_center=cell_center,
                cell_half=cell_half,
                tile_capacity=tile_capacity,
                pure_fallback=run_pure_dense_build,
            )
            if pallas_res is not None:
                return pallas_res

        return run_pure_dense_build()

    return jax.lax.cond(
        use_dense,
        lambda _: run_dense(cap_dense),
        lambda _: fallback_fn(),
        None,
    )


def _search_pairs_impl(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
    *,
    shared_loop: bool,
) -> tuple[Any, ...]:
    n = state.N
    if n == 0:
        if capacity is None:
            return jnp.zeros((0,), dtype=int), jnp.asarray(False)
        return (
            jnp.full((capacity,), -1, dtype=int),
            jnp.zeros((1,), dtype=int),
            jnp.asarray(False),
        )
    if not hasattr(system.collider, "neighbor_mask"):
        from .naive import NaiveSimulator

        if not isinstance(system.collider, NaiveSimulator):
            raise ValueError("NeighborList requires CellList, MultiCellList, or naive")
        return _search_pairs_naive(state, system, cutoff, capacity, search_radii, skin)

    def run_legacy() -> tuple[Any, ...]:
        sys_p, perm, starts, ends, hash_overflow = _prepare_search_partition(
            state, system, cutoff
        )
        pos = state.pos
        pos_s = pos[perm]
        clumps_s = state.clump_id[perm]
        bonds_s = state.bond_id[perm]
        search_radii_s = None if search_radii is None else search_radii[perm]
        batch_size = min(n, _SEARCH_BATCH_SIZE)
        padded_n = (n + batch_size - 1) // batch_size * batch_size
        batches = jnp.arange(padded_n).reshape(-1, batch_size)
        data = (
            pos,
            pos_s,
            perm,
            state.clump_id,
            clumps_s,
            bonds_s,
            sys_p,
            cutoff**2,
            search_radii,
            search_radii_s,
            skin,
        )

        def count_batch(rows: Any) -> Any:
            safe_rows = jnp.minimum(rows, n - 1)
            begin = starts[safe_rows]
            end = jnp.where(rows[:, None] < n, ends[safe_rows], begin)
            if shared_loop:
                steps = jnp.max((end - begin + 3) // 4, initial=0)
                initial = jnp.zeros_like(pos, shape=begin.shape, dtype=int)
                return _count_search_loop(steps, initial, (data, rows, begin, end))

            def step(carry: Any) -> Any:
                cursor, count = carry
                _, valid = _search_pair_candidates(
                    data, rows, cursor, end, need_dst=False
                )
                return cursor + 4, count + valid.sum(axis=-1)

            return jax.lax.while_loop(
                lambda x: jnp.any(x[0] < end),
                step,
                (
                    begin,
                    jnp.zeros_like(
                        _search_pair_candidates(data, rows, begin, end, need_dst=False)[
                            1
                        ].sum(axis=-1)
                    ),
                ),
            )[1]

        counts = jax.lax.map(count_batch, batches).reshape(padded_n, -1)[:n]
        if capacity is None:
            return counts.sum(axis=1), hash_overflow
        offsets, overflow = _capacity_offsets(counts.sum(axis=1), capacity)
        overflow = overflow | hash_overflow
        neighbors = jnp.full_like(counts, -1, shape=(capacity,), dtype=int)
        if capacity == 0:
            return neighbors, offsets, overflow

        def fill_batch(neighbors: Any, rows: Any) -> Any:
            safe_rows = jnp.minimum(rows, n - 1)
            slots, slot_ends = _csr_slot_intervals(offsets, counts, rows, safe_rows, n)
            begin = starts[safe_rows]
            end = jnp.where(rows[:, None] < n, ends[safe_rows], begin)
            if shared_loop:
                steps = jnp.max(
                    jnp.where(slot_ends > slots, (end - begin + 3) // 4, 0), initial=0
                )
                _, neighbors = _fill_search_loop(
                    steps, (slots, neighbors), (data, rows, begin, end, slot_ends)
                )
                return neighbors, None

            def step(carry: Any) -> Any:
                cursor, cur_slots, cur_neighbors = carry
                dst, valid = _search_pair_candidates(
                    data, rows, cursor, end, need_dst=True
                )
                ranks = jnp.cumsum(valid, axis=-1) - 1
                remaining = slot_ends - cur_slots
                writes = cur_slots[..., None] + jnp.minimum(ranks, remaining[..., None])
                writes = jnp.where(
                    valid & (ranks < remaining[..., None]), writes, capacity
                )
                cur_neighbors = cur_neighbors.at[writes].set(dst, mode="drop")
                return (
                    cursor + 4,
                    cur_slots + jnp.minimum(valid.sum(axis=-1), remaining),
                    cur_neighbors,
                )

            *_, neighbors = jax.lax.while_loop(
                lambda x: jnp.any(x[1] < slot_ends),
                step,
                (begin, slots, neighbors),
            )
            return neighbors, None

        neighbors, _ = jax.lax.scan(fill_batch, neighbors, batches)
        return neighbors, offsets, overflow

    sys_b = system.domain.update_bounds(state.pos, system, padding=cutoff)
    cell_size = _effective_cell_size(sys_b.collider, cutoff)
    return _dense_sorted_search_pairs(
        state,
        sys_b,
        cell_size,
        sys_b.collider.neighbor_mask,
        cutoff,
        capacity,
        search_radii,
        skin,
        prune_cells=False,
        tile_capacity=None,
        fallback_fn=run_legacy,
    )


def _search_pairs_scalar(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
) -> tuple[Any, ...]:
    return _search_pairs_impl(
        state, system, cutoff, capacity, search_radii, skin, shared_loop=False
    )


def _search_pairs_batched(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
) -> tuple[Any, ...]:
    return _search_pairs_impl(
        state, system, cutoff, capacity, search_radii, skin, shared_loop=True
    )


def remap_history(
    old: Any, old_src: Any, old_dst: Any, new_src: Any, new_dst: Any, initialized: Any
) -> Any:
    """Lexicographic pair lookup without N*N integer keys or quadratic matches."""
    if old_dst.size == 0 or old.size == 0:
        return initialized
    src, dst, order = jax.lax.sort(
        (old_src, old_dst, jnp.arange(old_dst.size)), num_keys=2
    )
    lo = jnp.zeros_like(new_dst, dtype=int)
    hi = jnp.full_like(new_dst, old_dst.size, dtype=int)

    def step(_: Any, bounds: Any) -> Any:
        lo, hi = bounds
        mid = (lo + hi) // 2
        idx = jnp.minimum(mid, old_dst.size - 1)
        less = (src[idx] < new_src) | ((src[idx] == new_src) & (dst[idx] < new_dst))
        return jnp.where(less, mid + 1, lo), jnp.where(less, hi, mid)

    # Old keys and new source IDs can have axes absent from new destinations.
    lo, hi = step(0, (lo, hi))
    lo, _ = jax.lax.fori_loop(1, old_dst.size.bit_length(), step, (lo, hi))
    idx = jnp.minimum(lo, old_dst.size - 1)
    found = (
        (lo < old_dst.size)
        & (src[idx] == new_src)
        & (dst[idx] == new_dst)
        & (new_dst >= 0)
    )
    found = found.reshape(found.shape + (1,) * (initialized.ndim - found.ndim))
    return jnp.where(found, old[order[idx]], initialized)


def _valid_pairs(neighbors: Any) -> tuple[Any, Any]:
    # The cache build already drops same-clump and bond-excluded pairs; editing
    # clump_id, bond_id, or interact_same_bond_id requires explicit invalidation.
    return jnp.maximum(neighbors, 0), neighbors >= 0


def pair_values(
    state: Any,
    system: Any,
    pos: Any,
    sources: Any,
    neighbors: Any,
    history: Any,
    *,
    advance_history: bool,
    quasistatic: bool = False,
) -> tuple[Any, ...]:
    """Evaluate scalar-pair laws with shared positions and interaction masking."""
    i = sources
    j, valid = _valid_pairs(neighbors)
    force, torque, updated = jax.vmap(
        lambda a, b, h: system.force_model.force(
            a,
            b,
            pos,
            state,
            system,
            h,
            advance_history=advance_history,
            quasistatic=quasistatic,
        ),
        in_axes=(None if i.ndim == 0 else 0, 0, 0),
    )(i, j, history)
    force = jnp.where(valid[:, None], force, 0.0)
    torque = jnp.where(valid[:, None], torque, 0.0)
    if advance_history:
        initialized = system.force_model.init_history(neighbors.shape, state.dim)
        mask = valid.reshape(valid.shape + (1,) * (updated.ndim - 1))
        updated = jnp.where(mask, updated, initialized)
    else:
        updated = history
    return force, torque, updated


def _build_one_cache(
    state: Any,
    system: Any,
    cutoff: Any,
    metric: Any,
    facet_id: Any,
) -> Any:
    """Unconditionally rebuild a single replica's CSR neighbor list and remap history."""
    from ._pallas import pallas_remap_csr_history

    col = system.collider
    search_radii = system.force_model.search_radii(state, system)
    nl, offsets, overflow = build_pairs(
        state,
        replace(system, collider=col.secondary_collider),
        cutoff + col.skin,
        col.neighbor_list.size,
        search_radii=search_radii,
        skin=col.skin,
    )
    initialized = system.force_model.init_history(nl.shape, state.dim) + jnp.zeros_like(
        col.history
    )
    if col.history.size:
        for source in (nl, offsets, col.neighbor_list, col.row_offsets):
            initialized = initialized + jnp.zeros_like(
                source, shape=col.history.shape, dtype=col.history.dtype
            )
    history = jax.lax.cond(
        col.n_build_times == 0,
        lambda _: initialized,
        lambda _: pallas_remap_csr_history(
            col.history,
            col.row_offsets,
            col.neighbor_list,
            offsets,
            nl,
            initialized,
            lambda: remap_history(
                col.history,
                pair_sources(col),
                col.neighbor_list,
                pair_sources(replace(col, neighbor_list=nl, row_offsets=offsets)),
                nl,
                initialized,
            ),
        ),
        None,
    )
    return replace(
        col,
        neighbor_list=nl,
        row_offsets=offsets,
        history=history,
        overflow=overflow,
        old_pos=state.pos,
        n_build_times=col.n_build_times + 1,
        metric_snapshot=metric,
        physical_cutoff_snapshot=cutoff,
        skin_snapshot=col.skin,
        facet_id_snapshot=facet_id,
        invalidated=jnp.zeros_like(col.invalidated),
    )


@custom_vmap
def _rebuild_selected(
    needs_build: Any,
    state: Any,
    system: Any,
    cutoff: Any,
    metric: Any,
    facet_id: Any,
) -> Any:
    """Rebuild only replicas with ``needs_build=True``, sub-batching under ``vmap``."""
    return jax.lax.cond(
        needs_build,
        lambda _: _build_one_cache(state, system, cutoff, metric, facet_id),
        lambda _: system.collider,
        None,
    )


@_rebuild_selected.def_vmap
def _rebuild_selected_vmap(
    axis_size: int,
    in_batched: Any,
    needs_build: Any,
    state: Any,
    system: Any,
    cutoff: Any,
    metric: Any,
    facet_id: Any,
) -> Any:
    needs_b, state_b, sys_b, cut_b, met_b, fid_b = in_batched
    col = system.collider
    col_b = sys_b.collider
    payload = (state, system, cutoff, metric, facet_id)
    payload_b = (state_b, sys_b, cut_b, met_b, fid_b)
    axes = jax.tree.map(lambda b: 0 if b else None, payload_b)

    def run_all(_: Any) -> Any:
        def one(nb: Any, *p: Any) -> Any:
            return jax.lax.cond(
                nb,
                lambda _: _build_one_cache(*p),
                lambda _: p[1].collider,
                None,
            )

        return jax.vmap(
            one, in_axes=(0 if needs_b else None, *axes), axis_size=axis_size
        )(needs_build, *payload)

    # Rebuild only the dirty replicas even for small batches: selecting whole
    # pools after rebuilding every replica cost 27% of B=6 step time.
    if not needs_b or axis_size < 2:
        out = run_all(None)
        return out, jax.tree.map(lambda _: True, out)

    n_dirty = jnp.sum(needs_build, dtype=int)
    ks = tuple(k for k in (1, 4, 16) if k < axis_size)

    def run_subbatch(k: int, _: Any) -> Any:
        ranks = jnp.cumsum(needs_build, dtype=int) - 1
        dirty_idx = (
            jnp.zeros((k,), dtype=int)
            .at[jnp.where(needs_build, ranks, k)]
            .set(jnp.arange(axis_size), mode="drop")
        )
        gathered = jax.tree.map(
            lambda x, b: x[dirty_idx] if b else x, payload, payload_b
        )
        rebuilt_k = jax.vmap(_build_one_cache, in_axes=axes, axis_size=k)(*gathered)
        w_idx = jnp.where(jnp.arange(k) < n_dirty, dirty_idx, axis_size)

        def scatter_leaf(old_v: Any, old_is_b: bool, new_v: Any) -> Any:
            base = (
                old_v
                if old_is_b
                else jnp.broadcast_to(old_v, (axis_size,) + old_v.shape)
            )
            return base.at[w_idx].set(new_v, mode="drop")

        return jax.tree.map(scatter_leaf, col, col_b, rebuilt_k)

    branch_idx = sum((n_dirty > k).astype(int) for k in ks)
    branches = tuple(partial(run_subbatch, k) for k in ks) + (run_all,)
    out = jax.lax.switch(branch_idx, branches, None)
    return out, jax.tree.map(lambda _: True, out)


@dataclass(frozen=True, slots=True)
class _CacheCheck:
    """Shared cache-validity predicates for rebuild, fused step, and deferred checks."""

    metric: Any
    cutoff: Any
    facet_id: Any
    max_disp2: Any
    displacement: Any | None
    rebuild: Any
    geometry_changed: Any

    def travel_displacement(self) -> Any:
        if self.displacement is not None:
            return self.displacement
        return 2.0 * jnp.sqrt(jax.lax.stop_gradient(self.max_disp2))

    def needs_rebuild(self, domain: Any, col: Any) -> Any:
        drift = domain.search_geometry_drift(col.metric_snapshot, col.old_pos)
        return self.rebuild | (
            self.geometry_changed & (self.travel_displacement() + drift > col.skin)
        )


def _max_facet_id(state: Any) -> Any:
    if state.only_spheres:
        return jnp.asarray(-1, dtype=state.facet_id.dtype)
    if state.facet_ids_extra.shape[-1] == 0:
        return jnp.max(state.facet_id, initial=-1)
    return jnp.maximum(
        jnp.max(state.facet_id, initial=-1),
        jnp.max(state.facet_ids_extra, initial=-1),
    )


def _motion_bound(state: Any, system: Any, pos: Any, horizon: int) -> Any:
    """Largest per-sphere travel since the last build, plus predicted travel.

    Each sphere adds a heuristic ``t |v| + t^2 |a| / 2`` estimate with
    ``t = horizon * dt`` (linear speed, clump rotational speed, and linear
    acceleration; angular acceleration and future force growth are caught on
    deferred steps by :func:`_flag_skin_violation`).
    """
    col = system.collider
    travel = norm(pos - col.old_pos)
    speed = norm(state.vel)
    if not state.only_spheres:
        speed = speed + norm(state.ang_vel) * norm(state._pos_p_rot)
    safe_mass = jnp.where(state.fixed, 1.0, state.mass)
    accel = jnp.where(state.fixed, 0.0, norm(state.force) / safe_mass)
    t = horizon * system.dt
    return jnp.max(
        jax.lax.stop_gradient(travel + t * speed + 0.5 * t * t * accel),
        initial=0.0,
    )


def _inspect_cache(state: Any, system: Any, pos: Any, *, horizon: int) -> _CacheCheck:
    """Evaluate all cache-validity predicates shared by checked and deferred paths."""
    col = system.collider
    dim = state.dim
    metric = system.domain.search_geometry_snapshot()
    if metric.shape != (dim + 3,):
        raise ValueError(
            f"Domain.search_geometry_snapshot() must have shape {(dim + 3,)}"
        )
    max_search_rad = jnp.max(
        system.force_model.search_radii(state, system), initial=0.0
    )
    cutoff = jnp.maximum(col.cutoff, 2 * max_search_rad)
    facet_id = _max_facet_id(state)

    if horizon > 0:
        displacement = 2.0 * _motion_bound(state, system, pos, horizon)
        max_disp2 = None
        skin_spent = displacement > col.skin
    else:
        max_disp2 = jnp.max(norm2(pos - col.old_pos), initial=0.0)
        displacement = None
        skin_spent = 4.0 * max_disp2 > col.skin * col.skin

    rebuild = (
        skin_spent
        | (col.n_build_times == 0)
        | col.invalidated
        | jnp.any(metric[dim + 1 :] != col.metric_snapshot[dim + 1 :])
        | (cutoff != col.physical_cutoff_snapshot)
        | (col.skin != col.skin_snapshot)
        | (facet_id != col.facet_id_snapshot)
    )
    geometry_changed = jnp.any(metric != col.metric_snapshot)
    return _CacheCheck(
        metric=metric,
        cutoff=cutoff,
        facet_id=facet_id,
        max_disp2=max_disp2,
        displacement=displacement,
        rebuild=rebuild,
        geometry_changed=geometry_changed,
    )


def _check_and_rebuild_body(
    state: Any, system: Any, pos: Any, *, zero_copy_while: bool
) -> Any:
    """Shared body for ``check_and_rebuild`` (while-loop primal, cond JVP/VJP)."""
    from ._pallas import has_manual_axes as _has_manual_axes

    col = system.collider
    chk = _inspect_cache(state, system, pos, horizon=col._check_horizon)
    use_subbatch = zero_copy_while and not _has_manual_axes(state, system)

    if col.history.size and _has_manual_axes(state, system):
        col = _with_axes(
            col,
            _build_one_cache(
                state,
                replace(system, collider=col),
                chk.cutoff,
                chk.metric,
                chk.facet_id,
            ),
        )

    def maybe_rebuild(cur_col: Any) -> Any:
        needs_build = chk.needs_rebuild(system.domain, cur_col)
        if use_subbatch:
            sys_c = replace(system, collider=cur_col)
            return jax.lax.cond(
                _shared_any(needs_build),
                lambda _: _rebuild_selected(
                    needs_build, state, sys_c, chk.cutoff, chk.metric, chk.facet_id
                ),
                lambda _: cur_col,
                None,
            )
        return jax.lax.cond(
            _shared_any(needs_build),
            lambda _: jax.lax.cond(
                needs_build,
                lambda __: _build_one_cache(
                    state,
                    replace(system, collider=cur_col),
                    chk.cutoff,
                    chk.metric,
                    chk.facet_id,
                ),
                lambda __: cur_col,
                None,
            ),
            lambda _: cur_col,
            None,
        )

    do_check = _shared_any(chk.rebuild | chk.geometry_changed)
    if zero_copy_while:
        # A 0-or-1-iteration while_loop lets XLA alias the carry buffer in place
        # on the ~98% no-rebuild steps, avoiding a full copy of neighbor_list
        # and history (29-205+ MB per step!) that lax.cond's false branch inserts.
        _, collider = jax.lax.while_loop(
            lambda carry: carry[0],
            lambda carry: (jnp.asarray(False), maybe_rebuild(carry[1])),
            (do_check, col),
        )
    else:
        collider = jax.lax.cond(
            do_check,
            lambda _: maybe_rebuild(col),
            lambda _: col,
            None,
        )
    return replace(system, collider=collider)


@jax.custom_jvp
def _check_and_rebuild_impl(state: Any, system: Any, pos: Any) -> Any:
    return _check_and_rebuild_body(state, system, pos, zero_copy_while=True)


@_check_and_rebuild_impl.defjvp
def _check_and_rebuild_jvp(primals: Any, tangents: Any) -> Any:
    primal_out = _check_and_rebuild_impl(*primals)
    tangent_out = jax.jvp(
        lambda st, sy, p: _check_and_rebuild_body(st, sy, p, zero_copy_while=False),
        primals,
        tangents,
    )[1]
    return primal_out, tangent_out


def check_and_rebuild(state: Any, system: Any, *, pos: Any | None = None) -> Any:
    """Rebuild the pooled pair cache when the Verlet skin budget is exhausted.

    Particle motion since the last build and any change of the periodic
    geometry since the last build are both charged against the skin: a pair
    separation can change by at most twice the largest displacement, plus
    the domain's bound on image motion. Changes to the search axes, the
    required reach, or the skin itself always rebuild. Adding facets also
    rebuilds, even when an existing larger primitive keeps the global reach
    unchanged. Manual rewiring of ``clump_id``, ``bond_id``, or
    ``interact_same_bond_id`` requires explicit invalidation.
    """
    pos = state.pos if pos is None else pos
    if system.collider._check_horizon < 0:
        return _flag_skin_violation(state, system, pos)
    return _check_and_rebuild_impl(state, system, pos)


def _flag_skin_violation(state: Any, system: Any, pos: Any) -> Any:
    """Deferred step: keep the cache and raise ``overflow`` if it becomes invalid.

    Detects skin exhaustion and cache invalidation without repairing; raise
    ``skin`` or lower ``check_every`` if overflow is observed.
    """
    col = system.collider
    chk = _inspect_cache(state, system, pos, horizon=0)
    exceeded = chk.needs_rebuild(system.domain, col)
    return replace(system, collider=replace(col, overflow=col.overflow | exceeded))


@jax.jit
def reindex_history(state: Any, system: Any, old: Any, new_to_old: jax.Array) -> Any:
    """Build a resized cache and transfer history by original particle index.

    ``system.collider`` must be allocated for ``state``. ``new_to_old[k]`` is
    the original index of particle ``k`` in the reduced state. Both endpoints
    of each new directed pair are mapped to the old cache before looking up
    history. New pairs and padding use the force law's initialized history.
    """
    system = check_and_rebuild(state, system)
    if old.history.size == 0:
        return system
    col = system.collider
    indices = jnp.concatenate((new_to_old, jnp.full((1,), -1, dtype=new_to_old.dtype)))
    source = indices[pair_sources(col)]
    target = indices[jnp.where(col.neighbor_list >= 0, col.neighbor_list, state.N)]
    history = remap_history(
        old.history,
        pair_sources(old),
        old.neighbor_list,
        source,
        target,
        col.history,
    )
    return replace(system, collider=replace(col, history=history))


def _row_reduce(col: Any, initial: Any, evaluate: Any, history: Any = None) -> Any:
    """Reduce bounded row batches, optionally writing disjoint history slots."""
    n, capacity = col.row_offsets.size - 1, col.neighbor_list.size
    if n == 0 or capacity == 0:
        totals = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), initial)
        return totals, history
    dim = col.old_pos.shape[-1]
    width = 16
    batch_size = _ROW_BATCH_SIZE[dim]
    lanes = jnp.arange(width)

    def batch(history: Any, rows: Any) -> Any:
        indices, starts, ends = rows
        limit = jnp.max(ends - starts)
        totals = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (indices.size,) + x.shape), initial
        )

        def body(carry: Any) -> Any:
            offset, totals, history = carry
            slots = starts[:, None] + offset + lanes
            valid = slots < ends[:, None]
            safe = jnp.minimum(slots, capacity - 1)
            neighbors = jnp.where(valid, col.neighbor_list[safe], -1)
            values, updated = evaluate(indices, neighbors, safe)
            totals = jax.tree.map(lambda a, b: a + b.sum(axis=1), totals, values)
            if history is not None:
                # CSR rows own disjoint intervals. Padding is dropped rather
                # than allowed to overwrite the following row's history.
                history = history.at[jnp.where(valid, slots, capacity)].set(
                    updated, mode="drop", unique_indices=True
                )
            return offset + width, totals, history

        carry = (0, totals, history)
        if history is not None:
            carry = _with_axes(carry, body(carry))
            carry = jax.lax.cond(
                _shared_any(limit > 0), body, lambda current: current, carry
            )
        _, totals, history = jax.lax.while_loop(lambda c: c[0] < limit, body, carry)
        return history, totals

    rows = (jnp.arange(n), col.row_offsets[:-1], col.row_offsets[1:])
    if n <= batch_size:
        history, totals = batch(history, rows)
        return totals, history
    full = n // batch_size * batch_size
    if history is not None:
        proposed, _ = batch(history, jax.tree.map(lambda x: x[:batch_size], rows))
        history = _with_axes(history, proposed)
    history, batches = jax.lax.scan(
        batch,
        history,
        jax.tree.map(lambda x: x[:full].reshape(-1, batch_size), rows),
    )
    totals = jax.tree.map(lambda x: x.reshape((full,) + x.shape[2:]), batches)
    if full < n:
        history, tail = batch(history, jax.tree.map(lambda x: x[full:], rows))
        totals = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), totals, tail)
    return totals, history


_CSR_DEGREE_BUCKETS = (0, 2, 4, 8, 12, 16, 24, 32, 64)
# (base_w, tail_w) two-tier CSR buckets for large N when a tiny outlier tail
# inflates max_deg far above typical particle degree (e.g., dilute or clumped).
_CSR_TWO_TIER_SPECS = ((2, 6), (3, 9), (8, 12), (12, 12), (16, 16))
_CSR_TWO_TIER_MIN_N = 4_096


def _eval_csr_chunked(
    state: Any,
    system: Any,
    advance_history: bool,
    quasistatic: bool,
    row_i: Any,
    row_start: Any,
    row_end: Any,
    width: int,
    history: Any = None,
) -> tuple[Any, Any]:
    """Evaluate CSR rows ``(row_i, row_start, row_end)`` up to ``width`` lanes."""
    col = system.collider
    m, capacity = int(row_i.shape[0]), col.neighbor_list.size
    pos = state.pos
    has_torque = system.force_model.has_pair_torque
    h_cur = col.history if history is None else history
    has_history = h_cur.size > 0
    write_history = has_history and advance_history
    batch_size = 16_384 if write_history else 65_536
    lanes = jnp.arange(width)
    empty_hist = system.force_model.init_history((width,), state.dim)

    def eval_row(i: Any, start: Any, end: Any, h_src: Any) -> Any:
        slots = start + lanes
        in_row = slots < end
        safe = jnp.minimum(slots, capacity - 1)
        nbr = col.neighbor_list[safe]
        j = jnp.maximum(nbr, 0)
        valid = in_row & (nbr >= 0)
        h_in = h_src[safe] if has_history else empty_hist
        f, t, h_new = jax.vmap(
            lambda b, h: system.force_model.force(
                i,
                b,
                pos,
                state,
                system,
                h,
                advance_history=advance_history,
                quasistatic=quasistatic,
            )
        )(j, h_in)
        f_masked = jnp.where(valid[:, None], f, 0.0)
        ft = (
            f_masked
            if not has_torque
            else (f_masked, jnp.where(valid[:, None], t, 0.0))
        )
        if not write_history:
            return ft
        h_mask = valid.reshape(valid.shape + (1,) * (h_new.ndim - 1))
        h_out = jnp.where(h_mask, h_new, empty_hist)
        w_slots = jnp.where(in_row, slots, capacity)
        return ft, h_out, w_slots

    def eval_batch_stateless(batch_rows: Any) -> Any:
        b_i, b_start, b_end = batch_rows
        raw = jax.vmap(lambda a, s, e: eval_row(a, s, e, h_cur))(b_i, b_start, b_end)
        if not has_torque:
            return _vec_sum(raw, 1)
        return _vec_sum(raw[0], 1), _vec_sum(raw[1], 1)

    def eval_batch_stateful(h_buf: Any, batch_rows: Any) -> tuple[Any, Any]:
        b_i, b_start, b_end = batch_rows
        ft_raw, h_out, w_slots = jax.vmap(lambda a, s, e: eval_row(a, s, e, h_buf))(
            b_i, b_start, b_end
        )
        ft_sum = (
            _vec_sum(ft_raw, 1)
            if not has_torque
            else (_vec_sum(ft_raw[0], 1), _vec_sum(ft_raw[1], 1))
        )
        h_buf = h_buf.at[w_slots].set(
            h_out.astype(h_buf.dtype), unique_indices=True, mode="drop"
        )
        return h_buf, ft_sum

    rows_all = (row_i, row_start, row_end)
    if not write_history:
        if m <= batch_size:
            return eval_batch_stateless(rows_all), h_cur
        full = m // batch_size * batch_size
        grouped = jax.tree.map(lambda x: x[:full].reshape(-1, batch_size), rows_all)
        out = jax.lax.map(eval_batch_stateless, grouped)
        out = jax.tree.map(lambda x: x.reshape((full,) + x.shape[2:]), out)
        if full < m:
            tail = eval_batch_stateless(jax.tree.map(lambda x: x[full:], rows_all))
            out = jax.tree.map(lambda a, b: jnp.concatenate((a, b), axis=0), out, tail)
        return out, h_cur

    if m <= batch_size:
        h_buf, out = eval_batch_stateful(h_cur, rows_all)
        return out, h_buf
    full = m // batch_size * batch_size
    grouped = jax.tree.map(lambda x: x[:full].reshape(-1, batch_size), rows_all)
    h_buf, out = jax.lax.scan(eval_batch_stateful, h_cur, grouped)
    out = jax.tree.map(lambda x: x.reshape((full,) + x.shape[2:]), out)
    if full < m:
        h_buf, tail = eval_batch_stateful(
            h_buf, jax.tree.map(lambda x: x[full:], rows_all)
        )
        out = jax.tree.map(lambda a, b: jnp.concatenate((a, b), axis=0), out, tail)
    return out, h_buf


def _finalize_csr_forces(state: Any, system: Any, out: Any) -> tuple[Any, Any]:
    """Attach clump center-of-mass lever-arm torque to summed pair forces."""
    if not system.force_model.has_pair_torque:
        force = out
        torque = (
            jnp.zeros_like(state.torque)
            if state.only_spheres
            else cross(state._pos_p_rot, force)
        )
    else:
        force, torque = out
        if not state.only_spheres:
            torque = torque + cross(state._pos_p_rot, force)
    return force, torque


def _eval_csr_width(
    state: Any,
    system: Any,
    advance_history: bool,
    quasistatic: bool,
    width: int,
    history: Any = None,
) -> tuple[Any, Any, Any]:
    """Evaluate CSR rows for a fixed degree bucket via variadic ``_vec_sum``."""
    col = system.collider
    h_cur = col.history if history is None else history
    if width == 0:
        return jnp.zeros_like(state.force), jnp.zeros_like(state.torque), h_cur
    n = state.N
    write_hist = h_cur.size > 0 and advance_history
    out, h_out = _eval_csr_chunked(
        state,
        system,
        advance_history,
        quasistatic,
        jnp.arange(n),
        col.row_offsets[:-1],
        col.row_offsets[1:],
        width,
        history=h_cur,
    )
    force, torque = _finalize_csr_forces(state, system, out)
    return force, torque, h_out if write_hist else h_cur


def _eval_csr_two_tier(
    state: Any,
    system: Any,
    advance_history: bool,
    quasistatic: bool,
    base_w: int,
    cap_tail: int,
    tail_w: int,
    history: Any = None,
) -> tuple[Any, Any, Any]:
    """Evaluate first ``base_w`` lanes for all rows, then compact overflow rows."""
    col = system.collider
    h_cur = col.history if history is None else history
    n = state.N
    starts = col.row_offsets[:-1]
    ends = col.row_offsets[1:]
    iota = jnp.arange(n)
    write_hist = h_cur.size > 0 and advance_history

    out, h_buf = _eval_csr_chunked(
        state,
        system,
        advance_history,
        quasistatic,
        iota,
        starts,
        ends,
        base_w,
        history=h_cur,
    )

    over = (ends - starts) > base_w
    ranks = jnp.cumsum(over, dtype=int) - 1
    slot = jnp.where(over, ranks, cap_tail)
    tail_i = jnp.full((cap_tail,), -1, dtype=int).at[slot].set(iota, mode="drop")
    valid_tail = tail_i >= 0
    safe_i = jnp.maximum(tail_i, 0)
    t_start = starts[safe_i] + base_w
    t_end = jnp.where(valid_tail, ends[safe_i], t_start)

    tail_out, h_buf = _eval_csr_chunked(
        state,
        system,
        advance_history,
        quasistatic,
        safe_i,
        t_start,
        t_end,
        tail_w,
        history=h_buf,
    )
    w_idx = jnp.where(valid_tail, safe_i, n)
    out = jax.tree.map(lambda a, da: a.at[w_idx].add(da, mode="drop"), out, tail_out)
    force, torque = _finalize_csr_forces(state, system, out)
    return force, torque, h_buf if write_hist else h_cur


def _csr_dispatch_index(n: int, deg: Any, max_deg: Any) -> tuple[Any, bool]:
    """Return ``(index, use_two_tier)`` selecting single- or two-tier CSR branch.

    Single-tier indices occupy ``0 .. len(_CSR_DEGREE_BUCKETS)`` (with the last
    index being the large-degree fallback). Two-tier indices follow immediately
    when ``n >= _CSR_TWO_TIER_MIN_N``.
    """
    single_idx = sum((max_deg > b).astype(int) for b in _CSR_DEGREE_BUCKETS)
    if n < _CSR_TWO_TIER_MIN_N:
        return single_idx, False
    cap_tail = max(n // 16, 64)
    n_single = len(_CSR_DEGREE_BUCKETS) + 1
    # Check two-tier specs in reverse so the narrowest valid base_w wins.
    idx = single_idx
    for k in range(len(_CSR_TWO_TIER_SPECS) - 1, -1, -1):
        base_w, tail_w = _CSR_TWO_TIER_SPECS[k]
        min_trigger = 4 if base_w == 3 else base_w
        n_over = _shared_max(jnp.sum(deg > base_w, dtype=int))
        ok = (
            (max_deg > min_trigger)
            & (max_deg <= base_w + tail_w)
            & (n_over <= cap_tail)
        )
        idx = jnp.where(ok, n_single + k, idx)
    return idx, True


def _eval_csr_large(
    state: Any,
    system: Any,
    advance_history: bool,
    quasistatic: bool,
    history: Any = None,
) -> tuple[Any, Any, Any]:
    """Fallback row-batched CSR evaluator when max degree exceeds bucket range."""
    col = system.collider
    h_cur = col.history if history is None else history
    pos = state.pos
    has_torque = system.force_model.has_pair_torque
    write_hist = h_cur.size > 0 and advance_history

    def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
        f, t, h = pair_values(
            state,
            system,
            pos,
            i,
            neighbors,
            h_cur[slots],
            advance_history=advance_history,
            quasistatic=quasistatic,
        )
        return (f if not has_torque else (f, t)), (h if write_hist else None)

    def evaluate_rows(i: Any, neighbors: Any, slots: Any) -> Any:
        ft, h = jax.vmap(evaluate)(i, neighbors, slots)
        return ft, h if write_hist else None

    initial = (
        jnp.zeros_like(state.force, shape=state.force.shape[1:])
        if not has_torque
        else (
            jnp.zeros_like(state.force, shape=state.force.shape[1:]),
            jnp.zeros_like(state.torque, shape=state.torque.shape[1:]),
        )
    )
    out, out_hist = _row_reduce(
        col,
        initial,
        evaluate_rows,
        h_cur if write_hist else None,
    )
    force, torque = _finalize_csr_forces(state, system, out)
    return force, torque, out_hist if write_hist else h_cur


_CSR_STATEFUL_BUCKETS = (0, 8, 16, 32, 64)


def _bucketed_csr_forces(
    state: Any,
    system: Any,
    quasistatic: bool,
    *,
    advance_history: bool = False,
) -> tuple[Any, Any, Any]:
    """Single-kernel degree-bucketed (or two-tier) CSR reduction."""
    col = system.collider
    n = state.N
    deg = col.row_offsets[1:] - col.row_offsets[:-1]
    max_deg = _shared_max(jnp.max(deg, initial=0))
    has_history = col.history.size > 0
    # Pass history as the explicit switch operand (not captured in `system`) so
    # XLA can alias the large history buffer in place across switch branches.
    h_op = col.history
    sys_eval = (
        replace(
            system,
            collider=replace(col, history=jnp.empty((0,), dtype=col.history.dtype)),
        )
        if has_history
        else system
    )
    buckets: tuple[int, ...]
    if has_history:
        # Stateful force laws (e.g. Cundall-Strack, combiners) have large jaxprs;
        # coarse buckets cut compile time 3x while keeping runtime throughput.
        buckets = _CSR_STATEFUL_BUCKETS
        branch_idx = sum((max_deg > b).astype(int) for b in buckets)
        use_two_tier = False
    else:
        buckets = _CSR_DEGREE_BUCKETS
        branch_idx, use_two_tier = _csr_dispatch_index(n, deg, max_deg)
    branches: tuple[Any, ...] = tuple(
        partial(
            lambda w, h: _eval_csr_width(
                state, sys_eval, advance_history, quasistatic, w, history=h
            ),
            b,
        )
        for b in buckets
    ) + (
        lambda h: _eval_csr_large(
            state, sys_eval, advance_history, quasistatic, history=h
        ),
    )
    if use_two_tier:
        cap_tail = max(n // 16, 64)
        branches = branches + tuple(
            partial(
                lambda bw, tw, h: _eval_csr_two_tier(
                    state,
                    sys_eval,
                    advance_history,
                    quasistatic,
                    bw,
                    cap_tail,
                    tw,
                    history=h,
                ),
                bw,
                tw,
            )
            for bw, tw in _CSR_TWO_TIER_SPECS
        )
    return jax.lax.switch(branch_idx, branches, h_op)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def rebuild_and_forces(
    state: Any, system: Any, quasistatic: bool = False
) -> tuple[Any, Any, Any]:
    """Check rebuild and evaluate stateless CSR forces in a single host-synced switch.

    Eliminates the back-to-back ``cond`` + ``switch`` GPU-to-CPU sync pair on
    the common no-rebuild path (~90-99% of simulation steps).
    """
    from ._pallas import can_use_pallas_csr_forces, pallas_csr_forces

    if can_use_pallas_csr_forces(
        state, system, advance_history=False, quasistatic=quasistatic
    ):
        pos = state.pos
        sys_new = check_and_rebuild(state, system, pos=pos)

        def _pure_stateless() -> tuple[Any, Any, Any]:
            return _bucketed_csr_forces(state, sys_new, quasistatic)

        force, torque, _ = pallas_csr_forces(
            state, sys_new, False, quasistatic, _pure_stateless, pos=pos
        )
        return sys_new, force, torque

    if system.collider._check_horizon < 0:
        sys_new = _flag_skin_violation(state, system, state.pos)
        force, torque, _ = _bucketed_csr_forces(state, sys_new, quasistatic)
        return sys_new, force, torque
    col, pos = system.collider, state.pos
    n = state.N
    chk = _inspect_cache(state, system, pos, horizon=col._check_horizon)
    needs_check = _shared_any(chk.rebuild | chk.geometry_changed)

    deg = col.row_offsets[1:] - col.row_offsets[:-1]
    max_deg = _shared_max(jnp.max(deg, initial=0))
    bucket_idx, use_two_tier = _csr_dispatch_index(n, deg, max_deg)
    dispatch_idx = jnp.where(needs_check, 0, bucket_idx + 1)

    def do_rebuild_and_eval(_: Any) -> tuple[Any, Any, Any]:
        sys_new = check_and_rebuild(state, system)
        col_new = sys_new.collider
        max_d = _shared_max(
            jnp.max(col_new.row_offsets[1:] - col_new.row_offsets[:-1], initial=0)
        )
        coarse = (0, 8, 16, 32, 64)
        idx = sum((max_d > b).astype(int) for b in coarse)
        f, t, _ = jax.lax.switch(
            idx,
            tuple(
                partial(
                    lambda w, _: _eval_csr_width(state, sys_new, False, quasistatic, w),
                    b,
                )
                for b in coarse
            )
            + (lambda _: _eval_csr_large(state, sys_new, False, quasistatic),),
            None,
        )
        return col_new, f, t

    def do_width(w: int, _: Any) -> tuple[Any, Any, Any]:
        f, t, _ = _eval_csr_width(state, system, False, quasistatic, w)
        return col, f, t

    def do_large(_: Any) -> tuple[Any, Any, Any]:
        f, t, _ = _eval_csr_large(state, system, False, quasistatic)
        return col, f, t

    branches: tuple[Any, ...] = (
        (do_rebuild_and_eval,)
        + tuple(partial(do_width, b) for b in _CSR_DEGREE_BUCKETS)
        + (do_large,)
    )
    if use_two_tier:
        cap_tail = max(n // 16, 64)

        def do_two_tier(bw: int, tw: int, _: Any) -> tuple[Any, Any, Any]:
            f, t, _ = _eval_csr_two_tier(
                state, system, False, quasistatic, bw, cap_tail, tw
            )
            return col, f, t

        branches = branches + tuple(
            partial(do_two_tier, bw, tw) for bw, tw in _CSR_TWO_TIER_SPECS
        )
    new_col, force, torque = jax.lax.switch(dispatch_idx, branches, None)
    return replace(system, collider=new_col), force, torque


@rebuild_and_forces.defjvp
def _rebuild_and_forces_jvp(quasistatic: bool, primals: Any, tangents: Any) -> Any:
    def fixed_path(state: Any, system: Any) -> tuple[Any, Any, Any]:
        rebuilt = check_and_rebuild(state, system)
        f, t, _ = _flat_forces(state, rebuilt, False, quasistatic)
        return rebuilt, f, t

    return (
        rebuild_and_forces(*primals, quasistatic),
        jax.jvp(fixed_path, primals, tangents)[1],
    )


@jax.custom_jvp
def energy(state: Any, system: Any) -> Any:
    """Sum half-weighted directed-pair energies over the cache."""
    pos = state.pos
    col = system.collider

    if (
        (col.history.size == 0 or col.neighbor_list.size > _HISTORY_PREFIX_SIZE)
        and state.N > 0
        and col.neighbor_list.size > 0
    ):
        n, capacity = state.N, col.neighbor_list.size
        starts = col.row_offsets[:-1]
        ends = col.row_offsets[1:]
        max_deg = _shared_max(jnp.max(ends - starts, initial=0))
        batch_size = _ROW_BATCH_SIZE[state.dim]
        rows_all = (jnp.arange(n), starts, ends)
        has_history = col.history.size > 0

        def eval_e_width(width: int) -> Any:
            if width == 0:
                return jnp.asarray(0.0, dtype=pos.dtype)
            lanes = jnp.arange(width)
            empty_hist = system.force_model.init_history((width,), state.dim)

            def eval_row(i: Any, start: Any, end: Any) -> Any:
                slots = start + lanes
                in_row = slots < end
                safe = jnp.minimum(slots, capacity - 1)
                nbr = col.neighbor_list[safe]
                j = jnp.maximum(nbr, 0)
                valid = in_row & (nbr >= 0)
                h_in = col.history[safe] if has_history else empty_hist
                e = jax.vmap(
                    lambda b, h: system.force_model.energy(i, b, pos, state, system, h)
                )(j, h_in)
                return jnp.sum(jnp.where(valid, e, 0.0))

            if n <= batch_size:
                row_e = jax.vmap(eval_row)(*rows_all)
            else:
                row_e = jax.lax.map(
                    lambda r: eval_row(*r), rows_all, batch_size=batch_size
                )
            return 0.5 * jnp.sum(row_e)

        def eval_e_large() -> Any:
            def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
                j, valid = _valid_pairs(neighbors)
                e = jax.vmap(
                    lambda b, h: system.force_model.energy(i, b, pos, state, system, h)
                )(j, col.history[slots])
                return jnp.where(valid, e, 0.0)

            rows, _ = _row_reduce(
                col,
                jnp.zeros_like(pos, shape=()),
                lambda i, neighbors, slots: (
                    jax.vmap(evaluate)(i, neighbors, slots),
                    None,
                ),
            )
            return 0.5 * rows.sum()

        buckets = _CSR_STATEFUL_BUCKETS if has_history else _CSR_DEGREE_BUCKETS
        branch_idx = sum((max_deg > b).astype(int) for b in buckets)
        branches = tuple(partial(lambda w, _: eval_e_width(w), b) for b in buckets) + (
            lambda _: eval_e_large(),
        )
        return jax.lax.switch(branch_idx, branches, None)

    if _uses_fixed_cache(system.collider, state.dim):
        minimum = (
            _HISTORY_PREFIX_SIZE
            if system.collider.history.size
            else _FIXED_PAIR_BATCH_SIZE
        )
        return _prefix_dispatch(_fixed_pool_energy, minimum)(state, system)

    def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
        j, valid = _valid_pairs(neighbors)
        e = jax.vmap(
            lambda b, h: system.force_model.energy(i, b, pos, state, system, h)
        )(j, system.collider.history[slots])
        return jnp.where(valid, e, 0.0)

    rows, _ = _row_reduce(
        system.collider,
        jnp.zeros_like(pos, shape=()),
        lambda i, neighbors, slots: (jax.vmap(evaluate)(i, neighbors, slots), None),
    )
    return 0.5 * rows.sum()


def _fixed_pool_energy(prefix: int, state: Any, system: Any) -> Any:
    """Evaluate one fixed prefix of the energy cache."""
    pos = state.pos

    def values(i: Any, neighbors: Any, slots: Any) -> Any:
        j, valid = _valid_pairs(neighbors)
        energies = jax.vmap(
            lambda a, b, h: system.force_model.energy(a, b, pos, state, system, h)
        )(i, j, system.collider.history[slots])
        return jnp.where(valid, energies, 0.0), None

    rows, _ = _fixed_pool_reduce(
        system.collider,
        jnp.zeros_like(pos, shape=()),
        values,
        prefix_capacity=prefix,
    )
    return 0.5 * rows.sum()


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def forces(
    state: Any, system: Any, advance_history: bool, quasistatic: bool = False
) -> tuple[Any, ...]:
    """Accumulate cached pairs by row, preserving or advancing packed history."""
    from ._pallas import can_use_pallas_csr_forces, pallas_csr_forces

    col = system.collider
    n, capacity = state.N, col.neighbor_list.size
    if n == 0 or capacity == 0:
        return jnp.zeros_like(state.force), jnp.zeros_like(state.torque), col.history

    use_default_row_batch = _ROW_BATCH_SIZE.get(state.dim) == _DEFAULT_ROW_BATCH_SIZE

    def _pure_forces() -> tuple[Any, ...]:
        from ._pallas import has_manual_axes

        if use_default_row_batch:
            if col.history.size > 0 and capacity <= _HISTORY_PREFIX_SIZE:
                return _flat_forces(state, system, advance_history, quasistatic)
            if col.history.size == 0 or not has_manual_axes(state, system):
                return _bucketed_csr_forces(
                    state, system, quasistatic, advance_history=advance_history
                )
        pos = state.pos
        if _uses_fixed_cache(col, state.dim):
            minimum = (
                _HISTORY_PREFIX_SIZE if col.history.size else _FIXED_PAIR_BATCH_SIZE
            )
            if capacity <= minimum:
                return _flat_forces(state, system, advance_history, quasistatic)

            def evaluate_prefix(prefix: int, st: Any, sy: Any) -> Any:
                return _fixed_pool_forces(
                    st, sy, advance_history, quasistatic, prefix_capacity=prefix
                )

            return _prefix_dispatch(
                evaluate_prefix,
                minimum,
                scalar_limit=_HISTORY_PAIR_BATCH_SIZE if col.history.size else None,
            )(state, system)
        write_history = col.history.size > 0 and advance_history

        def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
            return pair_values(
                state,
                system,
                pos,
                i,
                neighbors,
                col.history[slots],
                advance_history=advance_history,
                quasistatic=quasistatic,
            )

        def evaluate_rows(i: Any, neighbors: Any, slots: Any) -> Any:
            f, t, h = jax.vmap(evaluate)(i, neighbors, slots)
            return (f, t), h if write_history else None

        (force, torque), history = _row_reduce(
            col,
            (
                jnp.zeros_like(state.force, shape=state.force.shape[1:]),
                jnp.zeros_like(state.torque, shape=state.torque.shape[1:]),
            ),
            evaluate_rows,
            (
                system.force_model.init_history((capacity,), state.dim)
                + jnp.zeros_like(col.history)
                if write_history
                else None
            ),
        )
        return (
            force,
            torque + cross(state._pos_p_rot, force),
            history if write_history else col.history,
        )

    if use_default_row_batch and can_use_pallas_csr_forces(
        state, system, advance_history=advance_history, quasistatic=quasistatic
    ):
        return pallas_csr_forces(
            state, system, advance_history, quasistatic, _pure_forces
        )
    return _pure_forces()


def _fixed_pool_forces(
    state: Any,
    system: Any,
    advance_history: bool,
    quasistatic: bool,
    prefix_capacity: int | None = None,
) -> tuple[Any, ...]:
    """Evaluate the fixed pool with bounded force-law intermediates."""
    col = system.collider
    pos = state.pos
    write_history = col.history.size > 0 and advance_history

    def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
        f, t, h = pair_values(
            state,
            system,
            pos,
            i,
            neighbors,
            col.history[slots],
            advance_history=advance_history,
            quasistatic=quasistatic,
        )
        return (f, t), h if write_history else None

    (force, torque), history = _fixed_pool_reduce(
        col,
        (
            jnp.zeros_like(pos, shape=state.force.shape[1:], dtype=state.force.dtype),
            jnp.zeros_like(pos, shape=state.torque.shape[1:], dtype=state.torque.dtype),
        ),
        evaluate,
        (
            system.force_model.init_history(col.neighbor_list.shape, state.dim)
            + jnp.zeros_like(pos, shape=col.history.shape, dtype=col.history.dtype)
            if write_history
            else None
        ),
        prefix_capacity=prefix_capacity,
        preserve_dtype=not _uses_flat_cache(col, state.dim),
        join_forces=(
            col.history.size > 0
            and col.neighbor_list.size <= _HISTORY_PAIR_BATCH_SIZE
            and prefix_capacity in (None, col.neighbor_list.size)
        ),
    )
    return (
        force,
        torque + cross(state._pos_p_rot, force),
        history if write_history else col.history,
    )


# The dynamic backend cannot transpose its row loops. Differentiate the same
# pair laws over fixed cache slots, which also serve small static pools.
@energy.defjvp
def _energy_jvp(primals: Any, tangents: Any) -> Any:
    def fixed_pairs(state: Any, system: Any) -> Any:
        col = system.collider
        if state.N == 0 or col.neighbor_list.size == 0:
            return jnp.asarray(0.0, dtype=state.pos.dtype)
        i = jnp.minimum(pair_sources(col), max(state.N - 1, 0))
        j, valid = _valid_pairs(col.neighbor_list)
        pos = state.pos
        values = jax.vmap(
            lambda a, b, h: system.force_model.energy(a, b, pos, state, system, h)
        )(i, j, col.history)
        return 0.5 * jnp.where(valid, values, 0.0).sum()

    return energy(*primals), jax.jvp(fixed_pairs, primals, tangents)[1]


def _flat_forces(
    state: Any, system: Any, advance_history: bool, quasistatic: bool
) -> tuple[Any, ...]:
    """Evaluate fixed pool slots for a small cache or an AD tangent."""
    col = system.collider
    if state.N == 0 or col.neighbor_list.size == 0:
        return jnp.zeros_like(state.force), jnp.zeros_like(state.torque), col.history
    src = pair_sources(col)
    f, t, history = pair_values(
        state,
        system,
        state.pos,
        jnp.minimum(src, state.N - 1),
        col.neighbor_list,
        col.history,
        advance_history=advance_history,
        quasistatic=quasistatic,
    )
    # Padding has source N. Drop it instead of adding to an extra bucket.
    if (
        col.history.size > 0
        and col.neighbor_list.size <= _HISTORY_PAIR_BATCH_SIZE
        and f.dtype == t.dtype
        and _same_manual_axes(f, t)
    ):
        force_width = f.shape[-1]
        joined = jax.ops.segment_sum(
            jnp.concatenate((f, t), axis=-1), src, num_segments=state.N
        )
        force, torque = joined[..., :force_width], joined[..., force_width:]
    else:
        force = jax.ops.segment_sum(f, src, num_segments=state.N)
        torque = jax.ops.segment_sum(t, src, num_segments=state.N)
    return force, torque + cross(state._pos_p_rot, force), history


@forces.defjvp
def _forces_jvp(
    advance_history: bool, quasistatic: bool, primals: Any, tangents: Any
) -> Any:
    def fixed_pairs(state: Any, system: Any) -> Any:
        return _flat_forces(state, system, advance_history, quasistatic)

    return (
        forces(*primals, advance_history, quasistatic),
        jax.jvp(fixed_pairs, primals, tangents)[1],
    )


def refresh(
    state: Any, old: Any, new: Any, force_model: Any, *, reset_history: bool
) -> Any:
    same_n = old.old_pos.shape[0] == state.N
    old_size, new_size = old.neighbor_list.size, new.neighbor_list.size
    tail = old.history.shape[1:]
    has_history = tail != (0,)
    if not reset_history:
        if has_history and (not same_n or new_size < old_size):
            raise ValueError(
                "Changing particle count or shrinking history capacity requires reset_history=True"
            )
        if force_model is not None and tail != force_model.history_shape(state.dim):
            raise ValueError(
                "Changing the force-model history shape requires reset_history=True"
            )
    if reset_history or not same_n:
        if force_model is None and has_history:
            raise ValueError("force_model is required to initialize refreshed history")
        history = (
            new.history
            if force_model is None
            else force_model.init_history(new.neighbor_list.shape, state.dim)
        )
        return replace(new, history=history)
    if new_size > old_size and has_history and force_model is None:
        raise ValueError("force_model is required to initialize expanded history")
    size = min(old_size, new_size)
    history = (
        force_model.init_history(new.neighbor_list.shape, state.dim)
        if force_model is not None
        else jnp.empty((new_size,) + tail, old.history.dtype)
    )
    history = history.at[:size].set(old.history[:size])
    return replace(
        new,
        neighbor_list=new.neighbor_list.at[:size].set(old.neighbor_list[:size]),
        row_offsets=jnp.minimum(old.row_offsets, new_size),
        history=history,
        n_build_times=old.n_build_times,
        old_pos=old.old_pos,
        overflow=old.overflow,
        invalidated=jnp.asarray(True),
    )
