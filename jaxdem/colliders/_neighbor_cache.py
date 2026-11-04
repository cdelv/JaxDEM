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

from ..utils.linalg import cross, norm2
from . import valid_interaction_mask
from ._cell_lookup import _cell_starts
from ._dynamic_loop import shared_bounded_loop
from ._partition import (
    _capacity_offsets,
    _dedup_stencil_hashes,
    _get_spatial_partition,
    _sorted_cell_run_ends,
)

_SEARCH_BATCH_SIZE = 16_384
# Keep the more expensive 3D contact kernel within a smaller working set.
_ROW_BATCH_SIZE = {2: 65_536, 3: 16_384}
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


@custom_vmap
def _shared_idle(required: Any) -> Any:
    """Keep scalar checks unchanged and share one idle decision under vmap."""
    return jnp.asarray(False)


@_shared_idle.def_vmap
def _shared_idle_mapped(axis_size: int, in_batched: Any, required: Any) -> Any:
    return ~_shared_any(jnp.any(required)), False


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
    return getattr(left_type, "manual_axis_type", getattr(left_type, "vma", None)) == getattr(
        right_type, "manual_axis_type", getattr(right_type, "vma", None)
    )


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
            joined = jnp.concatenate(totals, axis=-1).at[src].add(
                jnp.concatenate(values, axis=-1), mode="drop"
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
    data: Any, rows: Any, cursor: Any, end: Any
) -> tuple[Any, Any]:
    pos, perm, clumps, bonds, system, cutoff_sq, search_radii, skin = data
    n = perm.shape[0]
    indices = cursor[..., None] + jnp.arange(4)
    safe = jnp.minimum(indices, n - 1)
    src = jnp.minimum(rows, n - 1)
    dst = perm[safe]
    valid = (rows[:, None, None] < n) & (indices < end[..., None])
    delta = system.domain.displacement(pos[src, None, None, :], pos[dst], system)
    distance_sq = norm2(delta)
    valid &= distance_sq <= cutoff_sq
    if search_radii is not None:
        reach = search_radii[src, None, None] + search_radii[dst] + skin
        valid &= distance_sq <= reach**2
    valid &= valid_interaction_mask(
        clumps[dst],
        clumps[src, None, None],
        bonds[dst],
        src[:, None, None],
        system.interact_same_bond_id,
    ).astype(bool)
    return dst, valid


def _count_search_step(index: Any, counts: Any, operands: Any) -> Any:
    data, rows, begin, end = operands
    _, valid = _search_pair_candidates(data, rows, begin + 4 * index, end)
    return counts + valid.sum(axis=-1)


def _fill_search_step(index: Any, carry: Any, operands: Any) -> Any:
    data, rows, begin, end, slot_ends = operands
    slots, neighbors = carry
    dst, valid = _search_pair_candidates(data, rows, begin + 4 * index, end)
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
        _SearchState(state.pos, state.clump_id, state.bond_id),
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


def _search_pairs_scalar(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
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
    collider = system.collider
    rows = jnp.arange(n)
    pos = state.pos
    if hasattr(collider, "neighbor_mask"):
        system = system.domain.update_bounds(pos, system, padding=cutoff)
        reach = jnp.maximum(jnp.max(collider.neighbor_mask), 1)
        cell_size = jnp.maximum(collider.cell_size, cutoff / reach)
        perm, hashes, stencil, hash_overflow = _get_spatial_partition(
            pos, system, cell_size, collider.neighbor_mask, rows
        )
        starts = _cell_starts(hashes, stencil)
        if system.domain.periodic:
            stencil = jax.vmap(_dedup_stencil_hashes)(stencil)
    else:
        # The all-pairs backend uses one cell, retaining O(N^2) search cost.
        from .naive import NaiveSimulator

        if not isinstance(collider, NaiveSimulator):
            raise ValueError("NeighborList requires CellList, MultiCellList, or naive")
        perm = rows
        hashes = jnp.zeros(n, dtype=int)
        stencil = jnp.zeros((n, 1), dtype=int)
        starts = jnp.zeros((n, 1), dtype=int)
        hash_overflow = jnp.asarray(False)

    lanes = jnp.arange(4)
    cutoff_sq = cutoff**2
    batch_size = min(n, _SEARCH_BATCH_SIZE)
    padded_n = (n + batch_size - 1) // batch_size * batch_size
    batches = jnp.arange(padded_n).reshape(-1, batch_size)

    def candidates(rows: Any, cursor: Any, target: Any) -> tuple[Any, Any]:
        indices = cursor[..., None] + lanes
        safe = jnp.minimum(indices, n - 1)
        src = jnp.minimum(rows, n - 1)
        dst = perm[safe]
        in_cell = (
            (rows[:, None, None] < n)
            & (indices < n)
            & (hashes[safe] == target[..., None])
        )
        dr = system.domain.displacement(pos[src, None, None, :], pos[dst], system)
        distance_sq = norm2(dr)
        if search_radii is None:
            within_reach = distance_sq <= cutoff_sq
        else:
            pair_reach = search_radii[src, None, None] + search_radii[dst] + skin
            within_reach = (distance_sq <= cutoff_sq) & (distance_sq <= pair_reach**2)
        valid = (
            in_cell
            & within_reach
            & valid_interaction_mask(
                state.clump_id[dst],
                state.clump_id[src, None, None],
                state.bond_id[dst],
                src[:, None, None],
                system.interact_same_bond_id,
            ).astype(bool)
        )
        return dst, valid

    def count_batch(rows: Any) -> Any:
        safe_rows = jnp.minimum(rows, n - 1)
        target = stencil[safe_rows]

        def step(carry: Any) -> Any:
            cursor, count = carry
            _, valid = candidates(rows, cursor, target)
            return cursor + 4, count + valid.sum(axis=-1)

        return jax.lax.while_loop(
            lambda x: jnp.any(
                (rows[:, None] < n)
                & (x[0] < n)
                & (hashes[jnp.minimum(x[0], n - 1)] == target)
            ),
            step,
            (
                starts[safe_rows],
                jnp.zeros_like(
                    candidates(rows, starts[safe_rows], target)[1].sum(axis=-1)
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
        target = stencil[safe_rows]
        # Each cell owns a disjoint interval within its particle's pooled row.
        # Clip before adding the base; padded particles receive no slots.
        base = offsets[safe_rows, None]
        budget = jnp.where(rows < n, offsets[safe_rows + 1] - offsets[safe_rows], 0)
        ends = base + jnp.minimum(counts[safe_rows].cumsum(axis=1), budget[:, None])
        slot_starts = jnp.concatenate((base, ends[:, :-1]), axis=1)

        def step(carry: Any) -> Any:
            cursor, slots, neighbors = carry
            dst, valid = candidates(rows, cursor, target)
            ranks = jnp.cumsum(valid, axis=-1) - 1
            remaining = ends - slots
            writes = slots[..., None] + jnp.minimum(ranks, remaining[..., None])
            writes = jnp.where(valid & (ranks < remaining[..., None]), writes, capacity)
            neighbors = neighbors.at[writes].set(dst, mode="drop")
            return (
                cursor + 4,
                slots + jnp.minimum(valid.sum(axis=-1), remaining),
                neighbors,
            )

        *_, neighbors = jax.lax.while_loop(
            lambda x: jnp.any(x[1] < ends),
            step,
            (starts[safe_rows], slot_starts, neighbors),
        )
        return neighbors, None

    neighbors, _ = jax.lax.scan(fill_batch, neighbors, batches)
    return neighbors, offsets, overflow


def _search_pairs_batched(
    state: Any,
    system: Any,
    cutoff: Any,
    capacity: int | None,
    search_radii: Any | None,
    skin: Any,
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
    collider = system.collider
    rows = jnp.arange(n)
    pos = state.pos
    if hasattr(collider, "neighbor_mask"):
        system = system.domain.update_bounds(pos, system, padding=cutoff)
        reach = jnp.maximum(jnp.max(collider.neighbor_mask), 1)
        cell_size = jnp.maximum(collider.cell_size, cutoff / reach)
        perm, hashes, stencil, hash_overflow = _get_spatial_partition(
            pos, system, cell_size, collider.neighbor_mask, rows
        )
        starts = _cell_starts(hashes, stencil)
        if system.domain.periodic:
            stencil = jax.vmap(_dedup_stencil_hashes)(stencil)
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

    batch_size = min(n, _SEARCH_BATCH_SIZE)
    padded_n = (n + batch_size - 1) // batch_size * batch_size
    batches = jnp.arange(padded_n).reshape(-1, batch_size)

    # Explicit operands let the loop's batching rule share one bound while
    # preserving independent domains and particle data in each environment.
    data = (
        pos,
        perm,
        state.clump_id,
        state.bond_id,
        system,
        cutoff**2,
        search_radii,
        skin,
    )

    def count_batch(rows: Any) -> Any:
        safe_rows = jnp.minimum(rows, n - 1)
        begin = starts[safe_rows]
        end = jnp.where(rows[:, None] < n, ends[safe_rows], begin)
        steps = jnp.max((end - begin + 3) // 4, initial=0)
        initial = jnp.zeros_like(pos, shape=begin.shape, dtype=int)
        return _count_search_loop(steps, initial, (data, rows, begin, end))

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
        # Each cell owns a disjoint interval within its particle's pooled row.
        # Clip before adding the base; padded particles receive no slots.
        base = offsets[safe_rows, None]
        budget = jnp.where(rows < n, offsets[safe_rows + 1] - offsets[safe_rows], 0)
        slot_ends = base + jnp.minimum(
            counts[safe_rows].cumsum(axis=1), budget[:, None]
        )
        slots = jnp.concatenate((base, slot_ends[:, :-1]), axis=1)
        begin = starts[safe_rows]
        end = jnp.where(rows[:, None] < n, ends[safe_rows], begin)
        steps = jnp.max(
            jnp.where(slot_ends > slots, (end - begin + 3) // 4, 0), initial=0
        )
        _, neighbors = _fill_search_loop(
            steps,
            (slots, neighbors),
            (data, rows, begin, end, slot_ends),
        )
        return neighbors, None

    neighbors, _ = jax.lax.scan(fill_batch, neighbors, batches)
    return neighbors, offsets, overflow


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


def _valid_pairs(state: Any, system: Any, i: Any, neighbors: Any) -> tuple[Any, Any]:
    j = jnp.maximum(neighbors, 0)
    valid = (neighbors >= 0) & valid_interaction_mask(
        state.clump_id[i],
        state.clump_id[j],
        state.bond_id[i],
        j,
        system.interact_same_bond_id,
    ).astype(bool)
    return j, valid


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
    j, valid = _valid_pairs(state, system, i, neighbors)
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


def check_and_rebuild(state: Any, system: Any) -> Any:
    """Rebuild the pooled pair cache when the Verlet skin budget is exhausted.

    Particle motion since the last build and any change of the periodic
    geometry since the last build are both charged against the skin: a pair
    separation can change by at most twice the largest displacement, plus
    the domain's bound on image motion. Changes to the search axes, the
    required reach, or the skin itself always rebuild. Adding facets also
    rebuilds, even when an existing larger primitive keeps the global reach
    unchanged. Manual rewiring of existing IDs requires explicit invalidation.
    """
    col, pos = system.collider, state.pos
    dim = state.dim
    metric = system.domain.search_geometry_snapshot()
    if metric.shape != (dim + 3,):
        raise ValueError(
            f"Domain.search_geometry_snapshot() must have shape {(dim + 3,)}"
        )
    search_radii = system.force_model.search_radii(state, system)
    cutoff = jnp.maximum(col.cutoff, 2 * jnp.max(search_radii, initial=0.0))
    facet_id = jnp.maximum(
        jnp.max(state.facet_id, initial=-1),
        jnp.max(state.facet_ids_extra, initial=-1),
    )

    displacement = 2.0 * jnp.sqrt(jnp.max(norm2(pos - col.old_pos), initial=0.0))
    rebuild = (
        (displacement > col.skin)
        | (col.n_build_times == 0)
        | col.invalidated
        | jnp.any(metric[dim + 1 :] != col.metric_snapshot[dim + 1 :])
        | (cutoff != col.physical_cutoff_snapshot)
        | (col.skin != col.skin_snapshot)
        | (facet_id != col.facet_id_snapshot)
    )
    geometry_changed = jnp.any(metric != col.metric_snapshot)

    def build(_: Any) -> Any:
        nl, offsets, overflow = build_pairs(
            state,
            replace(system, collider=col.secondary_collider),
            cutoff + col.skin,
            col.neighbor_list.size,
            search_radii=search_radii,
            skin=col.skin,
        )
        initialized = system.force_model.init_history(
            nl.shape, state.dim
        ) + jnp.zeros_like(col.history)
        if col.history.size:
            for source in (nl, offsets, col.neighbor_list, col.row_offsets):
                initialized = initialized + jnp.zeros_like(
                    source, shape=col.history.shape, dtype=col.history.dtype
                )
        history = jax.lax.cond(
            col.n_build_times == 0,
            lambda _: initialized,
            lambda _: remap_history(
                col.history,
                pair_sources(col),
                col.neighbor_list,
                pair_sources(replace(col, neighbor_list=nl, row_offsets=offsets)),
                nl,
                initialized,
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

    if col.history.size:
        col = _with_axes(col, build(None))

    def check_geometry(_: Any) -> Any:
        drift = system.domain.search_geometry_drift(col.metric_snapshot, col.old_pos)
        return jax.lax.cond(displacement + drift > col.skin, build, lambda _: col, None)

    def update(_: Any) -> Any:
        return jax.lax.cond(rebuild, build, check_geometry, None)

    def update_if_needed(_: Any) -> Any:
        return jax.lax.cond(rebuild | geometry_changed, update, lambda _: col, None)

    def update_active_batch(_: Any) -> Any:
        drift = system.domain.search_geometry_drift(col.metric_snapshot, col.old_pos)
        geometry_required = geometry_changed & (displacement + drift > col.skin)
        return jax.lax.cond(
            _shared_idle(rebuild | geometry_required),
            lambda _: col,
            update_if_needed,
            None,
        )

    collider = jax.lax.cond(
        _shared_idle(rebuild | geometry_changed),
        lambda _: col,
        update_active_batch,
        None,
    )
    return replace(system, collider=collider)


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
    width = 2 * dim
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


@jax.custom_jvp
def energy(state: Any, system: Any) -> Any:
    """Sum half-weighted directed-pair energies over the cache."""
    pos = state.pos

    if _uses_fixed_cache(system.collider, state.dim):
        minimum = (
            _HISTORY_PREFIX_SIZE
            if system.collider.history.size
            else _FIXED_PAIR_BATCH_SIZE
        )
        return _prefix_dispatch(_fixed_pool_energy, minimum)(state, system)

    def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
        j, valid = _valid_pairs(state, system, i, neighbors)
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
        j, valid = _valid_pairs(state, system, i, neighbors)
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
    col = system.collider
    n, capacity = state.N, col.neighbor_list.size
    if n == 0 or capacity == 0:
        return jnp.zeros_like(state.force), jnp.zeros_like(state.torque), col.history
    pos = state.pos
    # Keep scalar flat kernels and select occupied prefixes for explicit batches.
    if col.history.size == 0 and _uses_flat_cache(col, state.dim):
        if capacity <= _HISTORY_PREFIX_SIZE:
            return _flat_forces(state, system, advance_history, quasistatic)

        def evaluate_flat_prefix(prefix: int, st: Any, sy: Any) -> Any:
            if prefix == capacity:
                return _flat_forces(st, sy, advance_history, quasistatic)
            return _fixed_pool_forces(
                st, sy, advance_history, quasistatic, prefix_capacity=prefix
            )

        return _prefix_dispatch(
            evaluate_flat_prefix,
            minimum=_HISTORY_PREFIX_SIZE,
            scalar_limit=capacity,
        )(state, system)
    if _uses_fixed_cache(col, state.dim):
        minimum = _HISTORY_PREFIX_SIZE if col.history.size else _FIXED_PAIR_BATCH_SIZE
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
        j, valid = _valid_pairs(state, system, i, col.neighbor_list)
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
