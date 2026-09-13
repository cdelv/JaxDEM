# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Fixed-capacity packed directed pairs for the Verlet cache.

Two search passes count neighbors and fill a shared pool without a per-particle
capacity. CSR offsets identify each row;
source IDs are decoded only for operations that require a flat edge list.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from ..utils.linalg import cross, norm2
from . import valid_interaction_mask
from ._partition import _cell_starts
from .cell_list import _dedup_stencil_hashes, _get_spatial_partition

_SEARCH_BATCH_SIZE = 16_384
_ROW_FORCE_UNROLL = 4
# Bound force temporaries while retaining enough parallel work per GPU launch.
_ROW_BATCH_SIZE = 128 * 1024


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


def _capacity_offsets(counts: jax.Array, capacity: int) -> tuple[jax.Array, jax.Array]:
    """Saturate at capacity + 1 so even severe overflows cannot wrap integers."""
    limit = capacity + 1
    totals = jax.lax.associative_scan(
        lambda a, b: a + jnp.minimum(b, limit - a), jnp.minimum(counts, limit)
    )
    overflow = totals[-1] > capacity
    return jnp.concatenate(
        (jnp.zeros(1, dtype=counts.dtype), jnp.minimum(totals, capacity))
    ), overflow


def build_pairs(state: Any, system: Any, cutoff: Any, capacity: int) -> tuple[Any, ...]:
    n = state.N
    if n == 0:
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
        valid = (
            in_cell
            & (norm2(dr) <= cutoff_sq)
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
            (starts[safe_rows], jnp.zeros(target.shape, dtype=int)),
        )[1]

    counts = jax.lax.map(count_batch, batches).reshape(padded_n, -1)[:n]
    offsets, overflow = _capacity_offsets(counts.sum(axis=1), capacity)
    overflow = overflow | hash_overflow
    neighbors = jnp.full((capacity,), -1, dtype=int)
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


def remap_history(
    old: Any, old_src: Any, old_dst: Any, new_src: Any, new_dst: Any, initialized: Any
) -> Any:
    """Lexicographic pair lookup without N*N integer keys or quadratic matches."""
    if old_dst.size == 0 or old.size == 0:
        return initialized
    src, dst, order = jax.lax.sort(
        (old_src, old_dst, jnp.arange(old_dst.size)), num_keys=2
    )
    lo = jnp.zeros(new_dst.shape, dtype=int)
    hi = jnp.full(new_dst.shape, old_dst.size, dtype=int)

    def step(_: Any, bounds: Any) -> Any:
        lo, hi = bounds
        mid = (lo + hi) // 2
        idx = jnp.minimum(mid, old_dst.size - 1)
        less = (src[idx] < new_src) | ((src[idx] == new_src) & (dst[idx] < new_dst))
        return jnp.where(less, mid + 1, lo), jnp.where(less, hi, mid)

    lo, _ = jax.lax.fori_loop(0, old_dst.size.bit_length(), step, (lo, hi))
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
) -> tuple[Any, ...]:
    """Evaluate scalar-pair laws with shared positions and interaction masking."""
    i = sources
    j, valid = _valid_pairs(state, system, i, neighbors)
    force, torque, updated = jax.vmap(
        lambda a, b, h: system.force_model.force(
            a, b, pos, state, system, h, advance_history=advance_history
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
    col, pos = system.collider, state.pos
    metric = system.domain.search_geometry_snapshot()
    if metric.shape != (state.dim + 3,):
        raise ValueError(
            f"Domain.search_geometry_snapshot() must have shape {(state.dim + 3,)}"
        )
    cutoff = jnp.maximum(
        col.cutoff,
        2 * jnp.max(system.force_model.search_radii(state, system), initial=0.0),
    )
    moved = jnp.max(norm2(pos - col.old_pos), initial=0.0) > col.skin**2 / 4
    rebuild = (
        moved
        | (col.n_build_times == 0)
        | col.invalidated
        | jnp.any(metric != col.metric_snapshot)
        | (cutoff != col.physical_cutoff_snapshot)
        | (col.skin != col.skin_snapshot)
    )

    def build(_: Any) -> Any:
        nl, offsets, overflow = build_pairs(
            state,
            replace(system, collider=col.secondary_collider),
            cutoff + col.skin,
            col.neighbor_list.size,
        )
        initialized = system.force_model.init_history(nl.shape, state.dim)
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
        )

    col = jax.lax.cond(rebuild, build, lambda _: col, None)
    return replace(
        system,
        collider=replace(
            col,
            metric_snapshot=metric,
            physical_cutoff_snapshot=cutoff,
            skin_snapshot=col.skin,
            invalidated=jnp.asarray(False),
        ),
    )


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


def _row_reduce(col: Any, initial: Any, evaluate: Any) -> Any:
    """Reduce row batches with one precomputed loop bound per batch."""
    n, capacity = col.row_offsets.size - 1, col.neighbor_list.size
    if n == 0 or capacity == 0:
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), initial)
    lanes = jnp.arange(_ROW_FORCE_UNROLL)

    def batch(rows: Any) -> Any:
        indices, starts, ends = rows
        limit = jnp.max(ends - starts)
        totals = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (indices.size,) + x.shape), initial
        )

        def body(carry: Any) -> Any:
            offset, totals = carry
            slots = starts[:, None] + offset + lanes
            safe = jnp.minimum(slots, capacity - 1)
            neighbors = jnp.where(slots < ends[:, None], col.neighbor_list[safe], -1)
            values = jax.vmap(evaluate)(indices, neighbors, safe)
            totals = jax.tree.map(lambda a, b: a + b.sum(axis=1), totals, values)
            return offset + _ROW_FORCE_UNROLL, totals

        return jax.lax.while_loop(lambda c: c[0] < limit, body, (0, totals))[1]

    rows = (jnp.arange(n), col.row_offsets[:-1], col.row_offsets[1:])
    if n <= _ROW_BATCH_SIZE:
        return batch(rows)
    full = n // _ROW_BATCH_SIZE * _ROW_BATCH_SIZE
    batches = jax.lax.map(
        batch, jax.tree.map(lambda x: x[:full].reshape(-1, _ROW_BATCH_SIZE), rows)
    )
    results = jax.tree.map(lambda x: x.reshape((full,) + x.shape[2:]), batches)
    if full < n:
        tail = batch(jax.tree.map(lambda x: x[full:], rows))
        results = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), results, tail)
    return results


@jax.custom_jvp
def energy(state: Any, system: Any) -> Any:
    pos = state.pos

    def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
        j, valid = _valid_pairs(state, system, i, neighbors)
        e = jax.vmap(lambda b: system.force_model.energy(i, b, pos, state, system))(j)
        return jnp.where(valid, e, 0.0)

    rows = _row_reduce(system.collider, jnp.asarray(0.0, dtype=pos.dtype), evaluate)
    return 0.5 * rows.sum()


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def forces(state: Any, system: Any, advance_history: bool) -> tuple[Any, ...]:
    """Accumulate four neighbors per row, preserving or advancing packed history."""
    col = system.collider
    n, capacity = state.N, col.neighbor_list.size
    if n == 0 or capacity == 0:
        return jnp.zeros_like(state.force), jnp.zeros_like(state.torque), col.history
    pos = state.pos
    width = _ROW_FORCE_UNROLL
    starts, ends = col.row_offsets[:-1], col.row_offsets[1:]

    def evaluate(i: Any, neighbors: Any, slots: Any) -> Any:
        return pair_values(
            state,
            system,
            pos,
            i,
            neighbors,
            col.history[slots],
            advance_history=advance_history,
        )

    if col.history.size == 0 or not advance_history:
        force, torque = _row_reduce(
            col,
            (
                jnp.zeros(state.force.shape[1:], dtype=state.force.dtype),
                jnp.zeros(state.torque.shape[1:], dtype=state.torque.dtype),
            ),
            lambda i, neighbors, slots: evaluate(i, neighbors, slots)[:2],
        )
        return force, torque + cross(state._pos_p_rot, force), col.history
    # Flatten only this row block: profiling favors it for history-writing laws.
    sources, lanes = jnp.repeat(jnp.arange(n), width), jnp.arange(width)
    initial_history = system.force_model.init_history((capacity,), state.dim)
    limit = jnp.max(ends - starts)

    def body(carry: Any) -> Any:
        offset, force, torque, history = carry
        slots = starts[:, None] + offset + lanes
        valid = slots < ends[:, None]
        safe = jnp.minimum(slots, capacity - 1)
        dst = jnp.where(valid, col.neighbor_list[safe], -1)
        f, t, h = evaluate(sources, dst.reshape(-1), safe.reshape(-1))
        force += f.reshape(n, width, -1).sum(axis=1)
        torque += t.reshape(n, width, -1).sum(axis=1)
        history = history.at[jnp.where(valid, slots, capacity).reshape(-1)].set(
            h, mode="drop"
        )
        return offset + width, force, torque, history

    _, force, torque, history = jax.lax.while_loop(
        lambda x: x[0] < limit,
        body,
        (
            jnp.asarray(0, dtype=int),
            jnp.zeros_like(state.force),
            jnp.zeros_like(state.torque),
            initial_history,
        ),
    )
    return force, torque + cross(state._pos_p_rot, force), history


# Dynamic row loops avoid work on unused slots but cannot be transposed by JAX.
# Differentiate the same pair laws over fixed cache slots only when AD requests
# a tangent; ordinary force/energy evaluation always uses the row traversal.
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
            lambda a, b: system.force_model.energy(a, b, pos, state, system)
        )(i, j)
        return 0.5 * jnp.where(valid, values, 0.0).sum()

    return energy(*primals), jax.jvp(fixed_pairs, primals, tangents)[1]


@forces.defjvp
def _forces_jvp(advance_history: bool, primals: Any, tangents: Any) -> Any:
    def fixed_pairs(state: Any, system: Any) -> Any:
        col = system.collider
        if state.N == 0 or col.neighbor_list.size == 0:
            return (
                jnp.zeros_like(state.force),
                jnp.zeros_like(state.torque),
                col.history,
            )
        src = pair_sources(col)
        f, t, history = pair_values(
            state,
            system,
            state.pos,
            jnp.minimum(src, state.N - 1),
            col.neighbor_list,
            col.history,
            advance_history=advance_history,
        )
        force = jnp.zeros_like(state.force).at[src].add(f, mode="drop")
        torque = jnp.zeros_like(state.torque).at[src].add(t, mode="drop")
        return force, torque + cross(state._pos_p_rot, force), history

    return forces(*primals, advance_history), jax.jvp(fixed_pairs, primals, tangents)[1]


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
