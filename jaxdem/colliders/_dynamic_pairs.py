# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared dynamic force traversal for the cell colliders."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from . import valid_interaction_mask
from ._dynamic_loop import shared_bounded_loop
from ._cell_lookup import _cell_starts
from ._partition import (
    _dedup_stencil_hashes,
    _energy_pair_fn,
    _force_pair_fn,
    _get_spatial_partition,
    _loose_cell_aabbs,
    _sorted_cell_run_ends,
)

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


def _can_share_pair_loop(pair_fn: Callable[..., Any]) -> bool:
    """Recognize kernels whose system can enter the loop as an operand."""
    return (
        isinstance(pair_fn, partial)
        and pair_fn.func in (_force_pair_fn, _energy_pair_fn)
        and not pair_fn.args
        and "system" in pair_fn.keywords
        and set(pair_fn.keywords).issubset({"system", "quasistatic"})
    )


@partial(
    jax.jit,
    inline=True,
    static_argnames=("pair_fn", "legacy", "prune_cells", "pair_unroll", "batch_size"),
)
def _dispatch_pair_traversal(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Any,
    init_acc: Any,
    *,
    pair_system: System,
    legacy: Callable[..., Any],
    prune_cells: bool,
    pair_unroll: int,
    batch_size: int,
) -> tuple[Any, jax.Array]:
    """Keep scalar traversal and select shared traversal for explicit batching."""
    pair_impl = pair_fn.func
    pair_keywords = {
        key: value for key, value in pair_fn.keywords.items() if key != "system"
    }

    @custom_vmap
    def dispatch(
        state: State,
        system: System,
        bound_system: System,
        cell_size: jax.Array,
        neighbor_mask: jax.Array,
        initial: Any,
    ) -> Any:
        rebound = partial(pair_impl, system=bound_system, **pair_keywords)
        if state.N and neighbor_mask.shape[0]:
            # Match the geometry mask and pair law's read dependencies only.
            pos = state.pos
            perm, hashes, queries, _ = _get_spatial_partition(
                pos, system, cell_size, neighbor_mask, jnp.arange(state.N)
            )
            members = jnp.minimum(jnp.arange(pair_unroll), state.N - 1)
            destinations = perm[members]
            valid = (hashes[members] == queries[0, 0]) & valid_interaction_mask(
                state.clump_id[destinations],
                state.clump_id[0],
                state.bond_id[destinations],
                jnp.asarray(0, dtype=int),
                system.interact_same_bond_id,
            ).astype(bool)
            prototype = rebound(
                jax.tree.map(jnp.zeros_like, initial),
                jnp.asarray(0, dtype=int),
                destinations,
                pos,
                state,
                valid,
            )
            initial = jax.tree.map(
                lambda start, result: start + jnp.zeros_like(result.sum(axis=0)),
                initial,
                prototype,
            )
        return legacy(state, system, cell_size, neighbor_mask, rebound, initial)

    @dispatch.def_vmap
    def mapped(
        axis_size: int,
        in_batched: Any,
        state: State,
        system: System,
        bound_system: System,
        cell_size: jax.Array,
        neighbor_mask: jax.Array,
        initial: Any,
    ) -> Any:
        axes = jax.tree.map(lambda batched: 0 if batched else None, in_batched)

        def one(
            state: State,
            system: System,
            bound_system: System,
            cell_size: jax.Array,
            neighbor_mask: jax.Array,
            initial: Any,
        ) -> Any:
            rebound = partial(pair_impl, system=bound_system, **pair_keywords)
            return _traverse_shared_pairs(
                state,
                system,
                cell_size,
                neighbor_mask,
                rebound,
                initial,
                pair_system=bound_system,
                prune_cells=prune_cells,
                pair_unroll=pair_unroll,
                batch_size=batch_size,
            )

        result = jax.vmap(one, in_axes=tuple(axes), axis_size=axis_size)(
            state, system, bound_system, cell_size, neighbor_mask, initial
        )
        return result, jax.tree.map(lambda _: True, result)

    return dispatch(state, system, pair_system, cell_size, neighbor_mask, init_acc)


@partial(
    jax.jit,
    inline=True,
    static_argnames=("pair_fn", "prune_cells", "pair_unroll", "batch_size"),
)
def _traverse_shared_pairs(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Any,
    init_acc: Any,
    *,
    pair_system: System,
    prune_cells: bool,
    pair_unroll: int,
    batch_size: int,
) -> tuple[Any, jax.Array]:
    """Fold known contact kernels with one bound across explicit batch axes."""
    pair_impl = pair_fn.func
    pair_keywords = {
        key: value for key, value in pair_fn.keywords.items() if key != "system"
    }
    n = state.N
    if n == 0:
        return jax.tree.map(
            lambda x: jnp.empty((0,) + x.shape, x.dtype), init_acc
        ), jnp.asarray(False)
    rows = jnp.arange(n)
    pos = state.pos
    perm, hashes, targets, overflow = _get_spatial_partition(
        pos, system, cell_size, neighbor_mask, rows
    )
    starts = _cell_starts(hashes, targets)
    if system.domain.periodic:
        targets = jax.vmap(_dedup_stencil_hashes)(targets)
    safe = jnp.minimum(starts, n - 1)
    present = (starts < n) & (hashes[safe] == targets)
    ends = jnp.where(present, _sorted_cell_run_ends(hashes)[safe], starts)
    if prune_cells and not hasattr(system.domain, "gamma"):
        radii = system.force_model.search_radii(state, system)
        center, half = _loose_cell_aabbs(
            (pos - radii[:, None])[perm], (pos + radii[:, None])[perm], hashes
        )
        delta = system.domain.displacement(pos[:, None, :], center[safe], system)
        overlap = jnp.all(jnp.abs(delta) <= radii[:, None, None] + half[safe], axis=-1)
        ends = jnp.where(overlap, ends, starts)

    def body(step: jax.Array, accumulators: Any, operands: Any) -> Any:
        indices, begin, end, perm, pos, state, system, bound_system = operands
        cursor = begin + step * pair_unroll

        def row(
            orig: jax.Array, row_cursor: jax.Array, row_end: jax.Array, row_acc: Any
        ) -> Any:
            def cell(k: jax.Array, stop: jax.Array, acc: Any) -> Any:
                candidate = k + jnp.arange(pair_unroll)
                safe = jnp.minimum(candidate, n - 1)
                other = perm[safe]
                valid = (candidate < stop) & valid_interaction_mask(
                    state.clump_id[other],
                    state.clump_id[orig],
                    state.bond_id[other],
                    orig,
                    system.interact_same_bond_id,
                ).astype(bool)
                values = pair_impl(
                    jax.tree.map(jnp.zeros_like, acc),
                    orig,
                    other,
                    pos,
                    state,
                    valid,
                    system=bound_system,
                    **pair_keywords,
                )
                return jax.tree.map(
                    lambda old, values: old + values.sum(axis=0), acc, values
                )

            return jax.vmap(cell)(row_cursor, row_end, row_acc)

        return jax.vmap(row)(indices, cursor, end, accumulators)

    loop = shared_bounded_loop(body)

    def batch(args: tuple[jax.Array, jax.Array, jax.Array]) -> Any:
        indices, begin, end = args
        steps = jnp.max(
            (end - begin + pair_unroll - 1) // pair_unroll,
            initial=0,
        )
        initial = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (indices.shape[0], begin.shape[1]) + x.shape),
            init_acc,
        )
        result = loop(
            steps, initial, (indices, begin, end, perm, pos, state, system, pair_system)
        )
        return jax.tree.map(lambda x: x.sum(axis=1), result)

    width = min(n, batch_size)
    full = n // width * width
    inputs = (rows, starts, ends)
    grouped = jax.tree.map(
        lambda x: x[:full].reshape((full // width, width) + x.shape[1:]), inputs
    )
    values = jax.lax.map(batch, grouped)
    values = jax.tree.map(lambda x: x.reshape((full,) + x.shape[2:]), values)
    if full < n:
        tail = batch(jax.tree.map(lambda x: x[full:], inputs))
        values = jax.tree.map(lambda x, y: jnp.concatenate((x, y)), values, tail)
    return values, overflow
