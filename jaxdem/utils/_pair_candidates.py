# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Configured pair searches shared by contact and energy diagnostics."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


def _validate_state(state: State) -> None:
    if state.pos_c.ndim != 2 or state.dim not in (2, 3):
        raise ValueError("Pair analysis requires one 2D or 3D configuration.")


def _check_search(system: System) -> None:
    overflow = system.collider.overflow
    if not isinstance(overflow, jax.core.Tracer) and bool(overflow):
        raise ValueError(
            "The configured collider overflowed during pair analysis. "
            "Correct its search configuration before using these results."
        )


def _prepare_system(state: State, system: System) -> System:
    from ..colliders import (
        Collider,
        DynamicCellList,
        DynamicMultiCellList,
        NeighborList,
    )
    from ..colliders.naive import NaiveSimulator

    collider = system.collider
    if (
        not isinstance(
            collider,
            (NeighborList, DynamicCellList, DynamicMultiCellList, NaiveSimulator),
        )
        and type(collider) is not Collider
    ):
        raise NotImplementedError(
            f"Pair traversal is unavailable for {type(collider).__name__}."
        )
    if state.N:
        reach = jnp.max(system.force_model.search_radii(state, system), initial=0.0)
        system = system.domain.update_bounds(state.pos, system, padding=reach)
    if isinstance(collider, NeighborList):
        from ..colliders._neighbor_cache import check_and_rebuild

        system = check_and_rebuild(state, system)
    return system


def _record_overflow(system: System, overflow: jax.Array) -> System:
    return replace(
        system,
        collider=replace(system.collider, overflow=overflow),
        search_overflow=system.search_overflow | overflow,
    )


@jax.jit
def _cached_pairs(
    state: State, system: System
) -> tuple[System, jax.Array, jax.Array, jax.Array]:
    from ..colliders import NeighborList
    from ..colliders._neighbor_cache import _valid_pairs, pair_sources

    system = _prepare_system(state, system)
    collider = cast(NeighborList, system.collider)
    i, j = pair_sources(collider), collider.neighbor_list
    valid = jnp.arange(j.size) < collider.row_offsets[-1]
    if state.N:
        _, mask = _valid_pairs(state, system, jnp.minimum(i, state.N - 1), j)
        valid = valid & mask
    return (
        _record_overflow(system, collider.overflow),
        jnp.stack((i, j), axis=1),
        valid,
        collider.history,
    )


@partial(jax.jit, static_argnames=("capacity",))
def _uncached_pairs(
    state: State, system: System, cutoff: jax.Array, capacity: int
) -> tuple[jax.Array, jax.Array]:
    from ..colliders._neighbor_cache import build_pairs
    from ..colliders.naive import NaiveSimulator

    neighbors, offsets, overflow = build_pairs(state, system, cutoff, capacity)
    sources = jnp.repeat(
        jnp.arange(state.N), jnp.diff(offsets), total_repeat_length=capacity
    )
    # The packed builder's bond mask follows the cell traversal's destination
    # convention. Naive force traversal applies the source adjacency instead.
    pairs = jnp.stack((sources, neighbors), axis=1)
    if isinstance(system.collider, NaiveSimulator):
        pairs = pairs[:, ::-1]
    return pairs, overflow


def _collect_pair_candidates(
    state: State, system: System
) -> tuple[System, jax.Array, jax.Array, jax.Array]:
    """Return system, directed pair IDs, validity, and aligned pair history.

    NeighborList exposes its refreshed pooled cache. Other colliders collect
    candidates into an automatically sized packed buffer. Pair validity
    includes the collider's interaction exclusions. Forces and energies are
    not evaluated, and contact history is not advanced.
    """
    from ..colliders import Collider, NeighborList
    from ..colliders._neighbor_cache import count_pairs

    _validate_state(state)
    if not state.N or type(system.collider) is Collider:
        return (
            system,
            jnp.empty((0, 2), dtype=int),
            jnp.empty((0,), dtype=bool),
            system.force_model.init_history((0,), state.dim),
        )
    if isinstance(system.collider, NeighborList):
        system, pairs, valid, history = _cached_pairs(state, system)
    else:
        system = _prepare_system(state, system)
        cutoff = 2 * jnp.max(
            system.force_model.search_radii(state, system), initial=0.0
        )
        counts, overflow = count_pairs(state, system, cutoff)
        system = _record_overflow(system, overflow)
        _check_search(system)
        capacity = int(np.sum(np.asarray(counts), dtype=np.int64))
        pairs, overflow = _uncached_pairs(state, system, cutoff, capacity)
        system = _record_overflow(system, overflow)
        valid = jnp.ones((capacity,), dtype=bool)
        history = system.force_model.init_history((capacity,), state.dim)
    _check_search(system)
    return system, pairs, valid, history
