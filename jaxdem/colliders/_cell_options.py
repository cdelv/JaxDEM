# SPDX-License-Identifier: BSD-3-Clause
"""Shared options for dynamic and static cell traversal."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import replace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from ..state import State
    from ..system import System
    from .cell_list import DynamicCellList
    from .multi_cell_list import DynamicMultiCellList
    from .neighbor_list import NeighborList


def validate_cell_options(
    tile_capacity: int | None, clump_aware: bool, static: bool = False
) -> None:
    """Check the fixed tile budget and clump option."""
    if tile_capacity is not None:
        if (
            isinstance(tile_capacity, bool)
            or not isinstance(tile_capacity, int)
            or tile_capacity < 0
        ):
            raise ValueError("tile_capacity must be a non-negative Python integer")
        index_limit = (1 << (63 if jax.config.jax_enable_x64 else 31)) - 1
        if tile_capacity >= index_limit:
            raise ValueError("tile_capacity exceeds the particle-index integer range")
    if not isinstance(clump_aware, bool):
        raise ValueError("clump_aware must be a Python boolean")
    if not isinstance(static, bool):
        raise ValueError("static must be a Python boolean")


def validate_cell_capacity_estimate(
    state: State,
    cell_size: ArrayLike | None,
    tile_capacity: int | None,
    static: bool,
) -> None:
    """Require concrete reference data for an automatic static budget."""
    if (
        static
        and tile_capacity is None
        and any(
            isinstance(value, jax.core.Tracer)
            for value in (state.pos_c, state._rad, cell_size)
        )
    ):
        raise ValueError(
            "Static cell construction requires explicit `tile_capacity` "
            "when called while tracing."
        )


def estimate_neighbor_capacity(
    state: State,
    cutoff: jax.Array,
    number_density: float = 1.0,
    safety_factor: float = 1.2,
) -> int:
    """Estimate occupancy from number density and typical particle packing."""
    if state.N == 0:
        return 0
    max_rad = jnp.max(state._rad)
    pos_min = jnp.min(state.pos, axis=0)
    pos_max = jnp.max(state.pos, axis=0)
    box_size = jnp.maximum(pos_max - pos_min + 2.0 * max_rad, 1.0)
    number_density_est = (state.N / jnp.prod(box_size)).item()
    effective_density = jnp.maximum(number_density, number_density_est).item()
    r_eff_mean = 0.9 * jnp.mean(state._rad)
    packing_count = int(
        jnp.where(
            r_eff_mean > 0,
            jnp.ceil(((cutoff + r_eff_mean) / r_eff_mean) ** state.dim),
            state.N,
        ).item()
    )
    search_volume = (
        jnp.pi * cutoff**state.dim * (1.0 if state.dim == 2 else (4.0 / 3.0))
    )
    density_count = int(
        jnp.ceil(safety_factor * search_volume * effective_density).item()
    )
    return min(max(density_count, packing_count, 0), state.N)


def resolve_cell_tile_capacity(
    state: State,
    cell_size: jax.Array,
    stencil_size: int,
    tile_capacity: int | None,
    static: bool,
) -> int | None:
    """Resolve an explicit budget or estimate tiles for all queried cells."""
    if tile_capacity is not None or not static:
        return tile_capacity
    validate_cell_capacity_estimate(state, cell_size, tile_capacity, static)
    from ._partition import PAIR_UNROLL

    # A sphere with this radius contains one complete grid cell.
    cell_radius = jnp.sqrt(float(state.dim)) * cell_size / 2.0
    occupancy = estimate_neighbor_capacity(state, cell_radius)
    tiles_per_cell = (occupancy + PAIR_UNROLL - 1) // PAIR_UNROLL
    capacity = state.N * stencil_size * tiles_per_cell
    capacity = min(capacity, state.N * state.N)
    index_limit = (1 << (63 if jax.config.jax_enable_x64 else 31)) - 1
    if capacity >= index_limit:
        # The full system measures the actual work before allocating a pool.
        capacity = state.N * min(stencil_size, state.N)
    validate_cell_options(capacity, False, static)
    return capacity


def configure_cell_capacity(state: State, system: System) -> System:
    """Size an automatic pool once the search geometry and force law exist."""
    collider = system.collider
    backend = getattr(collider, "secondary_collider", collider)
    if not getattr(backend, "_auto_capacity", False):
        return system
    backend = cast("DynamicCellList | DynamicMultiCellList", backend)
    if any(
        isinstance(value, jax.core.Tracer) for value in jax.tree.leaves((state, system))
    ):
        raise ValueError(
            "Automatic static capacity requires concrete reference data. "
            "Set tile_capacity when constructing a system while tracing."
        )
    if state.pos_c.ndim > 2:
        from ..state import State

        capacity = max(
            _measure_cell_capacity(snapshot, system)
            for snapshot in State.unstack(state)
        )
    else:
        capacity = _measure_cell_capacity(state, system)
    backend = replace(backend, tile_capacity=capacity)
    collider = (
        replace(cast("NeighborList", collider), secondary_collider=backend)
        if hasattr(collider, "secondary_collider")
        else backend
    )
    return replace(system, collider=collider)


def _measure_cell_capacity(state: State, system: System) -> int:
    """Add headroom to measured tiles, with room for occupied stencil cells."""
    from ._partition import PAIR_UNROLL
    from ._static_cell import _partition_tile_counts

    collider = system.collider
    backend = cast(
        "DynamicCellList | DynamicMultiCellList",
        getattr(collider, "secondary_collider", collider),
    )
    n = state.N
    if n == 0:
        return 0
    radii = system.force_model.search_radii(state, system)
    radius = jnp.max(radii, initial=0.0)
    stencil_reach = jnp.maximum(jnp.max(jnp.abs(backend.neighbor_mask)), 1)
    if hasattr(collider, "secondary_collider"):
        neighbor = cast("NeighborList", collider)
        reach = jnp.maximum(neighbor.cutoff, 2 * radius) + neighbor.skin
        member_radii, query_radii = None, reach
        padding = reach
        system = replace(system, collider=backend)
    else:
        reach = 2 * radius
        member_radii, query_radii = radii, radii
        padding = radius
    cell_size = jnp.maximum(backend.cell_size, reach / stencil_reach)
    geometry = system.domain.update_bounds(state.pos, system, padding=padding)
    perm, starts, ends, counts, overflow = _partition_tile_counts(
        state.pos,
        geometry,
        cell_size,
        backend.neighbor_mask,
        prune_cells=backend._prune_cells,
        member_radii=member_radii,
        query_radii=query_radii,
    )
    if bool(overflow):
        raise ValueError("Reference cell grid exceeds the supported hash range")
    stencil_size = backend.neighbor_mask.shape[0]
    if backend.clump_aware:
        from ._clump_exclusions import build_clump_index, clump_exclusion_ranges

        index = build_clump_index(perm, state.clump_id)
        own_start, own_end = clump_exclusion_ranges(index, state.clump_id, starts, ends)
        eligible = ends - starts - (own_end - own_start)
        counts = eligible // PAIR_UNROLL + (eligible % PAIR_UNROLL != 0)
        _, inverse, sizes = np.unique(
            np.asarray(state.clump_id), return_inverse=True, return_counts=True
        )
        available = n - sizes[inverse]
        floor = int(np.minimum(available, stencil_size).sum(dtype=np.int64))
        upper = int(available.sum(dtype=np.int64))
    else:
        floor = n * min(stencil_size, n)
        upper = n * n
    # Host int64 also handles dense reference states beyond the device index range.
    required = int(np.asarray(counts).sum(dtype=np.int64))
    capacity = min(max((3 * required + 1) // 2, floor), upper)
    validate_cell_options(capacity, backend.clump_aware, True)
    return capacity


def align_cell_capacities(systems: list[System]) -> list[System]:
    """Give stacked automatic pools one shared capacity without resizing explicit pools."""
    backends = [getattr(s.collider, "secondary_collider", s.collider) for s in systems]
    automatic = [getattr(backend, "_auto_capacity", False) for backend in backends]
    if not any(automatic):
        return systems
    capacities = [getattr(backend, "tile_capacity", None) for backend in backends]
    if not all(automatic) and len(set(capacities)) != 1:
        return systems
    capacity = max(cast(list[int], capacities))
    result = []
    for system, backend in zip(systems, backends):
        backend = cast("DynamicCellList | DynamicMultiCellList", backend)
        backend = replace(
            backend, tile_capacity=capacity, _auto_capacity=all(automatic)
        )
        collider = (
            replace(cast("NeighborList", system.collider), secondary_collider=backend)
            if hasattr(system.collider, "secondary_collider")
            else backend
        )
        result.append(replace(system, collider=collider))
    return result
