# SPDX-License-Identifier: BSD-3-Clause
"""Shared options for dynamic and static cell traversal."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from ..state import State


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
    validate_cell_options(capacity, False, static)
    return capacity
