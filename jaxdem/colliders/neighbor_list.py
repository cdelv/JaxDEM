# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Neighbor List Collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field, replace
from typing import Tuple, Any, Optional, TYPE_CHECKING, cast
from functools import partial

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from . import Collider, DynamicCellList

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@Collider.register("NeighborList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NeighborList(Collider):
    r"""
    Verlet Neighbor List collider.

    This collider caches a list of neighbors for every particle. It only rebuilds
    the list when particles have moved more than half the 'skin' distance.

    **Performance Note:** You must provide a non-zero `skin` (e.g., 0.1 * radius)
    for this collider to be efficient. If `skin=0`, it rebuilds every step.

    Attributes
    ----------
    cell_list : DynamicCellList
        The underlying spatial partitioner used to build the list.
    neighbor_list : jax.Array
        Shape (N, max_neighbors). Contains the IDs of neighboring particles.
        padded with -1.
    old_pos : jax.Array
        Shape (N, dim). Positions of particles at the last build time.
    n_build_times : int
        Counter for how many times the list has been rebuilt.
    cutoff : float
        The interaction radius (force cutoff).
    skin : float
        Buffer distance. The list is built with `radius = cutoff + skin` and
        rebuilt when `max_displacement > skin / 2`.
    max_neighbors : int
        Static buffer size for the neighbor list.
    """

    cell_list: DynamicCellList
    neighbor_list: jax.Array
    old_pos: jax.Array
    n_build_times: int
    cutoff: float
    skin: float
    max_neighbors: int = field(metadata={"static": True})

    @classmethod
    def Create(
        cls,
        state: "State",
        cutoff: float,
        skin: float = 0.05,
        max_neighbors: int = 24,
        cell_size: Optional[float] = None,
    ) -> Self:
        r"""
        Creates a NeighborList collider.

        Parameters
        ----------
        state : State
            Initial simulation state.
        cutoff : float
            The physical interaction cutoff radius.
        skin : float, default 0.0
            The buffer distance. **Must be > 0.0 for performance.**
        max_neighbors : int, default 30
            Maximum neighbors to store per particle.
        cell_size : float, optional
            Override for the underlying cell list size.
        """
        skin *= cutoff
        list_cutoff = cutoff + skin
        if cell_size is None:
            cell_size = list_cutoff

        # Initialize inner CellList
        cl = DynamicCellList.Create(state, cell_size=cell_size)

        # Initialize buffers
        # We start with current positions. The n_build_times=0 flag will
        # force an immediate rebuild in the first compute_force call.
        current_pos = state.pos
        dummy_nl = jnp.full((state.N, max_neighbors), -1, dtype=int)

        return cls(
            cell_list=cl,
            neighbor_list=dummy_nl,
            old_pos=current_pos,
            n_build_times=0,
            cutoff=float(cutoff),
            skin=float(skin),
            max_neighbors=int(max_neighbors),
        )

    @staticmethod
    def _rebuild(
        collider: "NeighborList", state: "State", system: "System"
    ) -> Tuple[jax.Array, jax.Array, int]:
        """
        Static internal method to rebuild the neighbor list.
        """
        list_cutoff = collider.cutoff + collider.skin

        # Create a view of the system using the inner collider
        inner_system = replace(system, collider=collider.cell_list)

        # 1. Get neighbors using the spatial partitioner
        # Returns: Sorted State, ..., Neighbors Indices (pointing to Sorted State)
        (
            sorted_state,
            _,
            sorted_nl_indices,
            _,
        ) = collider.cell_list.create_neighbor_list(
            state, inner_system, list_cutoff, collider.max_neighbors
        )

        # return the sorted state to avoid having to un-sort the neighbor list
        return (
            sorted_state,
            sorted_nl_indices,
            sorted_state.pos,
            collider.n_build_times + 1,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        iota = jax.lax.iota(dtype=int, size=state.N)  # should this be cached?

        collider = cast(NeighborList, system.collider)

        # 1. Check Displacement & Trigger Rebuild
        disp = system.domain.displacement(state.pos, collider.old_pos, system)
        max_disp_sq = jnp.max(jnp.sum(disp**2, axis=-1))

        trigger_dist_sq = (collider.skin * 0.5) ** 2

        # Force rebuild if displacement is large OR if this is the first step (count == 0)
        should_rebuild = (max_disp_sq > trigger_dist_sq) + (collider.n_build_times == 0)

        def rebuild_branch(
            operands: Tuple[Any, Any, "NeighborList"],
        ) -> Tuple[jax.Array, jax.Array, int]:
            s, sys, col = operands
            return col._rebuild(col, s, sys)

        def no_rebuild_branch(
            operands: Tuple[Any, Any, "NeighborList"],
        ) -> Tuple[jax.Array, jax.Array, int]:
            _, _, col = operands
            return state, col.neighbor_list, col.old_pos, col.n_build_times

        state, nl, old_pos, n_build = jax.lax.cond(
            should_rebuild > 0,
            rebuild_branch,
            no_rebuild_branch,
            (state, system, collider),
        )

        # 2. Compute Forces
        # Pre-calculate contact points in global frame for torque
        pos_p_global = state.q.rotate(state.q, state.pos_p)

        def per_particle_force(
            i: jax.Array, pos_pi: jax.Array, neighbors: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            # i: ID of the current particle
            # pos_pi: vector from COM to surface for particle i
            # neighbors: array of neighbor IDs

            def per_neighbor_force(j_id: jax.Array) -> Tuple[jax.Array, jax.Array]:
                # We mask computations for padding (-1)
                valid = j_id != -1
                safe_j = jnp.maximum(j_id, 0)

                f, t = system.force_model.force(i, safe_j, state, system)
                return f * valid, t * valid

            forces, torques = jax.vmap(per_neighbor_force)(neighbors)

            f_sum = jnp.sum(forces, axis=0)
            # Add contact torque: T_total = Sum(T_ij) + (r_i x F_total)
            t_sum = jnp.sum(torques, axis=0) + jnp.cross(pos_pi, f_sum)

            return f_sum, t_sum

        # Vmap over particle IDs [0, 1, ..., N]
        total_force, total_torque = jax.vmap(per_particle_force)(iota, pos_p_global, nl)

        # Aggregate over particles in clumps
        state.force += total_force
        state.torque += total_torque
        state.torque = jax.ops.segment_sum(state.torque, state.ID, num_segments=state.N)
        state.force = jax.ops.segment_sum(state.force, state.ID, num_segments=state.N)
        state.force = state.force[state.ID]
        state.torque = state.torque[state.ID]

        # Update collider cache
        system.collider = replace(
            collider, neighbor_list=nl, old_pos=old_pos, n_build_times=n_build
        )

        return state, system

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        iota = jax.lax.iota(dtype=int, size=state.N)

        collider = cast(NeighborList, system.collider)

        def per_particle_energy(i: jax.Array) -> jax.Array:
            neighbors = collider.neighbor_list[i]

            def per_neighbor_energy(j_id: jax.Array) -> jax.Array:
                valid = j_id != -1
                safe_j = jnp.maximum(j_id, 0)
                e = system.force_model.energy(i, safe_j, state, system)
                return e * valid

            # Sum energies and divide by 2 (double counting in neighbor list)
            return 0.5 * jnp.sum(jax.vmap(per_neighbor_energy)(neighbors))

        return jax.vmap(per_particle_energy)(iota)
