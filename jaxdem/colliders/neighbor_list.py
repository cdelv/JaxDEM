# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Neighbor List Collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast
from functools import partial

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from . import Collider, valid_interaction_mask

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@Collider.register("NeighborList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NeighborList(Collider):
    r"""
    Implementation of a Verlet neighbor list collider.

    This collider caches a list of neighbors for every particle, significantly reducing
    the number of distance calculations required at each time step. The list is only
    rebuilt when the maximum displacement of any particle exceeds half of the specified
    ``skin`` distance.

    Complexity
    ----------
    - Time: :math:`O(N)` between rebuilds. Rebuild complexity depends on the underlying
      ``cell_list`` collider (typically :math:`O(N \log N)`).

    Notes
    -----
    You must provide a non-zero ``skin`` (e.g., 0.1 * radius) for this collider to
    be efficient. If ``skin = 0``, the list is rebuilt every step, which is
    computationally expensive.
    """

    cell_list: Collider
    """The underlying collider used to build the list via ``create_neighbor_list``."""

    neighbor_list: jax.Array
    """Shape (N, max_neighbors). Contains the IDs of neighboring particles, padded with -1."""

    old_pos: jax.Array
    """Shape (N, dim). Positions of particles at the last build time."""

    n_build_times: jax.Array
    """Counter for how many times the list has been rebuilt."""

    cutoff: jax.Array
    """The interaction radius (force cutoff)."""

    skin: jax.Array
    """
    Buffer distance. The list is built with ``radius = cutoff + skin`` and
    rebuilt when ``max_displacement > skin / 2``.
    """

    overflow: jax.Array
    """Boolean flag indicating if the neighbor list overflowed during build."""

    max_neighbors: int = field(metadata={"static": True})
    """Static buffer size for the neighbor list."""

    @classmethod
    def Create(
        cls,
        state: State,
        cutoff: float,
        skin: float = 0.05,
        max_neighbors: Optional[int] = None,
        number_density: float = 1.0,
        safety_factor: float = 1.2,
        secondary_collider_type: str = "CellList",
        secondary_collider_kw: Optional[dict[str, Any]] = None,
    ) -> Self:
        r"""
        Creates a NeighborList collider.

        Parameters
        ----------
        state : State
            The initial simulation state used to determine system dimensions and
            particle count.
        cutoff : float
            The physical interaction cutoff radius.
        skin : float, default 0.05
            The buffer distance added to the cutoff for the neighbor list.
            **Must be > 0.0 for performance.**
        max_neighbors : int, optional
            Maximum number of neighbors to store per particle. If not provided,
            it is estimated from the ``number_density``.
        number_density : float, default 1.0
            Number density of the system used to estimate ``max_neighbors`` if
            not explicitly provided.
        safety_factor : float, default 1.2
            Multiplier applied to the estimated number of neighbors to account
            for fluctuations in local density.
        secondary_collider_type : str, default "CellList"
            Registered collider type used internally to build the neighbor lists.
        secondary_collider_kw : dict[str, Any], optional
            Keyword arguments passed to the constructor of the internal collider.
            If None, ``cell_size`` is set to ``cutoff + skin``.

        Returns
        -------
        NeighborList
            A configured NeighborList collider instance.
        """
        skin *= cutoff
        list_cutoff = cutoff + skin

        if max_neighbors is None:  # estimate max_neighbors if it is not provided
            nl_volume = (
                jnp.pi
                * (safety_factor * list_cutoff) ** state.dim
                * ((1) if state.dim == 2 else (4 / 3))
            )
            max_neighbors = max(int(nl_volume * number_density), 10)

        if (
            secondary_collider_type.lower() == "celllist"
            or secondary_collider_type.lower() == "staticcelllist"
            and secondary_collider_kw is None
        ):
            secondary_collider_kw = dict(state=state, cell_size=list_cutoff)

        if secondary_collider_kw is None:
            secondary_collider_kw = dict()
        else:
            secondary_collider_kw = dict(secondary_collider_kw)

        cl = Collider.create(secondary_collider_type, **secondary_collider_kw)

        # Initialize buffers
        # We start with current positions. The n_build_times=0 flag will
        # force an immediate rebuild in the first compute_force call.
        current_pos = state.pos
        dummy_nl = jnp.full((state.N, max_neighbors), -1, dtype=int)

        return cls(
            cell_list=cl,
            neighbor_list=dummy_nl,
            old_pos=current_pos,
            n_build_times=jnp.array(0, dtype=int),
            cutoff=jnp.asarray(cutoff, dtype=float),
            skin=jnp.asarray(skin, dtype=float),
            overflow=jnp.asarray(False, dtype=bool),
            max_neighbors=int(max_neighbors),
        )

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="NeighborList.create_neighbor_list")
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple[State, System, jax.Array, jax.Array]:
        r"""
        Returns the cached neighbor list from this collider.

        This method does not rebuild the neighbor list. It returns the current
        cached ``neighbor_list`` and ``overflow`` flag stored in the collider.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.
        cutoff : float
            Ignored; provided for API compatibility.
        max_neighbors : int
            Ignored; provided for API compatibility.

        Returns
        -------
        Tuple[State, System, jax.Array, jax.Array]
            A tuple containing:
            - state: The simulation state.
            - system: The simulation system.
            - neighbor_list: The cached neighbor list of shape (N, max_neighbors).
            - overflow: Boolean flag indicating if the list overflowed during the
              last build.

        Notes
        -----
        - The returned neighbor indices refer to the internal particle ordering
          established during the most recent rebuild inside ``compute_force``.
        """
        collider = cast(NeighborList, system.collider)
        return state, system, collider.neighbor_list, collider.overflow

    @staticmethod
    @partial(jax.named_call, name="NeighborList._rebuild")
    def _rebuild(
        collider: NeighborList, state: State, system: System
    ) -> Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""
        Internal method to rebuild the neighbor list using the secondary collider.

        Parameters
        ----------
        collider : NeighborList
            The current collider instance.
        state : State
            The current simulation state.
        system : System
            The simulation system.

        Returns
        -------
        Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]
            A tuple containing:
            - Sorted State
            - New neighbor list indices
            - New reference positions for displacement tracking
            - Incremented build counter
            - Overflow flag
        """
        list_cutoff = collider.cutoff + collider.skin

        # Create a view of the system using the inner collider
        system.collider = collider.cell_list

        # 1. Get neighbors using the spatial partitioner
        # Returns: Sorted State, ..., Neighbors Indices (pointing to Sorted State)
        (
            sorted_state,
            _,
            sorted_nl_indices,
            overflow_flag,
        ) = collider.cell_list.create_neighbor_list(
            state, system, list_cutoff, collider.max_neighbors
        )

        # return the sorted state to avoid having to un-sort the neighbor list
        return (
            sorted_state,
            sorted_nl_indices,
            sorted_state.pos,
            collider.n_build_times + 1,
            overflow_flag,
        )

    @staticmethod
    @jax.jit(donate_argnames=("state", "system"))
    @partial(jax.named_call, name="NeighborList.compute_force")
    def compute_force(state: State, system: System) -> Tuple[State, System]:
        r"""
        Computes total forces acting on each particle, rebuilding the neighbor list if necessary.

        This method checks if any particle has moved enough to trigger a rebuild
        (displacement > skin/2). If so, it invokes the internal spatial partitioner
        to refresh the neighbor list. It then sums force contributions using the
        cached list.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated ``State`` object with computed forces
            and the updated ``System`` object (with refreshed collider cache).

        Note
        ----
        - This method donates ``state`` and ``system``.
        """
        iota = jax.lax.iota(dtype=int, size=state.N)  # should this be cached?
        collider = cast(NeighborList, system.collider)

        # 1. Check Displacement & Trigger Rebuild
        # disp = system.domain.displacement(state.pos, collider.old_pos, system)
        disp = state.pos - collider.old_pos  # this should not be a periodic distance
        max_disp_sq = jnp.max(jnp.sum(disp * disp, axis=-1))
        trigger_dist_sq = collider.skin**2 / 4

        # Force rebuild if displacement is large OR if this is the first step (count == 0)
        should_rebuild = (max_disp_sq > trigger_dist_sq) + (collider.n_build_times == 0)

        def rebuild_branch(
            operands: Tuple[State, System, NeighborList],
        ) -> Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            s, sys, col = operands
            return col._rebuild(col, s, sys)

        def no_rebuild_branch(
            operands: Tuple[State, System, NeighborList],
        ) -> Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            _, _, col = operands
            return (
                state,
                col.neighbor_list,
                col.old_pos,
                col.n_build_times,
                col.overflow,
            )

        state, nl, old_pos, n_build, overflow = jax.lax.cond(
            should_rebuild > 0,
            rebuild_branch,
            no_rebuild_branch,
            (state, system, collider),
        )

        # 2. Compute Forces
        # Pre-calculate contact points in global frame for torque
        pos_p_global = state.q.rotate(state.q, state.pos_p)
        pos = state.pos_c + pos_p_global

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
                valid = valid * valid_interaction_mask(
                    state.clump_id[i],
                    state.clump_id[safe_j],
                    state.bond_id[i],
                    state.bond_id[safe_j],
                    system.interact_same_bond_id,
                )

                f, t = system.force_model.force(i, safe_j, pos, state, system)
                return f * valid, t * valid

            forces, torques = jax.vmap(per_neighbor_force)(neighbors)

            f_sum = jnp.sum(forces, axis=0)
            # Add contact torque: T_total = Sum(T_ij) + (r_i x F_total)
            t_sum = jnp.sum(torques, axis=0) + jnp.cross(pos_pi, f_sum)

            return f_sum, t_sum

        # Vmap over particle IDs [0, 1, ..., N]
        state.force, state.torque = jax.vmap(per_particle_force)(iota, pos_p_global, nl)

        # Update collider cache
        system.collider = replace(
            collider,
            neighbor_list=nl,
            old_pos=old_pos,
            n_build_times=n_build,
            overflow=overflow,
        )

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="NeighborList.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        r"""
        Computes the potential energy associated with each particle using the cached neighbor list.

        This method iterates over the cached neighbors for each particle and sums
        the potential energy contributions computed by the ``system.force_model``.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            One-dimensional array containing the total potential energy contribution for each particle.
        """
        iota = jax.lax.iota(dtype=int, size=state.N)

        collider = cast(NeighborList, system.collider)

        def per_particle_energy(i: jax.Array) -> jax.Array:
            neighbors = collider.neighbor_list[i]

            def per_neighbor_energy(j_id: jax.Array) -> jax.Array:
                valid = j_id != -1
                safe_j = jnp.maximum(j_id, 0)
                valid = valid * valid_interaction_mask(
                    state.clump_id[i],
                    state.clump_id[safe_j],
                    state.bond_id[i],
                    state.bond_id[safe_j],
                    system.interact_same_bond_id,
                )
                e = system.force_model.energy(i, safe_j, state.pos, state, system)
                return e * valid

            # Sum energies and divide by 2 (double counting in neighbor list)
            return 0.5 * jnp.sum(jax.vmap(per_neighbor_energy)(neighbors))

        return jax.vmap(per_particle_energy)(iota)
