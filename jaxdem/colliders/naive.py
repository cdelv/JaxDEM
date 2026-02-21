# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Naive :math:`O(N^2)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from ..utils.linalg import cross
from . import Collider, valid_interaction_mask

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Collider.register("naive")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NaiveSimulator(Collider):
    r"""
    Implementation that computes forces and potential energies using a naive :math:`O(N^2)` all-pairs interaction loop.

    Notes
    -----
    Due to its :math:`O(N^2)` complexity, `NaiveSimulator` is suitable for simulations
    with a relatively small number of particles. For larger systems, a more
    efficient spatial partitioning collider should be used. However, this collider should be the fastest
    option for small systems (:math:`<1k-5k` spheres depending on the GPU).
    """

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="NaiveSimulator.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        r"""
        Computes the potential energy associated with each particle using a naive :math:`O(N^2)` all-pairs loop.

        This method iterates over all particle pairs (i, j) and sums the potential energy
        contributions computed by the ``system.force_model``.

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
        pos = state.pos

        def row_energy(i: jax.Array, st: State, sys: System) -> jax.Array:
            e_ij = jax.vmap(
                sys.force_model.energy, in_axes=(None, 0, None, None, None)
            )(i, iota, pos, st, sys)
            mask = valid_interaction_mask(
                st.clump_id[i],
                st.clump_id,
                st.bond_id[i],
                st.bond_id,
                sys.interact_same_bond_id,
            )
            e_ij *= mask
            return 0.5 * e_ij.sum(axis=0)

        return jax.vmap(row_energy, in_axes=(0, None, None))(iota, state, system)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="NaiveSimulator.create_neighbor_list")
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple[State, System, jax.Array, jax.Array]:
        r"""
        Computes a neighbor list using a naive :math:`O(N^2)` all-pairs search.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.
        cutoff : float
            The interaction radius (force cutoff).
        max_neighbors : int
            Maximum number of neighbors to store per particle.

        Returns
        -------
        Tuple[State, System, jax.Array, jax.Array]
            A tuple containing:
            - state: The simulation state.
            - system: The simulation system.
            - neighbor_list: Array of shape (N, max_neighbors) containing neighbor indices.
            - overflow: Boolean flag indicating if any particle exceeded ``max_neighbors``.
        """
        # Preserve documented semantics: always return shape (N, max_neighbors),
        # padded with -1. But `lax.top_k` requires k <= len(candidates), so we
        # clamp internally when `max_neighbors` exceeds N.
        if max_neighbors == 0:
            empty = jnp.empty((state.N, 0), dtype=int)
            return state, system, empty, jnp.asarray(False)

        iota = jax.lax.iota(dtype=int, size=state.N)
        pos = state.pos
        cutoff_sq = jnp.asarray(cutoff, dtype=pos.dtype) ** 2

        def per_particle(i: jax.Array) -> Tuple[jax.Array, jax.Array]:
            dr = system.domain.displacement(pos[i], pos, system)  # (N, dim)
            dist_sq = jnp.sum(dr * dr, axis=-1)
            valid = valid_interaction_mask(
                state.clump_id[i],
                state.clump_id,
                state.bond_id[i],
                state.bond_id,
                system.interact_same_bond_id,
            ) * (dist_sq <= cutoff_sq)
            num_neighbors = jnp.sum(valid)
            overflow_flag = num_neighbors > max_neighbors
            candidates = jnp.where(valid, iota, -1)
            k_eff = min(max_neighbors, candidates.shape[0])
            topk = jax.lax.top_k(candidates, k_eff)[0]
            # If max_neighbors > N, pad back to the requested buffer size.
            if k_eff < max_neighbors:
                topk = jnp.concatenate(
                    [topk, jnp.full((max_neighbors - k_eff,), -1, dtype=topk.dtype)]
                )
            return topk, overflow_flag

        nl, overflows = jax.vmap(per_particle)(iota)
        return state, system, nl, jnp.any(overflows)

    @staticmethod
    @jax.jit(donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="NaiveSimulator.compute_force")
    def compute_force(state: State, system: System) -> Tuple[State, System]:
        r"""
        Computes the total force acting on each particle using a naive :math:`O(N^2)` all-pairs loop.

        This method sums the force contributions from all particle pairs (i, j)
        as computed by the ``system.force_model`` and updates the particle forces.

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
            and the unmodified ``System`` object.

        Note
        -----
        - This method donates state and system
        """
        iota = jax.lax.iota(dtype=int, size=state.N)
        pos_p = state.q.rotate(state.q, state.pos_p)
        pos = state.pos_c + pos_p

        def per_particle_i(
            i: jax.Array, pos_pi: jax.Array, st: State, sys: System
        ) -> Tuple[jax.Array, jax.Array]:
            res_f, res_t = sys.force_model.force(i, iota, pos, st, sys)
            mask = valid_interaction_mask(
                st.clump_id[i],
                st.clump_id,
                st.bond_id[i],
                st.bond_id,
                sys.interact_same_bond_id,
            )[..., None]
            f_i = jnp.sum(res_f * mask, axis=0)
            t_i = jnp.sum(res_t * mask, axis=0) + cross(pos_pi, f_i)
            return f_i, t_i

        state.force, state.torque = jax.vmap(
            per_particle_i, in_axes=(0, 0, None, None)
        )(iota, pos_p, state, system)

        return state, system


__all__ = ["NaiveSimulator"]
