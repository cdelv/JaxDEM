# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Naive :math:`O(N^2)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING
from functools import partial

from . import Collider

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
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
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

        def row_energy(i, st, sys):
            j = jax.lax.iota(dtype=int, size=st.N)
            e_ij = jax.vmap(sys.force_model.energy, in_axes=(None, 0, None, None))(i, j, st, sys)
            # Mask out self-interaction (i == j) to avoid counting self-energy
            mask = jnp.not_equal(j, i)
            e_ij = e_ij * mask
            return e_ij.sum(axis=0)

        return jax.vmap(row_energy, in_axes=(0, None, None))(iota, state, system)

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="NaiveSimulator.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
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
        """
        iota = jax.lax.iota(dtype=int, size=state.N)

        def pairwise_accumulate(i, st, sys):
            j = jax.lax.iota(dtype=int, size=st.N)
            forces, torques = jax.vmap(
                sys.force_model.force, in_axes=(None, 0, None, None)
            )(i, j, st, sys)
            # Mask out self-interaction (i == j)
            mask = jnp.not_equal(j, i)[..., None]
            forces = forces * mask
            torques = torques * mask
            return forces.sum(axis=0), torques.sum(axis=0)

        total_force, total_torque = jax.vmap(
            pairwise_accumulate, in_axes=(0, None, None)
        )(iota, state, system)

        state.force += total_force
        state.torque += total_torque

        return state, system


__all__ = ["NaiveSimulator"]
