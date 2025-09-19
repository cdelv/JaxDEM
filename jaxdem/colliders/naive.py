# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Naive :math:`O(N^2)` collider implementation."""

from __future__ import annotations

import jax

from dataclasses import dataclass, replace
from typing import Tuple, TYPE_CHECKING

from . import Collider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Collider.register("naive")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class NaiveSimulator(Collider):
    r"""
    Implementation that computes forces and potential energies using a naive :math:`O(N^2)` all-pairs interaction loop.

    Notes
    -----
    Due to its :math:`O(N^2)` complexity, `NaiveSimulator` is suitable for simulations
    with a relatively small number of particles. For larger systems, a more
    efficient spatial partitioning collider should be used. However, thhis collider should be the fastest
    option for small systems (:math:`<1k-5k` spheres depending on the GPU).
    """

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        r"""
        Computes the total force on each particle using a naive :math:`O(N^2)` all-pairs loop.

        This method iterates over all particle pairs (i, j) and sums the forces
        computed by the `system.force_model`.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated `State` object with computed accelerations
            and the `System` object.

        """
        rng = jax.lax.iota(dtype=int, size=state.N)
        return jax.vmap(
            lambda i, j, st, sys: jax.vmap(
                sys.force_model.energy, in_axes=(None, 0, None, None)
            )(i, j, st, sys).sum(axis=0),
            in_axes=(0, None, None, None),
        )(rng, rng, state, system)

    @staticmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        r"""
        Computes the total potential energy of the system using a naive :math:`O(N^2)` all-pairs loop.

        This method sums the potential energy contributions from all particle pairs (i, j)
        as computed by the `system.force_model`.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            A scalar JAX array representing the total potential energy of the system.
        """
        rng = jax.lax.iota(dtype=int, size=state.N)
        accel = state.accel + (
            jax.vmap(
                lambda i, j, st, sys: jax.vmap(
                    sys.force_model.force, in_axes=(None, 0, None, None)
                )(i, j, st, sys).sum(axis=0),
                in_axes=(0, None, None, None),
            )(rng, rng, state, system)
            / state.mass[:, None]
        )
        state = replace(state, accel=accel)
        return state, system


__all__ = ["NaiveSimulator"]
