# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Unbounded (free) simulation domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Tuple

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("free")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class FreeDomain(Domain):
    """
    A `Domain` implementation representing an unbounded, "free" space.

    In a `FreeDomain`, there are no explicit boundary conditions applied to
    particles. Particles can move indefinitely in any direction, and the
    concept of a "simulation box" is only used to define the bounding box of the system.

    Notes
    -----
    - The `box_size` and `anchor` attributes are dynamically updated in
      the `shift` method to encompass all particles. Some hashing tools require the domain size.
    """

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        r"""
        Computes the displacement vector between two particles.

        In a free domain, the displacement is simply the direct vector difference
        between the particle positions.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle :math:`r_i`.
        rj : jax.Array
            Position vector of the second particle :math:`r_j`.
        _ : System
            The system object.

        Returns
        -------
        jax.Array
            The direct displacement vector :math:`r_i - r_j`.
        """
        return ri - rj

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Updates the `System`'s domain `anchor` and `box_size` to encompass all particles. Does not apply any transformations to the state.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The current system configuration.

        Returns
        -------
        Tuple[State, System]
            The original `State` object (unchanged) and the `System` object
            with updated `domain.anchor` and `domain.box_size`.
        """
        p_min = jnp.min(state.pos - state.rad[..., None], axis=-2)
        p_max = jnp.max(state.pos + state.rad[..., None], axis=-2)
        domain = replace(system.domain, box_size=p_max - p_min, anchor=p_min)
        return state, replace(system, domain=domain)


__all__ = ["FreeDomain"]
