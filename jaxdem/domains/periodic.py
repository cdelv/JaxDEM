# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Periodic boundary-condition domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Tuple, ClassVar

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("periodic")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class PeriodicDomain(Domain):
    """
    A `Domain` implementation that enforces periodic boundary conditions.

    Particles that move out of one side of the simulation box re-enter from the
    opposite side. The displacement vector between particles is computed using the minimum image convention.

    Notes
    -----
    - This domain type is periodic (`periodic = True`).
    """

    periodic: ClassVar[bool] = True

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
        """
        Computes the minimum image displacement vector between two particles :math:`r_i` and :math:`r_j`.

        For periodic boundary conditions, the displacement is calculated as the
        shortest vector that connects :math:`r_j` to :math:`r_i`, potentially by crossing
        periodic boundaries.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle :math:`r_i`.
        rj : jax.Array
            Position vector of the second particle :math:`r_j`.
        system : System
            The configuration of the simulation, containing the `domain` instance
            with `anchor` and `box_size` for periodicity.

        Returns
        -------
        jax.Array
            The minimum image displacement vector:

            .. math::
                & r_{ij} = (r_i - a) - (r_j - a) \\\\
                & r_{ij} = r_{ij} - B \\cdot \\text{round}(r_{ij}/B)

            where:
                - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
                - :math:`B` is the domain box size (:attr:`Domain.box_size`)
        """
        rij = ri - rj
        return rij - system.domain.box_size * jnp.floor(
            rij / system.domain.box_size + 0.5
        )

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Wraps particles back into the primary simulation box.

        .. math::
            r = r - B \\cdot \\text{floor}((r - a)/B) \\\\

        where:
            - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
            - :math:`B` is the domain box size (:attr:`Domain.box_size`)

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            The updated `State` object with wrapped particle positions, and the
            `System` object.
        """
        pos = state.pos - system.domain.box_size * jnp.floor(
            (state.pos - system.domain.anchor) / system.domain.box_size
        )
        state = replace(state, pos=pos)
        return state, system


__all__ = ["PeriodicDomain"]
