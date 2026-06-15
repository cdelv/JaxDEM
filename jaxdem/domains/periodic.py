# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Periodic boundary-condition domain."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("periodic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PeriodicDomain(Domain):
    """A `Domain` implementation that enforces periodic boundary conditions.

    Particles that move out of one side of the simulation box re-enter from the
    opposite side. The displacement vector between particles is computed using the minimum image convention.
    """

    @property
    def periodic(self) -> bool:
        """Whether the domain enforces periodic boundary conditions."""
        return True

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="PeriodicDomain.displacement")
    def displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        r"""Computes the minimum image displacement vector between two particles :math:`r_i` and :math:`r_j`.

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
                & r_{ij} = (r_i - a) - (r_j - a) \\
                & r_{ij} = r_{ij} - B \cdot \text{round}(r_{ij}/B)

            where:
                - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
                - :math:`B` is the domain box size (:attr:`Domain.box_size`)

        """
        rij = ri - rj
        return rij - system.domain.box_size * jnp.round(
            rij / system.domain.box_size
        )

    @staticmethod
    @jax.jit(inline=True)
    def _displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        rij = ri - rj
        return rij - system.domain.box_size * jnp.round(
            rij * system.domain.inv_box_size
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="PeriodicDomain.shift")
    def shift(state: State, system: System) -> tuple[State, System]:
        r"""Wraps particles back into the primary simulation box.

        .. math::
            r = r - B \cdot \text{floor}((r - a)/B)

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
        # Wrap by the clump representative (the body center `pos_c`, shared by all
        # members of a clump) rather than per-sphere positions: per-sphere wrapping
        # would give clump members straddling a boundary different shifts and tear
        # the clump apart.
        state.pos_c -= system.domain.box_size[..., None, :] * jnp.floor(
            (state.pos_c - system.domain.anchor[..., None, :])
            / system.domain.box_size[..., None, :]
        )
        return state, system


__all__ = ["PeriodicDomain"]
