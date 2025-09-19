# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Tuple

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("reflect")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ReflectDomain(Domain):
    """
    A `Domain` implementation that enforces reflective boundary conditions.

    Particles that attempt to move beyond the defined `box_size` will have their
    positions reflected back into the box and their velocities reversed in the
    direction normal to the boundary.

    Notes
    -----
    - The reflection occurs at the boundaries defined by `anchor` and `anchor + box_size`.
    """

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        r"""
        Computes the displacement vector between two particles.

        In a reflective domain, the displacement is simply the direct vector difference.

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
        Applies reflective boundary conditions to particles.

        Particles are checked against the domain boundaries.
        If a particle attempts to move beyond a boundary, its position is reflected
        back into the box, and its velocity component normal to that boundary is reversed.

        .. math::
            l &= a + R \\\\
            u &= a + B - R \\\\
            v' &= \\begin{cases} -v & \\text{if } r < l \\text{ or } r > u \\\\ v & \\text{otherwise} \\end{cases} \\\\
            r' &= \\begin{cases} 2l - r & \\text{if } r < l \\\\ r & \\text{otherwise} \\end{cases} \\\\
            r'' &= \\begin{cases} 2u - r' & \\text{if } r' > u \\\\ r' & \\text{otherwise} \\end{cases}
            r = r''

        where:
            - :math:`r` is the current particle position (:attr:`state.pos`)
            - :math:`v` is the current particle velocity (:attr:`state.vel`)
            - :math:`a` is the domain anchor (:attr:`system.domain.anchor`)
            - :math:`B` is the domain box size (:attr:`system.domain.box_size`)
            - :math:`R` is the particle radius (:attr:`state.rad`)
            - :math:`l` is the lower boundary for the particle center
            - :math:`u` is the upper boundary for the particle center

        TO DO: Ensure correctness when adding different types of shapes and angular vel

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            The updated `State` object with reflected positions and velocities,
            and the `System` object.
        """
        lo = system.domain.anchor + state.rad[:, None]
        hi = system.domain.anchor + system.domain.box_size - state.rad[:, None]

        over_lo = jnp.maximum(0.0, lo - state.pos)
        over_hi = jnp.maximum(0.0, state.pos - hi)

        x_ref = state.pos + 2.0 * over_lo - 2.0 * over_hi

        hit = jnp.logical_or((over_lo > 0), (over_hi > 0))
        sign = 1.0 - 2.0 * hit
        v_ref = state.vel * sign

        return replace(state, pos=x_ref, vel=v_ref), system


__all__ = ["ReflectDomain"]
