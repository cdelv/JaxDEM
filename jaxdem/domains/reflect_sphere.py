# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

from __future__ import annotations

import jax

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple
from functools import partial

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("reflectsphere")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ReflectSphereDomain(Domain):
    """
    A `Domain` implementation that enforces reflective boundary conditions only for spheres.
    We have this dedicated version for performance reasons.

    Particles that attempt to move beyond the defined `box_size` will have their
    positions reflected back into the box and their velocities reversed in the
    direction normal to the boundary.

    Notes
    -----
    - The reflection occurs at the boundaries defined by `anchor` and `anchor + box_size`.
    """

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="ReflectSphereDomain.apply")
    def apply(state: "State", system: "System") -> Tuple["State", "System"]:
        r"""
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
            - :math:`r` is the current particle position (:attr:`jaxdem.State.pos`)
            - :math:`v` is the current particle velocity (:attr:`jaxdem.State.vel`)
            - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
            - :math:`B` is the domain box size (:attr:`Domain.box_size`)
            - :math:`R` is the particle radius (:attr:`jaxdem.State.rad`)
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

        Note
        -----
        - This method donates state and system
        - Only works for states with *ONLY* spheres.
        """
        pos = state.pos
        lo = system.domain.anchor + state.rad[:, None]
        hi = system.domain.anchor + system.domain.box_size - state.rad[:, None]

        # over_lo = jnp.maximum(0.0, lo - state.pos)
        over_lo = lo - pos
        over_lo *= over_lo > 0

        # over_hi = jnp.maximum(0.0, state.pos - hi)
        over_hi = pos - hi
        over_hi *= over_hi > 0

        # hit = jnp.logical_or(over_lo > 0, over_hi > 0)
        hit = ((over_lo > 0) + (over_hi > 0)) > 0
        sign = 1.0 - 2.0 * (hit > 0)
        state.pos_c += 2.0 * (over_lo - over_hi)
        state.vel *= sign
        return state, system


__all__ = ["ReflectSphereDomain"]
