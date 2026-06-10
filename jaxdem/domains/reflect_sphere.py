# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

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


@Domain.register("reflectsphere")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ReflectSphereDomain(Domain):
    """A `Domain` implementation that enforces reflective boundary conditions only for spheres.
    We have this dedicated version for performance reasons.

    Particles that attempt to move beyond the defined `box_size` will have their
    positions reflected back into the box and their velocities reversed in the
    direction normal to the boundary.

    Notes
    -----
    - The reflection occurs at the boundaries defined by `anchor` and `anchor + box_size`.

    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ReflectSphereDomain.apply")
    def apply(state: State, system: System) -> tuple[State, System]:
        r"""Applies reflective boundary conditions to particles.

        Particles are checked against the domain boundaries.
        If a particle attempts to move beyond a boundary, its position is reflected
        back into the box, and its velocity component normal to that boundary is reversed.

        .. math::
            l &= a + R \\
            u &= a + B - R \\
            v' &= \begin{cases} -v & \text{if } r < l \text{ or } r > u \\ v & \text{otherwise} \end{cases} \\
            r' &= \begin{cases} 2l - r & \text{if } r < l \\ r & \text{otherwise} \end{cases} \\
            r'' &= \begin{cases} 2u - r' & \text{if } r' > u \\ r' & \text{otherwise} \end{cases} \\
            r &= r''

        where:
            - :math:`r` is the current particle position (:attr:`jaxdem.State.pos`)
            - :math:`v` is the current particle velocity (:attr:`jaxdem.State.vel`)
            - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
            - :math:`B` is the domain box size (:attr:`Domain.box_size`)
            - :math:`R` is the particle radius (:attr:`jaxdem.State.rad`)
            - :math:`l` is the lower boundary for the particle center
            - :math:`u` is the upper boundary for the particle center

        **Verlet Time-of-Collision Correction**

        Under Verlet integration, the collision time fraction :math:`\alpha \in [0, 1]` is solved exactly using the quadratic equation:

        .. math::
            A \alpha^2 + B \alpha + C = 0

        where:
            - :math:`A = - \frac{1}{2} a_{n} \Delta t^2`
            - :math:`B = - v_{0, n} \Delta t`
            - :math:`C = -\delta - v_{mid, n} \Delta t`
            - :math:`v_{0, n}`: Normal velocity of the particle at the start of the step.
            - :math:`a_{n}`: Normal acceleration of the particle.
            - :math:`v_{mid, n}`: Normal velocity of the particle at the end of the step.
            - :math:`\delta`: Penetration depth (overlap).

        The solution for :math:`\alpha` is given by:

        .. math::
            \alpha = \frac{2 C}{B + \sqrt{B^2 - 4 A C}}

        The velocity at the moment of collision :math:`v_{col}` is then calculated and used to update the pre-collision velocity.

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
        - Only works for states with *ONLY* spheres.

        """
        pos = state.pos_c

        rad = state.rad[:, None]
        lo = system.domain.anchor + rad
        hi = system.domain.anchor + system.domain.box_size - rad

        over_lo = jnp.maximum(0.0, lo - pos)
        over_hi = jnp.maximum(0.0, pos - hi)

        hit = jnp.sign(over_lo + over_hi)



        delta = jnp.maximum(over_lo, over_hi)
        wall_sign = (over_lo > 0).astype(float) - (over_hi > 0).astype(float)

        acc = state.force / state.mass[:, None]
        v_mid = state.vel
        v_start = state.vel - 0.5 * system.dt * acc

        v_start_n = v_start * wall_sign
        acc_n = acc * wall_sign

        d_wall_pos = -delta - system.dt * v_mid * wall_sign
        B_pos = -v_start_n * system.dt
        A_pos = -0.5 * acc_n * system.dt * system.dt

        disc = B_pos * B_pos + 4.0 * A_pos * d_wall_pos
        disc = jnp.maximum(0.0, disc)

        alpha = (
            2.0
            * d_wall_pos
            / jnp.where(B_pos + jnp.sqrt(disc) > 1e-10, B_pos + jnp.sqrt(disc), 1.0)
        )
        alpha = jnp.clip(alpha, 0.0, 1.0)

        v_col = state.vel + (alpha - 0.5) * system.dt * acc
        dv = -2.0 * v_col * hit
        new_vel = state.vel + dv

        state.vel = jnp.where(state.fixed[:, None], state.vel, new_vel)

        displacement = (1.0 - alpha) * system.dt * dv
        displacement = jnp.where(state.fixed[:, None], 0.0, displacement)
        state.pos_c += displacement

        return state, system


__all__ = ["ReflectSphereDomain"]
