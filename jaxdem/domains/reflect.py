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


@Domain.register("reflect")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
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
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="ReflectDomain.apply")
    def apply(state: "State", system: "System") -> Tuple["State", "System"]:
        r"""
        Applies reflective boundary conditions to particles.

        Particles are checked against the domain boundaries. If a particle attempts
        to move beyond a boundary, it is reflected. The reflection is governed by the
        impulse-momentum equations for rigid bodies.

        **Velocity Update (Impulse)**

        .. math::
            \vec{v}' &= \vec{v} + \frac{1}{m}\vec{J} \\
            \vec{\omega}' &= \vec{\omega} + \mathbf{I}^{-1} (\vec{r}_{p} \times \vec{J})

        where the impulse vector :math:`J` is:

        .. math::
            \vec{J} = \frac{-(1+e)(\vec{v}_{contact} \cdot \hat{n})}{\frac{1}{m} + [\mathbf{I}^{-1} (\vec{r}_{p} \times \hat{n})] \cdot (\vec{r}_{p} \times \hat{n})} \hat{n}

        and the velocity of the contact point :math:`\vec{v}_{contact}` is:

        .. math::
            \vec{v}_{contact} = \vec{v} + \vec{\omega} \times \vec{r}_{p}

        **Position Update**

        Finally, the particle is moved out of the boundary by reflecting its position
        based on the penetration depth :math:`\delta`:

        .. math::
            \vec{r}_c' = \vec{r}_c + 2 \delta \hat{n}

        **Definitions**

        - :math:`\vec{r}_c`: Particle center of mass position (:attr:`jaxdem.State.pos_c`).
        - :math:`\vec{r}_{p}`: Vector from COM to contact sphere in the lab frame (:attr:`jaxdem.State.pos_p`).
        - :math:`\vec{v}`: Particle linear velocity (:attr:`jaxdem.State.vel`).
        - :math:`\vec{\omega}`: Particle angular velocity (:attr:`jaxdem.State.angVel`).
        - :math:`\hat{n}`: Boundary normal vector (pointing into the domain).
        - :math:`\delta`: Penetration depth (positive value).
        - :math:`e`: Coefficient of restitution. We assume the collision is ellastic  :math:`e = 1`:

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

        Reference
        ----------
        https://www.myphysicslab.com/engine2D/collision-en.html
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

        body_over_lo = jax.ops.segment_max(over_lo, state.ID, num_segments=state.N)
        body_over_hi = jax.ops.segment_max(over_hi, state.ID, num_segments=state.N)

        over_lo = body_over_lo[state.ID]
        over_hi = body_over_hi[state.ID]

        # hit = jnp.logical_or(over_lo > 0, over_hi > 0)
        hit = ((over_lo > 0) + (over_hi > 0)) > 0
        sign = 1.0 - 2.0 * (hit > 0)
        state.pos_c += 2.0 * (over_lo - over_hi)
        state.vel *= sign
        # state.angVel *= sign # Is this correct?
        return state, system


__all__ = ["ReflectDomain"]
