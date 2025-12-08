# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
        e = 0.98
        pos_p_lab = state.q.rotate_back(state.q, state.pos_p)
        pos = state.pos_c + pos_p_lab

        # --- Boundaries ---
        lo = system.domain.anchor + state.rad[:, None]
        hi = system.domain.anchor + system.domain.box_size - state.rad[:, None]

        over_lo = lo - pos
        over_lo *= over_lo > 0
        over_hi = pos - hi
        over_hi *= over_hi > 0

        # --- Position Correction ---
        body_over_lo = jax.ops.segment_max(over_lo, state.ID, num_segments=state.N)
        body_over_hi = jax.ops.segment_max(over_hi, state.ID, num_segments=state.N)
        max_lo = body_over_lo[state.ID]
        max_hi = body_over_hi[state.ID]
        shift = 2.0 * (max_lo - max_hi)
        state.pos_c += shift

        # --- Impulse Calculation ---
        n = jnp.eye(state.dim, state.dim)
        n_prime = jax.vmap(state.q.rotate, in_axes=(0, None))(state.q, n)
        r_p_cross_n = jnp.cross(state.pos_p[:, None, :], n_prime)
        denom = 1.0 / state.mass[:, None] + jnp.einsum(
            "nk,nwk,nwk->nw", 1.0 / state.inertia, r_p_cross_n, r_p_cross_n
        )
        v_rel = state.vel + jnp.cross(state.angVel, pos_p_lab)
        j = -(1 + e) * v_rel / denom  # Shape (N, dim)

        # --- Tie-Breaking Mask ---
        # Identify particles that are the deepest
        is_deepest_lo = (over_lo > 0) * (over_lo == max_lo)
        is_deepest_hi = (over_hi > 0) * (over_hi == max_hi)
        active_mask = (is_deepest_lo + is_deepest_hi).astype(float)
        count_active = jax.ops.segment_sum(active_mask, state.ID, num_segments=state.N)

        # Avoid division by zero for clumps that aren't touching walls (count=0)
        # We set count to 1.0 where it is 0.0, because those have j_active=0 anyway.
        count_safe = jnp.where(count_active > 0, count_active, 1.0)
        weight = active_mask / count_safe[state.ID]
        j *= weight

        # --- Linear Velocity Update ---
        dv = j / state.mass[:, None]
        dv = jax.ops.segment_sum(dv, state.ID, num_segments=state.N)
        state.vel += dv[state.ID]

        # --- Angular Velocity Update ---
        d_omega = jnp.sum(j[..., None] * r_p_cross_n, axis=1) / state.inertia
        d_omega = state.q.rotate(state.q, d_omega)
        d_omega = jax.ops.segment_sum(d_omega, state.ID, num_segments=state.N)
        state.angVel += d_omega[state.ID]

        return state, system


__all__ = ["ReflectDomain"]
