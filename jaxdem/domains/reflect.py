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
        e = 1.0

        # pos_p is in Body Frame. Transform Body -> Global using rotate_back.
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

        # --- Boundary normal vectors ---
        n = jnp.eye(state.dim, state.dim)
        n_body = jax.vmap(state.q.rotate, in_axes=(0, None))(state.q, n)  # Body frame

        # --- DIMENSION BRANCHING ---
        if state.dim == 3:
            r_p_cross_n = jnp.cross(state.pos_p[:, None, :], n_body)  # Body frame
            v_rot = jnp.cross(state.angVel, pos_p_lab)
        else:
            r_p_cross_n = jnp.cross(state.pos_p[:, None, :], n_body)[
                ..., None
            ]  # Body frame
            perp_r = jnp.stack([-pos_p_lab[:, 1], pos_p_lab[:, 0]], axis=-1)
            v_rot = state.angVel * perp_r

        # --- Impulse ---
        denom = 1.0 / state.mass[:, None] + jnp.einsum(
            "nk,nwk,nwk->nw", 1.0 / state.inertia, r_p_cross_n, r_p_cross_n
        )
        j = -(1 + e) * jnp.dot(state.vel + v_rot, n) / denom

        # --- Masking & Weights ---
        is_deepest_lo = (over_lo > 0) * (over_lo == max_lo)
        is_deepest_hi = (over_hi > 0) * (over_hi == max_hi)
        active_mask = ((is_deepest_lo + is_deepest_hi) > 0).astype(float)

        count_active = jax.ops.segment_sum(active_mask, state.ID, num_segments=state.N)
        count_safe = jnp.where(count_active > 0, count_active, 1.0)
        weight = active_mask / count_safe[state.ID]
        j *= weight

        # --- Linear Update ---
        dv = j / state.mass[:, None]
        dv = jax.ops.segment_sum(dv, state.ID, num_segments=state.N)
        state.vel += dv[state.ID]

        # --- Angular Update ---
        # 1. Calculate torque impulse in BODY frame (using body frame lever arm r_p_cross_n)
        impulse_vecs = j[:, :, None] * n_body
        d_omega_body = jnp.sum(jnp.cross(state.pos_p[:, None, :], impulse_vecs), axis=1)
        d_omega_global = state.q.rotate_back(state.q, d_omega_body / state.inertia)

        d_omega_sum = jax.ops.segment_sum(
            d_omega_global, state.ID, num_segments=state.N
        )
        state.angVel += d_omega_sum[state.ID]

        return state, system


__all__ = ["ReflectDomain"]
