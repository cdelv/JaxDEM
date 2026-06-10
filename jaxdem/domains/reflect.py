# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from ..utils.linalg import cross, cross_3X3D_1X2D, unit_and_norm
from ..utils.quaternion import Quaternion
from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("reflect")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ReflectDomain(Domain):
    """A `Domain` implementation that enforces reflective boundary conditions.

    Particles that attempt to move beyond the defined `box_size` will have their
    positions reflected back into the box and their velocities reversed in the
    direction normal to the boundary.
    """

    restitution_coefficient: jax.Array

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: jax.Array | None = None,
        anchor: jax.Array | None = None,
        restitution_coefficient: float = 1.0,
    ) -> Self:
        """Default factory method for the Domain class.

        This method constructs a new Domain instance with a box-shaped domain
        of the given dimensionality. If `box_size` or `anchor` are not provided,
        they are initialized to default values.

        Parameters
        ----------
        dim : int
            The dimensionality of the domain (e.g., 2, 3).
        box_size : jax.Array, optional
            The size of the domain along each dimension. If not provided,
            defaults to an array of ones with shape `(dim,)`.
        anchor : jax.Array, optional
            The anchor (origin) of the domain. If not provided,
            defaults to an array of zeros with shape `(dim,)`.
        restitution_coefficient : float
            Restitution coefficient between 0 and 1 to modulate energy conservation with wall.

        Returns
        -------
        ReflectDomain
            A new instance of the Domain subclass with the specified
            or default configuration.

        Raises
        ------
        AssertionError
            If `box_size` and `anchor` do not have the same shape.

        """
        if box_size is None:
            box_size = jnp.ones(dim, dtype=float)
        box_size = jnp.asarray(box_size, dtype=float)

        if box_size.shape != (dim,):
            raise ValueError(
                f"box_size must have shape ({dim},), got shape {box_size.shape}."
            )

        if anchor is None:
            anchor = jnp.zeros_like(box_size, dtype=float)
        anchor = jnp.asarray(anchor, dtype=float)

        if anchor.shape != (dim,):
            raise ValueError(
                f"anchor must have shape ({dim},), got shape {anchor.shape}."
            )

        assert (restitution_coefficient <= 1) * (restitution_coefficient > 0)
        return cls(
            box_size=box_size,
            anchor=anchor,
            restitution_coefficient=jnp.asarray(restitution_coefficient, dtype=float),
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ReflectDomain.apply")
    def apply(state: State, system: System) -> tuple[State, System]:
        r"""Applies reflective boundary conditions to particles.

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

        **Verlet Time-of-Collision Correction**

        Under Verlet integration, the collision time fraction :math:`\alpha \in [0, 1]` is solved exactly using the quadratic equation:

        .. math::
            A \alpha^2 + B \alpha + C = 0

        where:
            - :math:`A = - \frac{1}{2} a_{n} \Delta t^2`
            - :math:`B = - v_{0, n} \Delta t`
            - :math:`C = -\delta - v_{mid, n} \Delta t`
            - :math:`v_{0, n}`: Normal velocity of the contact point at the start of the step.
            - :math:`a_{n}`: Normal acceleration of the contact point.
            - :math:`v_{mid, n}`: Normal velocity of the contact point at the end of the step.
            - :math:`\delta`: Penetration depth (overlap).

        The solution for :math:`\alpha` is given by:

        .. math::
            \alpha = \frac{2 C}{B + \sqrt{B^2 - 4 A C}}

        The velocity at the moment of collision :math:`v_{col}` is then calculated and used to update the pre-collision velocity.

        **Definitions**

        - :math:`\vec{r}_c`: Particle center of mass position (:attr:`jaxdem.State.pos_c`).
        - :math:`\vec{r}_{p}`: Vector from COM to contact sphere in the lab frame (:attr:`jaxdem.State.pos_p`).
        - :math:`\vec{v}`: Particle linear velocity (:attr:`jaxdem.State.vel`).
        - :math:`\vec{\omega}`: Particle angular velocity (:attr:`jaxdem.State.ang_vel`).
        - :math:`\hat{n}`: Boundary normal vector (pointing into the domain).
        - :math:`\delta`: Penetration depth (positive value).
        - :math:`e`: Coefficient of restitution.

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

        Reference
        ----------
        https://www.myphysicslab.com/engine2D/collision-en.html

        """
        domain = cast(ReflectDomain, system.domain)
        e = domain.restitution_coefficient
        pos_p_lab = state.q.rotate(state.q, state.pos_p)
        pos = state.pos_c + pos_p_lab

        rad = state.rad[:, None]
        lo = system.domain.anchor + rad
        hi = system.domain.anchor + system.domain.box_size - rad
        over_lo = jnp.maximum(0.0, lo - pos)
        over_hi = jnp.maximum(0.0, pos - hi)

        body_over_lo = jax.ops.segment_max(
            over_lo, state.clump_id, num_segments=state.N
        )
        body_over_hi = jax.ops.segment_max(
            over_hi, state.clump_id, num_segments=state.N
        )
        max_lo = body_over_lo[state.clump_id]
        max_hi = body_over_hi[state.clump_id]

        inv_mass = (1.0 / state.mass)[:, None]
        inv_inertia = 1.0 / state.inertia

        n = jnp.eye(state.dim, state.dim)
        n_prime = jax.vmap(state.q.rotate_back, in_axes=(0, None))(state.q, n)
        r_p_cross_n = cross(state.pos_p[:, None, :], n_prime)

        denom = inv_mass + jnp.sum(
            inv_inertia[:, None, :] * (r_p_cross_n * r_p_cross_n), axis=-1
        )
        denom = jnp.where(denom > 1e-10, denom, 1.0)

        delta = jnp.maximum(max_lo, max_hi)
        is_deepest_lo = (over_lo > 0) & (over_lo == max_lo)
        is_deepest_hi = (over_hi > 0) & (over_hi == max_hi)
        wall_sign = is_deepest_lo.astype(float) - is_deepest_hi.astype(float)
        active_mask = jnp.abs(wall_sign)

        v_contact_step = state.vel + cross_3X3D_1X2D(state.ang_vel, pos_p_lab)

        acc_clump = state.force * inv_mass
        alpha_rot = state.torque * inv_inertia
        acc_contact = acc_clump + cross_3X3D_1X2D(alpha_rot, pos_p_lab)

        v_start = v_contact_step - 0.5 * system.dt * acc_contact

        v_start_n = v_start * wall_sign
        acc_n = acc_contact * wall_sign

        d_wall_pos = -delta - system.dt * v_contact_step * wall_sign
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

        # Clump-level collision time fraction (minimum among all active coordinates of all spheres in the clump)
        alpha_clump = jax.ops.segment_min(
            jnp.where(active_mask > 0, alpha, 1.0), state.clump_id, num_segments=state.N
        )
        alpha_clump = jnp.min(alpha_clump, axis=-1, keepdims=True)
        alpha_clump_flat = alpha_clump[state.clump_id]

        dt_factor = (alpha_clump_flat - 0.5) * system.dt
        v_col = state.vel + dt_factor * acc_clump
        ang_vel_col = state.ang_vel + dt_factor * alpha_rot

        v_contact = v_col + cross_3X3D_1X2D(ang_vel_col, pos_p_lab)
        v_rel_dot_n = v_contact
        j_magnitude = -(1.0 + e) * v_rel_dot_n / denom  # (N, D)

        closing_mask = (v_rel_dot_n * wall_sign) < 0.0

        count_active = jax.ops.segment_sum(
            active_mask, state.clump_id, num_segments=state.N
        )
        count_safe = jnp.maximum(count_active, 1.0)
        weight = active_mask / count_safe[state.clump_id]
        j_magnitude *= weight * closing_mask

        # --- Linear Velocity Update ---
        dv = jax.ops.segment_sum(
            j_magnitude * inv_mass, state.clump_id, num_segments=state.N
        )
        dv_flat = dv[state.clump_id]
        dv_flat = jnp.where(state.fixed[:, None], 0.0, dv_flat)
        state.vel += dv_flat

        # --- Angular Velocity Update (Optimized) ---
        j_body = state.q.rotate_back(state.q, j_magnitude)
        moment_net_body = cross(state.pos_p, j_body)

        if state.dim == 2:
            d_omega_lab = moment_net_body[..., -1:] * inv_inertia
        else:
            d_omega_body = moment_net_body * inv_inertia
            d_omega_lab = state.q.rotate(state.q, d_omega_body)

        d_omega_net = jax.ops.segment_sum(
            d_omega_lab, state.clump_id, num_segments=state.N
        )
        d_omega_net_flat = d_omega_net[state.clump_id]
        d_omega_net_flat = jnp.where(state.fixed[:, None], 0.0, d_omega_net_flat)
        state.ang_vel += d_omega_net_flat

        # --- Position Correction ---
        displacement_clump = (1.0 - alpha_clump) * system.dt * dv
        displacement = displacement_clump[state.clump_id]
        displacement = jnp.where(state.fixed[:, None], 0.0, displacement)
        state.pos_c += displacement

        # --- Quaternion Correction for time-of-collision ---
        w_pre_lab = state.ang_vel - d_omega_net_flat
        w_post_lab = state.ang_vel

        w_pre_body_est = state.q.rotate_back(state.q, w_pre_lab)

        def get_dq(dt_val, w_body):
            if state.dim == 3:
                w_hat, w_norm = unit_and_norm(w_body)
            else:
                w_norm = jnp.abs(w_body)
                w_hat = jnp.sign(w_body)
            theta = dt_val[:, None] * 0.5 * w_norm
            cos = jnp.cos(theta)
            sin = jnp.sin(theta) * w_hat
            if state.dim == 2:
                return Quaternion(cos, jnp.concatenate([jnp.zeros_like(sin), jnp.zeros_like(sin), sin], axis=-1))
            else:
                return Quaternion(cos, sin)

        dq_pre_full = get_dq(jnp.ones_like(alpha_clump_flat[..., 0]) * system.dt, w_pre_body_est)
        q_old = state.q @ dq_pre_full.inv(dq_pre_full)

        w_pre_body = q_old.rotate_back(q_old, w_pre_lab)
        alpha_p = alpha_clump_flat[..., 0]

        dq_pre = get_dq(alpha_p * system.dt, w_pre_body)
        q_col = q_old @ dq_pre

        # Post-collision rotation is integrated starting from the orientation at collision (q_col).
        w_post_body_col = q_col.rotate_back(q_col, w_post_lab)
        dq_post_rem = get_dq((1.0 - alpha_p) * system.dt, w_post_body_col)

        q_corrected = q_col @ dq_post_rem

        # A clump is active if it had any active collision mask
        clump_active = jax.ops.segment_sum(active_mask, state.clump_id, num_segments=state.N) > 0.0
        clump_active = clump_active.any(axis=-1, keepdims=True)
        clump_active_flat = clump_active[state.clump_id]

        state.q = Quaternion(
            w=jnp.where(clump_active_flat, q_corrected.w, state.q.w),
            xyz=jnp.where(clump_active_flat, q_corrected.xyz, state.q.xyz)
        )

        return state, system
