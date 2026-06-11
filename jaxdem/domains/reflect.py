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
        pos_p_lab = state._pos_p_rot
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

        if state.dim == 3:
            R = n_prime
            R_T = jnp.swapaxes(n_prime, -1, -2)
            torque_body = jnp.einsum('...ij,...j->...i', R_T, state.torque)
            ang_vel_body = jnp.einsum('...ij,...j->...i', R_T, state.ang_vel)
            alpha_body = (
                torque_body - cross(ang_vel_body, state.inertia * ang_vel_body)
            ) * inv_inertia
            alpha_rot = jnp.einsum('...ij,...j->...i', R, alpha_body)
        else:
            alpha_rot = state.torque * inv_inertia

        acc_contact = acc_clump + cross_3X3D_1X2D(alpha_rot, pos_p_lab)

        v_0 = v_contact_step - system.dt * acc_contact
        v_mid = v_contact_step - 0.5 * system.dt * acc_contact

        v_0_n = v_0 * wall_sign
        v_mid_n = v_mid * wall_sign
        acc_n = acc_contact * wall_sign

        A_pos = 0.5 * acc_n * system.dt * system.dt
        B_pos = v_0_n * system.dt
        d_wall_pos = delta + v_mid_n * system.dt

        disc = B_pos * B_pos + 4.0 * A_pos * d_wall_pos
        disc = jnp.maximum(0.0, disc)

        alpha_stable = jnp.where(
            B_pos < 0,
            2.0
            * d_wall_pos
            / jnp.where(B_pos - jnp.sqrt(disc) < -1e-10, B_pos - jnp.sqrt(disc), -1.0),
            (-B_pos - jnp.sqrt(disc))
            / jnp.where(jnp.abs(2.0 * A_pos) > 1e-10, 2.0 * A_pos, 1.0),
        )
        alpha = jnp.clip(alpha_stable, 0.0, 1.0)

        # Clump-level collision time fraction (minimum among all active coordinates of all spheres in the clump)
        alpha_clump = jax.ops.segment_min(
            jnp.where(active_mask > 0, alpha, 1.0), state.clump_id, num_segments=state.N
        )
        alpha_clump = jnp.min(alpha_clump, axis=-1, keepdims=True)
        alpha_clump_flat = alpha_clump[state.clump_id]

        dt_factor = (alpha_clump_flat - 1.0) * system.dt
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
        if state.dim == 3:
            j_body = jnp.einsum('...ij,...j->...i', R_T, j_magnitude)
        else:
            j_body = jnp.einsum('...ji,...j->...i', n_prime, j_magnitude)
            
        moment_net_body = cross(state.pos_p, j_body)

        if state.dim == 2:
            d_omega_lab = moment_net_body[..., -1:] * inv_inertia
        else:
            d_omega_body = moment_net_body * inv_inertia
            d_omega_lab = jnp.einsum('...ij,...j->...i', R, d_omega_body)

        d_omega_net = jax.ops.segment_sum(
            d_omega_lab, state.clump_id, num_segments=state.N
        )
        d_omega_net_flat = d_omega_net[state.clump_id]
        d_omega_net_flat = jnp.where(state.fixed[:, None], 0.0, d_omega_net_flat)
        state.ang_vel += d_omega_net_flat

        # --- Orientation Correction ---
        dt_remaining = (1.0 - alpha_clump_flat) * system.dt
        delta_theta = d_omega_net_flat * dt_remaining

        theta_hat, theta_norm = unit_and_norm(delta_theta)

        half_theta = 0.5 * theta_norm
        cos_half = jnp.cos(half_theta)
        sin_half = jnp.sin(half_theta) * theta_hat

        if state.dim == 2:
            dq = Quaternion(
                cos_half,
                jnp.concatenate(
                    [jnp.zeros_like(sin_half), jnp.zeros_like(sin_half), sin_half],
                    axis=-1,
                ),
            )
        else:
            dq = Quaternion(cos_half, sin_half)

        q_corrected = dq @ state.q
        state.q = Quaternion.unit(q_corrected)

        # --- Position Correction ---
        displacement_clump = (1.0 - alpha_clump) * system.dt * dv
        displacement = displacement_clump[state.clump_id]
        displacement = jnp.where(state.fixed[:, None], 0.0, displacement)
        state.pos_c += displacement

        return state, system
