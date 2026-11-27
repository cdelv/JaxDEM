# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Self, cast

from ..utils.linalg import cross_3X3D_1X2D
from ..utils.quaternion import Quaternion
from . import Domain, SearchGeometry
from ._toc import verlet_collision_fraction

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("reflect")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ReflectDomain(Domain):
    """A `Domain` implementation that enforces reflective boundary conditions.

    When a particle moves beyond the defined `box_size`, the domain reflects
    its position back into the box. It also reverses the velocity component
    normal to the boundary.
    """

    search_geometry = SearchGeometry.ORTHOGONAL

    restitution_coefficient: jax.Array

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: jax.Array | None = None,
        anchor: jax.Array | None = None,
        restitution_coefficient: float = 1.0,
        **kw: Any,
    ) -> Self:
        """Default factory method for the Domain class.

        This method constructs a new Domain instance with a box-shaped domain
        of the given dimensionality. If you do not provide `box_size` or
        `anchor`, they default to the values below.

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
        ValueError
            If `box_size` or `anchor` have the wrong shape, or if
            `restitution_coefficient` is outside `(0, 1]`.

        """
        if not (0.0 < restitution_coefficient <= 1.0):
            raise ValueError(
                "restitution_coefficient must be in (0, 1], got "
                f"{restitution_coefficient}."
            )
        # Explicit two-arg super: dataclass(slots=True) recreates the class, so
        # zero-arg super()'s __class__ cell points at the discarded original
        # and raises TypeError on Python < 3.14.
        return super(ReflectDomain, cls).Create(
            dim,
            box_size=box_size,
            anchor=anchor,
            restitution_coefficient=jnp.asarray(restitution_coefficient, dtype=float),
            **kw,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ReflectDomain.apply")
    def apply(state: State, system: System) -> tuple[State, System]:
        r"""Apply reflective boundary conditions to particles.

        The method checks particles against the domain boundaries. When a
        particle moves beyond a boundary, the method reflects it. The
        impulse-momentum equations for rigid bodies govern the reflection.

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

        **Verlet Time-of-Collision Correction**

        The shared Verlet-consistent solver
        :func:`jaxdem.domains._toc.verlet_collision_fraction` (also used by
        :class:`ReflectSphereDomain <jaxdem.domains.reflect_sphere.ReflectSphereDomain>`) computes the collision time fraction
        :math:`\alpha \in [0, 1]` per clump at the contact point. Before the
        impulse, the method reconstructs the contact-point velocity and
        angular velocity at the moment of collision as
        :math:`v_{col} = v + (\alpha - 1) \Delta t\, a`. It then integrates
        the post-impulse velocity change over the remaining
        :math:`(1 - \alpha) \Delta t` to correct positions and orientations.

        **Definitions**

        - :math:`\vec{r}_c`: Particle center of mass position (:attr:`jaxdem.State.pos_c <jaxdem.state.State.pos_c>`).
        - :math:`\vec{r}_{p}`: Vector from COM to contact sphere in the lab frame (:attr:`jaxdem.State.pos_p <jaxdem.state.State.pos_p>`).
        - :math:`\vec{v}`: Particle linear velocity (:attr:`jaxdem.State.vel <jaxdem.state.State.vel>`).
        - :math:`\vec{\omega}`: Particle angular velocity (:attr:`jaxdem.State.ang_vel <jaxdem.state.State.ang_vel>`).
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
        if state.only_spheres:
            from .reflect_sphere import ReflectSphereDomain

            return ReflectSphereDomain.apply(state, system)
        domain = cast(ReflectDomain, system.domain)
        e = domain.restitution_coefficient
        pos_p_lab = state._pos_p_rot
        pos = state.pos_c + pos_p_lab

        rad = state.rad[..., None]
        lo = system.domain.anchor + rad
        hi = system.domain.anchor + system.domain.box_size - rad
        over_lo = jnp.maximum(0.0, lo - pos)
        over_hi = jnp.maximum(0.0, pos - hi)
        from ..colliders._partition import _shared_all

        single_body = _shared_all(
            jnp.all(
                state.clump_id == jax.lax.iota(dtype=state.clump_id.dtype, size=state.N)
            )
        )

        def _apply_body(
            *, is_clumped: bool
        ) -> tuple[jax.Array, jax.Array, Quaternion, jax.Array]:
            if is_clumped:
                hit = jnp.any((over_lo > 0.0) | (over_hi > 0.0), axis=-1)
                hit_cid = jnp.where(hit, state.clump_id, state.N)
                body_over = jnp.maximum(
                    0.0,
                    jax.ops.segment_max(
                        jnp.concatenate((over_lo, over_hi), axis=-1),
                        hit_cid,
                        num_segments=state.N,
                    )[state.clump_id],
                )
                max_lo = body_over[..., : state.dim]
                max_hi = body_over[..., state.dim :]
            else:
                max_lo = over_lo
                max_hi = over_hi

            inv_mass = 1.0 / state.mass
            inv_inertia = 1.0 / state.inertia

            if state.dim == 3:
                qw = state.q.w[..., 0]
                qx, qy, qz = (
                    state.q.xyz[..., 0],
                    state.q.xyz[..., 1],
                    state.q.xyz[..., 2],
                )
                px, py, pz = (
                    state.pos_p[..., 0],
                    state.pos_p[..., 1],
                    state.pos_p[..., 2],
                )
                i0, i1, i2 = (
                    inv_inertia[..., 0],
                    inv_inertia[..., 1],
                    inv_inertia[..., 2],
                )

                r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
                r01 = 2.0 * (qx * qy - qw * qz)
                r02 = 2.0 * (qx * qz + qw * qy)
                r10 = 2.0 * (qx * qy + qw * qz)
                r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
                r12 = 2.0 * (qy * qz - qw * qx)
                r20 = 2.0 * (qx * qz - qw * qy)
                r21 = 2.0 * (qy * qz + qw * qx)
                r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

                def _rot_denom(
                    nx: jax.Array, ny: jax.Array, nz: jax.Array
                ) -> jax.Array:
                    cx = py * nz - pz * ny
                    cy = pz * nx - px * nz
                    cz = px * ny - py * nx
                    return i0 * cx * cx + i1 * cy * cy + i2 * cz * cz

                denom_rot = jnp.stack(
                    (
                        _rot_denom(r00, r01, r02),
                        _rot_denom(r10, r11, r12),
                        _rot_denom(r20, r21, r22),
                    ),
                    axis=-1,
                )
            else:
                qw = state.q.w[..., 0]
                qz = state.q.xyz[..., 2]
                c = qw * qw - qz * qz
                s = 2.0 * qw * qz
                px, py = state.pos_p[..., 0], state.pos_p[..., 1]
                c0 = px * (-s) - py * c
                c1 = px * c - py * s
                denom_rot = jnp.stack((c0 * c0, c1 * c1), axis=-1) * inv_inertia

            denom = inv_mass[..., None] + denom_rot
            denom = jnp.where(denom == 0.0, 1.0, denom)

            delta = jnp.maximum(max_lo, max_hi)
            is_deepest_lo = (over_lo > 0) * (over_lo == max_lo)
            is_deepest_hi = (over_hi > 0) * (over_hi == max_hi)
            wall_sign = is_deepest_lo.astype(float) - is_deepest_hi.astype(float)
            active_mask = jnp.abs(wall_sign)

            v_contact_step = state.vel + cross_3X3D_1X2D(state.ang_vel, pos_p_lab)
            acc_clump = state.force * inv_mass[..., None]

            if state.dim == 3:
                tx, ty, tz = (
                    state.torque[..., 0],
                    state.torque[..., 1],
                    state.torque[..., 2],
                )
                wx, wy, wz = (
                    state.ang_vel[..., 0],
                    state.ang_vel[..., 1],
                    state.ang_vel[..., 2],
                )
                tbx = r00 * tx + r10 * ty + r20 * tz
                tby = r01 * tx + r11 * ty + r21 * tz
                tbz = r02 * tx + r12 * ty + r22 * tz
                wbx = r00 * wx + r10 * wy + r20 * wz
                wby = r01 * wx + r11 * wy + r21 * wz
                wbz = r02 * wx + r12 * wy + r22 * wz
                I0, I1, I2 = (
                    state.inertia[..., 0],
                    state.inertia[..., 1],
                    state.inertia[..., 2],
                )
                abx = (tbx - (wby * (I2 * wbz) - wbz * (I1 * wby))) * i0
                aby = (tby - (wbz * (I0 * wbx) - wbx * (I2 * wbz))) * i1
                abz = (tbz - (wbx * (I1 * wby) - wby * (I0 * wbx))) * i2
                alpha_rot = jnp.stack(
                    (
                        r00 * abx + r01 * aby + r02 * abz,
                        r10 * abx + r11 * aby + r12 * abz,
                        r20 * abx + r21 * aby + r22 * abz,
                    ),
                    axis=-1,
                )
            else:
                alpha_rot = state.torque * inv_inertia

            acc_contact = acc_clump + cross_3X3D_1X2D(alpha_rot, pos_p_lab)
            alpha = verlet_collision_fraction(
                v_contact_step, acc_contact, delta, wall_sign, system.dt
            )
            alpha_min_dim = jnp.min(
                jnp.where(active_mask > 0, alpha, 1.0), axis=-1, keepdims=True
            )
            if is_clumped:
                active_cid = jnp.where(
                    jnp.any(active_mask > 0.0, axis=-1), state.clump_id, state.N
                )
                alpha_clump = jnp.minimum(
                    1.0,
                    jax.ops.segment_min(
                        alpha_min_dim, active_cid, num_segments=state.N
                    ),
                )
                alpha_clump_flat = alpha_clump[state.clump_id]
            else:
                alpha_clump_flat = alpha_min_dim

            dt_factor = (alpha_clump_flat - 1.0) * system.dt
            v_contact = v_contact_step + dt_factor * acc_contact
            v_rel_dot_n = v_contact
            j_magnitude = -(1.0 + e) * v_rel_dot_n / denom  # (N, D)
            closing_mask = (v_rel_dot_n * wall_sign) < 0.0

            if is_clumped:
                count_active = jax.ops.segment_sum(
                    active_mask, active_cid, num_segments=state.N
                )
                count_safe = jnp.maximum(count_active, 1.0)
                weight = active_mask / count_safe[state.clump_id]
            else:
                weight = active_mask
            j_magnitude *= weight * closing_mask

            dv_particle = j_magnitude * inv_mass[..., None]

            if state.dim == 2:
                jx, jy = j_magnitude[..., 0], j_magnitude[..., 1]
                jbx = c * jx + s * jy
                jby = -s * jx + c * jy
                d_omega_lab = (px * jby - py * jbx)[..., None] * inv_inertia
            else:
                jx, jy, jz = (
                    j_magnitude[..., 0],
                    j_magnitude[..., 1],
                    j_magnitude[..., 2],
                )
                jbx = r00 * jx + r10 * jy + r20 * jz
                jby = r01 * jx + r11 * jy + r21 * jz
                jbz = r02 * jx + r12 * jy + r22 * jz
                dbx = (py * jbz - pz * jby) * i0
                dby = (pz * jbx - px * jbz) * i1
                dbz = (px * jby - py * jbx) * i2
                d_omega_lab = jnp.stack(
                    (
                        r00 * dbx + r01 * dby + r02 * dbz,
                        r10 * dbx + r11 * dby + r12 * dbz,
                        r20 * dbx + r21 * dby + r22 * dbz,
                    ),
                    axis=-1,
                )

            if is_clumped:
                velocity_update = jax.ops.segment_sum(
                    jnp.concatenate((dv_particle, d_omega_lab), axis=-1),
                    active_cid,
                    num_segments=state.N,
                )[state.clump_id]
                velocity_update = jnp.where(
                    state.fixed[..., None], 0.0, velocity_update
                )
                dv_flat = velocity_update[..., : state.dim]
                d_omega_net_flat = velocity_update[..., state.dim :]
            else:
                free = ~state.fixed[..., None]
                dv_flat = jnp.where(free, dv_particle, 0.0)
                d_omega_net_flat = jnp.where(free, d_omega_lab, 0.0)

            vel_out = state.vel + dv_flat
            ang_vel_out = state.ang_vel + d_omega_net_flat
            dt_remaining = (1.0 - alpha_clump_flat) * system.dt
            delta_theta = d_omega_net_flat * dt_remaining
            if state.dim == 2:
                delta_theta = jnp.concatenate(
                    [
                        jnp.zeros_like(delta_theta),
                        jnp.zeros_like(delta_theta),
                        delta_theta,
                    ],
                    axis=-1,
                )
            dq = Quaternion.from_small_rotvec(delta_theta)
            q_out = Quaternion.unit(dq @ state.q)
            pos_c_out = state.pos_c + dv_flat * dt_remaining
            return vel_out, ang_vel_out, q_out, pos_c_out

        state.vel, state.ang_vel, state.q, state.pos_c = jax.lax.cond(
            single_body,
            lambda _: _apply_body(is_clumped=False),
            lambda _: _apply_body(is_clumped=True),
            None,
        )
        return state, system
