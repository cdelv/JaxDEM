# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Angular-velocity integrator based on the spiral scheme."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from . import RotationIntegrator, free_mask
from ..utils.quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
@partial(jax.named_call, name="spiral.omega_dot")
def omega_dot(
    w: jax.Array, torque: jax.Array, inertia: jax.Array, inv_inertia: jax.Array
) -> jax.Array:
    """Compute the time derivative of the angular velocity for diagonal inertia."""
    D = w.shape[-1]
    if D == 3:
        wx, wy, wz = w[..., 0:1], w[..., 1:2], w[..., 2:3]
        ix, iy, iz = inertia[..., 0:1], inertia[..., 1:2], inertia[..., 2:3]
        tx, ty, tz = torque[..., 0:1], torque[..., 1:2], torque[..., 2:3]
        cx = wy * (iz * wz) - wz * (iy * wy)
        cy = wz * (ix * wx) - wx * (iz * wz)
        cz = wx * (iy * wy) - wy * (ix * wx)
        return jnp.concatenate([tx - cx, ty - cy, tz - cz], axis=-1) * inv_inertia
    if D == 1:
        return torque * inv_inertia

    raise ValueError(f"omega_dot supports D in {{1,3}}, got shape {w.shape}")


@RotationIntegrator.register("spiral")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Spiral(RotationIntegrator):
    """Non-leapfrog spiral integrator for angular velocities.

    The implementation follows the velocity update described in
    `del Valle et al. (2023) <https://doi.org/10.1016/j.cpc.2023.109077>`_.
    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="spiral.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        r"""Advance angular velocities by a single time step.

        A third-order Runge–Kutta scheme (SSPRK3) integrates the rigid-body angular
        momentum equations in the principal axis frame. The quaternion is updated based on the spiral
        non-leapfrog algorithm.

        - SPIRAL algorithm:

        .. math::
            q(t + \Delta t) = q(t) \cdot e^{\left(\frac{\Delta t}{2}\omega\right)}  \cdot e^{\left(\frac{\Delta t^2}{4}\dot{\omega}\right)}

        Where the angular velocity and its derivative are purely imaginary quaternions (scalar part is zero and the vector part is equal to the vector). The exponential map of a purely imaginary quaternion is

        .. math::
            e^u = \cos(|u|) + \frac{\vec{u}}{|u|}\sin(|u|)

        Angular velocity is then updated using SSPRK3:

        .. math::
            & \vec{\omega}(t + \Delta t) = \vec{\omega}(t) + \frac{1}{6}(k_1 + k_2 + 4k_3) \\
            & k_1 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t), \vec{\tau}(t + \Delta t)) \\
            & k_2 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t) + k_1, \vec{\tau}(t + \Delta t)) \\
            & k_3 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t) + (k_1 + k_2)/4, \vec{\tau}(t + \Delta t))

        Where the angular velocity derivative is a function of the torque and angular velocity:

        .. math::
            \dot{\vec{\omega}} = (\tau - \vec{\omega} \times (I \vec{\omega}))I^{-1}

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The updated state and system after one time step.


        Reference
        ----------
        del Valle et. al, SPIRAL: An efficient algorithm for the integration of the equation of rotational motion, https://doi.org/10.1016/j.cpc.2023.109077.

        """
        dt_2 = system.dt / 2
        inv_inertia = 1.0 / state.inertia

        if state.dim == 3:
            ang_vel = state.q.rotate_back(state.q, state.ang_vel)
            torque = state.q.rotate_back(state.q, state.torque)
        else:
            ang_vel = state.ang_vel  # (N, 1)
            torque = state.torque  # (N, 1)
        w_dot = omega_dot(ang_vel, torque, state.inertia, inv_inertia)

        # Rotation vectors whose half-angles are dt/2*|w| and (dt/2)^2*|w_dot|.
        rotvec1 = 2.0 * dt_2 * ang_vel
        rotvec2 = 2.0 * dt_2 * dt_2 * w_dot
        if state.dim == 2:
            rotvec1 = jnp.array([0.0, 0.0, 1.0]) * rotvec1
            rotvec2 = jnp.array([0.0, 0.0, 1.0]) * rotvec2

        dq = Quaternion.from_rotvec(rotvec1) @ Quaternion.from_rotvec(rotvec2)

        q_next = state.q @ dq
        state.q = q_next.unit(q_next)

        k1 = system.dt * w_dot
        k2 = system.dt * omega_dot(ang_vel + k1, torque, state.inertia, inv_inertia)
        k3 = system.dt * omega_dot(
            ang_vel + 0.25 * (k1 + k2), torque, state.inertia, inv_inertia
        )
        ang_vel += free_mask(state) * (k1 + k2 + 4.0 * k3) / 6.0

        if state.dim == 3:
            state.ang_vel = state.q.rotate(state.q, ang_vel)  # to lab
        else:
            state.ang_vel = ang_vel

        return state, system


__all__ = ["Spiral"]
