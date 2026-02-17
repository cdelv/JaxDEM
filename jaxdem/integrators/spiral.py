# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Angular-velocity integrator based on the spiral scheme."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import RotationIntegrator
from ..utils.quaternion import Quaternion

from ..utils.linalg import cross

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="spiral.omega_dot")
def omega_dot(
    w: jax.Array, torque: jax.Array, inertia: jax.Array, inv_inertia: jax.Array
) -> jax.Array:
    r"""Compute the time derivative of the angular velocity for diagonal inertia.

    Parameters
    ----------
    w : jax.Array
        Angular velocity with shape ``(..., N, D)`` where ``D`` is ``1`` for planar
        simulations and ``3`` for spatial simulations.
    ang_accel : jax.Array
        Angular acceleration obtained from external torques divided by the inertia
        (same shape as ``w``).
    inertia : jax.Array
        Diagonal inertia tensor with the same trailing dimension as ``w``.

    Returns
    -------
    jax.Array
        :math:`\dot{\boldsymbol{\omega}}`, the angular acceleration consistent with the
        rigid-body equations of motion.
    """
    D = w.shape[-1]
    if D == 3:
        return (torque - cross(w, inertia * w)) * inv_inertia
    else:
        return torque * inv_inertia

    raise ValueError(f"omega_dot supports D in {{1,3}}, got D={D}")


@RotationIntegrator.register("spiral")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Spiral(RotationIntegrator):
    """
    Non-leapfrog spiral integrator for angular velocities.

    The implementation follows the velocity update described in
    `del Valle et al. (2023) <https://doi.org/10.1016/j.cpc.2023.109077>`_.
    """

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="spiral.step_after_force")
    def step_after_force(state: State, system: System) -> Tuple[State, System]:
        r"""
        Advance angular velocities by a single time step.

        A third-order Runge–Kutta scheme (SSPRK3) integrates the rigid-body angular
        momentum equations in the principal axis frame. The quaternion is updated based on the spiral
        non-leapfrog algorithm.

        - SPIRAL algorithm:

        .. math::
            q(t + \Delta t) = q(t) \cdot e^\left(\frac{\Delta t}{2}\omega\right)  \cdot e^\left(\frac{\Delta t^2}{4}\dot{\omega}\right)

        Where the angular velocity and its derivative are purely imaginary quaternions (scalar part is zero and the vector part is equal to the vector). The exponential map of a purely imaginary quaternion is

        .. math::
            e^u = \cos(|u|) + \frac{\vec{u}}{|u|}\sin(|u|)

        Angular velocity is then updated using SSPRK3:

        .. math::
            & \vec{\omega}(t + \Delta t) = \vec{\omega}(t) + \frac{1}{6}(k_1 + k_2 + 4k_3) \\
            & k_1 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t + \Delta t / 2), \vec{\tau}(t + \Delta t)) \\
            & k_2 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t + \Delta t / 2) + k1, \vec{\tau}(t + \Delta t)) \\
            & k_3 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t + \Delta t / 2) + (k1 + k2)/4, \vec{\tau}(t + \Delta t)) \\

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
        --------
        Tuple[State, System]
            The updated state and system after one time step.


        Reference
        ----------
        del Valle et. al, SPIRAL: An efficient algorithm for the integration of the equation of rotational motion, https://doi.org/10.1016/j.cpc.2023.109077.


        Note
        -----
        - This method donates state and system
        """
        dt_2 = system.dt / 2
        inv_inertia = 1.0 / state.inertia

        if state.dim == 3:
            angVel = state.q.rotate_back(state.q, state.angVel)
            torque = state.q.rotate_back(state.q, state.torque)
            w_norm2 = jnp.sum(angVel * angVel, axis=-1)[..., None]
            w_norm = jnp.sqrt(w_norm2)
            w_dot = omega_dot(angVel, torque, state.inertia, inv_inertia)
            w_dot_norm2 = jnp.sum(w_dot * w_dot, axis=-1)[..., None]
            w_dot_norm = jnp.sqrt(w_dot_norm2)

        else:
            angVel = state.angVel  # (N, 1)
            torque = state.torque  # (N, 1)
            w_norm = jnp.abs(angVel)
            w_dot = omega_dot(angVel, torque, state.inertia, inv_inertia)
            w_dot_norm = jnp.abs(w_dot)

        theta1 = dt_2 * w_norm
        theta2 = jnp.power(dt_2, 2) * w_dot_norm
        w_norm = jnp.where(w_norm == 0, 1.0, w_norm)
        w_dot_norm = jnp.where(w_dot_norm == 0, 1.0, w_dot_norm)
        cos1 = jnp.cos(theta1)
        sin1 = jnp.sin(theta1) * angVel / w_norm
        cos2 = jnp.cos(theta2)
        sin2 = jnp.sin(theta2) * w_dot / w_dot_norm

        if state.dim == 2:
            dq = Quaternion(cos1, jnp.array([0, 0, 1]) * sin1) @ Quaternion(
                cos2, jnp.array([0, 0, 1]) * sin2
            )
        else:
            dq = Quaternion(cos1, sin1) @ Quaternion(cos2, sin2)

        state.q @= dq
        state.q = jax.lax.cond(
            jnp.mod(system.step_count, 5000) != 0,
            lambda q: q,
            state.q.unit,
            state.q,
        )

        k1 = system.dt * w_dot
        k2 = system.dt * omega_dot(angVel + k1, torque, state.inertia, inv_inertia)
        k3 = system.dt * omega_dot(
            angVel + 0.25 * (k1 + k2), torque, state.inertia, inv_inertia
        )
        angVel += (1 - state.fixed)[..., None] * (k1 + k2 + 4.0 * k3) / 6.0

        if state.dim == 3:
            state.angVel = state.q.rotate(state.q, angVel)  # to lab
        else:
            state.angVel = angVel

        return state, system


__all__ = ["Spiral"]
