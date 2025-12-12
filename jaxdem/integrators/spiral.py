# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Angular-velocity integrator based on the spiral scheme."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import RotationIntegrator
from ..utils.quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="spiral.omega_dot")
def omega_dot(w: jax.Array, torque: jax.Array, inertia: jax.Array) -> jax.Array:
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
    if D == 1:
        return torque / inertia

    if D == 3:
        return (torque - jnp.linalg.cross(w, inertia * w)) / inertia

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
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
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
            & k_1 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t), \vec{\tau}(t)) \\
            & k_2 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t) + k1, \vec{\tau}(t)) \\
            & k_3 = \Delta t\; \dot{\vec{\omega}}(\vec{\omega}(t) + (k1 + k2)/4, \vec{\tau}(t)) \\
        
        Where the angular velocity derivative is a function of the torque and angular velocity:

        .. math::
            \dot{\vec{\omega}} = (\tau + \vec{\omega} \times (I \vec{\omega}))I^{-1}

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
        - TO DO: make it work without padding the vectors
        """
        if state.dim == 2:
            angVel_lab_3d = jnp.pad(state.angVel, ((0, 0), (2, 0)), constant_values=0.0)
            torque_lab_3d = jnp.pad(state.torque, ((0, 0), (2, 0)), constant_values=0.0)
        else:  # state.dim == 3
            angVel_lab_3d = state.angVel
            torque_lab_3d = state.torque

        angVel = state.q.rotate_back(state.q, angVel_lab_3d)  # to body
        torque = state.q.rotate_back(state.q, torque_lab_3d)

        w_dot = omega_dot(angVel, torque, state.inertia)

        w_norm2 = jnp.sum(angVel * angVel, axis=-1, keepdims=True)
        w_dot_norm2 = jnp.sum(w_dot * w_dot, axis=-1, keepdims=True)
        w_norm = jnp.sqrt(w_norm2)
        w_dot_norm = jnp.sqrt(w_dot_norm2)

        theta1 = system.dt * w_norm / 2
        theta2 = jnp.power(system.dt, 2) * w_dot_norm / 4

        w_norm = jnp.where(w_norm == 0, 1.0, w_norm)
        w_dot_norm = jnp.where(w_dot_norm == 0, 1.0, w_dot_norm)

        state.q @= Quaternion(
            jnp.cos(theta1),
            jnp.sin(theta1) * angVel / w_norm,
        ) @ Quaternion(
            jnp.cos(theta2),
            jnp.sin(theta2) * w_dot / w_dot_norm,
        )
        state.q = state.q.unit(state.q)

        k1 = system.dt * w_dot
        k2 = system.dt * omega_dot(angVel + k1, torque, state.inertia)
        k3 = system.dt * omega_dot(angVel + 0.25 * (k1 + k2), torque, state.inertia)
        angVel += (1 - state.fixed)[..., None] * (k1 + k2 + 4.0 * k3) / 6.0

        if state.dim == 2:
            state.angVel = angVel[..., -1:]
        else:
            state.angVel = state.q.rotate(state.q, angVel)  # to lab

        return state, system


__all__ = ["Spiral"]
