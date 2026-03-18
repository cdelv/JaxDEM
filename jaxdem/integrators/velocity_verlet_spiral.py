# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Angular-velocity integrator based on the spiral scheme."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from . import RotationIntegrator
from .spiral import omega_dot
from ..utils.quaternion import Quaternion
from ..utils.linalg import unit_and_norm

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@RotationIntegrator.register("verletspiral")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VelocityVerletSpiral(RotationIntegrator):
    """Leapfrog spiral integrator for angular velocities adapted to Velocity Verlet.

    The implementation follows the velocity update described in
    `del Valle et al. (2023) <https://doi.org/10.1016/j.cpc.2023.109077>`_.
    """

    @staticmethod
    @jax.jit(inline=True, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="VelocityVerletSpiral.step_before_force")
    def step_before_force(state: State, system: System) -> tuple[State, System]:
        r"""Advances the simulation state by one half-step before the force calculation using the Velocity Verlet scheme.

        A third-order Runge–Kutta scheme (SSPRK3) integrates the rigid-body angular
        momentum equations in the principal axis frame. The quaternion is updated with
        the spiral leapfrog algorithm to implement a Velocity Verlet-like method.

        - SPIRAL algorithm:

        .. math::
            q(t + \Delta t) = q(t) \cdot e^\left(\frac{\Delta t}{2}\omega(t + \Delta t/2)\right)

        Where the angular velocity and its derivative are purely imaginary quaternions (scalar part is zero and the vector part is equal to the vector). The exponential map of a purely imaginary quaternion is

        .. math::
            e^u = \cos(|u|) + \frac{\vec{u}}{|u|}\sin(|u|)

        Angular velocity is then updated using SSPRK3 which we:

        .. math::
            & \vec{\omega}(t + \Delta t/2) = \vec{\omega}(t) + \frac{1}{6}(k_1 + k_2 + 4k_3) \\
            & k_1 = \Delta t/2\; \dot{\vec{\omega}}(\vec{\omega}(t), \vec{\tau}(t)) \\
            & k_2 = \Delta t/2\; \dot{\vec{\omega}}(\vec{\omega}(t) + k1, \vec{\tau}(t)) \\
            & k_3 = \Delta t/2\; \dot{\vec{\omega}}(\vec{\omega}(t) + (k1 + k2)/4, \vec{\tau}(t)) \\

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
        -----------
        del Valle et. al, SPIRAL: An efficient algorithm for the integration of the equation of rotational motion, https://doi.org/10.1016/j.cpc.2023.109077.

        Note
        -----
        - This method donates ``state`` and ``system``.

        """
        dt_2 = system.dt / 2.0
        inv_inertia = 1.0 / state.inertia
        mask = (1.0 - state.fixed.astype(jnp.float32))[..., None]

        if state.dim == 3:
            ang_vel = state.q.rotate_back(state.q, state.ang_vel)
            torque = state.q.rotate_back(state.q, state.torque)
            w_hat, w_norm = unit_and_norm(ang_vel)
        else:
            ang_vel = state.ang_vel  # (N, 1)
            torque = state.torque  # (N, 1)
            w_norm = jnp.abs(ang_vel)
            w_hat = jnp.sign(ang_vel)

        k1 = dt_2 * omega_dot(ang_vel, torque, state.inertia, inv_inertia)
        k2 = dt_2 * omega_dot(ang_vel + k1, torque, state.inertia, inv_inertia)
        k3 = dt_2 * omega_dot(
            ang_vel + 0.25 * (k1 + k2), torque, state.inertia, inv_inertia
        )

        ang_vel += mask * (k1 + k2 + 4.0 * k3) / 6.0

        theta1 = dt_2 * w_norm
        cos = jnp.cos(theta1)
        sin = jnp.sin(theta1) * w_hat

        if state.dim == 2:
            dq = Quaternion(cos, jnp.array([0.0, 0.0, 1.0]) * sin)
        else:
            dq = Quaternion(cos, sin)

        state.q @= dq
        state.q = jax.lax.cond(
            jnp.mod(system.step_count, 5000) != 0,
            lambda q: q,
            state.q.unit,
            state.q,
        )

        if state.dim == 3:
            state.ang_vel = state.q.rotate(state.q, ang_vel)  # to lab
        else:
            state.ang_vel = ang_vel

        return state, system

    @staticmethod
    @jax.jit(inline=True, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="VelocityVerletSpiral.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        r"""Advances the simulation state by one half-step after the force calculation using the Velocity Verlet scheme.

        A third-order Runge–Kutta scheme (SSPRK3) integrates the rigid-body angular
        momentum equations in the principal axis frame. The quaternion is updated with
        the spiral leapfrog algorithm to implement a Velocity Verlet-like method.

        .. math::
            & \vec{\omega}(t + \Delta t) = \vec{\omega}(t + \Delta t/2) + \frac{1}{6}(k_1 + k_2 + 4k_3) \\
            & k_1 = \Delta t/2\; \dot{\vec{\omega}}(\vec{\omega}(t), \vec{\tau}(t)) \\
            & k_2 = \Delta t/2\; \dot{\vec{\omega}}(\vec{\omega}(t) + k1, \vec{\tau}(t)) \\
            & k_3 = \Delta t/2\; \dot{\vec{\omega}}(\vec{\omega}(t) + (k1 + k2)/4, \vec{\tau}(t)) \\

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
        -----------
        del Valle et. al, SPIRAL: An efficient algorithm for the integration of the equation of rotational motion, https://doi.org/10.1016/j.cpc.2023.109077.

        Note
        -----
        - This method donates ``state`` and ``system``.

        """
        dt_2 = system.dt / 2.0
        inv_inertia = 1.0 / state.inertia
        mask = (1.0 - state.fixed.astype(jnp.float32))[..., None]

        if state.dim == 3:
            ang_vel = state.q.rotate_back(state.q, state.ang_vel)
            torque = state.q.rotate_back(state.q, state.torque)
        else:
            ang_vel = state.ang_vel  # (N, 1)
            torque = state.torque  # (N, 1)

        k1 = dt_2 * omega_dot(ang_vel, torque, state.inertia, inv_inertia)
        k2 = dt_2 * omega_dot(ang_vel + k1, torque, state.inertia, inv_inertia)
        k3 = dt_2 * omega_dot(
            ang_vel + 0.25 * (k1 + k2), torque, state.inertia, inv_inertia
        )

        ang_vel += mask * (k1 + k2 + 4.0 * k3) / 6.0

        if state.dim == 3:
            state.ang_vel = state.q.rotate(state.q, ang_vel)  # to lab
        else:
            state.ang_vel = ang_vel

        return state, system


__all__ = ["VelocityVerletSpiral"]
