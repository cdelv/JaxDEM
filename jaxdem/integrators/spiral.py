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
def omega_dot(w: jax.Array, ang_accel: jax.Array, inertia: jax.Array) -> jax.Array:
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
        return ang_accel

    if D == 3:
        return ang_accel - jnp.linalg.cross(w, inertia * w) / inertia

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
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="spiral.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Advance angular velocities by a single time step.

        A third-order Runge–Kutta scheme (SSPRK3) integrates the rigid-body angular
        momentum equations in the principal axis frame. The quaternion is updated based on the spiral
        non-leapfrog algorithm.

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

        Note
        -----
        - This method donates state and system
        """
        state.angVel = state.q.rotate(state.q, state.angVel)
        torque = state.q.rotate(state.q, state.torque)

        ang_accel = torque / state.inertia
        if state.dim == 2:
            state.angVel = jnp.pad(state.angVel, ((0, 0), (2, 0)), constant_values=0.0)
            ang_accel = jnp.pad(ang_accel, ((0, 0), (2, 0)), constant_values=0.0)

        w_dot = omega_dot(state.angVel, ang_accel, state.inertia)
        w_norm = jnp.linalg.norm(state.angVel, axis=-1, keepdims=True)
        w_dot_norm = jnp.linalg.norm(w_dot, axis=-1, keepdims=True)

        theta1 = 0.5 * system.dt * w_norm
        theta2 = 0.25 * jnp.power(system.dt, 2) * w_dot_norm

        w_norm = jnp.where(w_norm == 0, 1.0, w_norm)
        w_dot_norm = jnp.where(w_dot_norm == 0, 1.0, w_dot_norm)

        state.q @= Quaternion(
            jnp.cos(theta1),
            jnp.sin(theta1) * state.angVel / w_norm,
        ) @ Quaternion(
            jnp.cos(theta2),
            jnp.sin(theta2) * w_dot / w_dot_norm,
        )

        k1 = system.dt * w_dot
        k2 = system.dt * omega_dot(state.angVel + k1, ang_accel, state.inertia)
        k3 = system.dt * omega_dot(
            state.angVel + 0.25 * (k1 + k2), ang_accel, state.inertia
        )
        state.angVel += (
            system.dt * (1 - state.fixed)[..., None] * (k1 + k2 + 4 * k3) / 6
        )

        state.angVel = state.q.rotate_back(state.q, state.angVel)

        if state.dim == 2:
            state.angVel = state.angVel[..., -1:]

        return state, system


__all__ = ["Spiral"]
