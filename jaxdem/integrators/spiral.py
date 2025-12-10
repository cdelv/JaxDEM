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
from ..utils.geometric_algebra import Rotor, Bivector

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="spiral.omega_dot")
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
    # L = w * inertia
    return ang_accel - Bivector.commutator(w, w) * 0.5


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
        state.angVel = Rotor.rotate_bivector(state.q, state.angVel)
        torque = Rotor.rotate_bivector(state.q, state.torque)

        ang_accel = torque  # * (1 / state.inertia)[..., None, None]
        w_dot = omega_dot(state.angVel, ang_accel, state.inertia)

        state.q *= Bivector.exp(0.5 * system.dt * state.angVel) * Bivector.exp(
            0.25 * jnp.square(system.dt) * w_dot
        )

        k1 = system.dt * w_dot
        k2 = system.dt * omega_dot(state.angVel + k1, ang_accel, state.inertia)
        k3 = system.dt * omega_dot(
            state.angVel + 0.25 * (k1 + k2), ang_accel, state.inertia
        )
        state.angVel += (
            system.dt * (1 - state.fixed)[..., None, None] * (k1 + k2 + 4 * k3) / 6
        )

        state.angVel = Rotor.rotate_back_bivector(state.q, state.angVel)
        return state, system


__all__ = ["Spiral"]
