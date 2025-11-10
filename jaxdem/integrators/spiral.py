# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Direct (forward) Integrator."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import RotationIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
def omega_dot(w: jax.Array, a: jax.Array, I: jax.Array) -> jax.Array:
    """
    w, a, I: shape (N, D) with D in {1, 3}.
    Returns shape (N, D).
    Computes:  ω̇ = a − (ω × (I ∘ ω)) ∘ I^{-1}  (elementwise I for diagonal inertia)
    """
    D = w.shape[-1]
    if D == 1:
        return a

    if D == 3:
        term = jnp.cross(w, I * w) / I
        return a - term

    raise ValueError(f"omega_dot supports D in {{1,3}}, got D={D}")


@RotationIntegrator.register("spiral")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Spiral(RotationIntegrator):
    """
    Implements the non-leapfrog version of the spiral algorithm (https://doi.org/10.1016/j.cpc.2023.109077).
    """

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="spiral.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Advances the simulation state by one time step after the torque calculation using the non-leapfrog spiral method.

        The update equations are: TO DO: FIX EQUATIONS

        .. math::
            & v(t + \\Delta t) &= v(t) + \\Delta t a(t) \\\\
            & r(t + \\Delta t) &= r(t) + \\Delta t v(t + \\Delta t)

        where:
            - :math:`r` is the particle position (:attr:`jaxdem.State.pos`)
            - :math:`v` is the particle velocity (:attr:`jaxdem.State.vel`)
            - :math:`a` is the particle acceleration (:attr:`jaxdem.State.accel`)
            - :math:`\\Delta t` is the time step (:attr:`jaxdem.System.dt`)

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
        """
        # q(t + dt) = q(t)*exp(dt/2*w + dt^2/4*w_dot)
        k1 = system.dt * omega_dot(state.angVel, state.angAccel, state.inertia)
        k2 = system.dt * omega_dot(state.angVel, state.angAccel, state.inertia)
        k3 = system.dt * omega_dot(state.angVel, state.angAccel, state.inertia)
        state.angVel += (
            system.dt * (1 - state.fixed)[..., None] * (k1 + k2 + 4 * k3) / 6
        )
        return state, system


__all__ = ["Spiral"]
