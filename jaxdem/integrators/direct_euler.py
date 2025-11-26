# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Direct (forward) Integrator."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import LinearIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearIntegrator.register("euler")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DirectEuler(LinearIntegrator):
    """
    Implements the explicit (forward) Euler integration method.
    """

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="DirectEuler.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Advances the simulation state by one time step after the force calculation using the Direct Euler method.

        The update equations are:

        .. math::
            & v(t + \\Delta t) &= v(t) + \\Delta t a(t) \\\\
            & r(t + \\Delta t) &= r(t) + \\Delta t v(t + \\Delta t)

        where:
            - :math:`r` is the particle position (:attr:`jaxdem.State.pos`)
            - :math:`v` is the particle velocity (:attr:`jaxdem.State.vel`)
            - :math:`a` is the particle acceleration computed from forces (:attr:`jaxdem.State.force`)
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
        accel = state.force / state.mass[..., None]
        state.vel += system.dt * accel * (1 - state.fixed)[..., None]
        state.pos_c += system.dt * state.vel * (1 - state.fixed)[..., None]
        state.pos_p += (
            system.dt
            * (state.vel + jnp.cross(state.angVel, state.pos_p))
            * (1 - state.fixed)[..., None]
        )
        return state, system


__all__ = ["DirectEuler"]
