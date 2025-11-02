# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Direct (forward) Euler integrator."""

from __future__ import annotations

import jax

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import Integrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Integrator.register("euler")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DirectEuler(Integrator):
    """
    Implements the explicit (forward) Euler integration method.
    """

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="DirectEuler.step")
    def step(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Advances the simulation state by one time step using the Direct Euler method.

        The update equations are:

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
        system.time += system.dt
        system.step_count += 1
        state, system = system.domain.shift(state, system)
        state, system = system.force_manager.apply(state, system)
        state, system = system.collider.compute_force(state, system)
        state.vel += system.dt * state.accel * (1 - state.fixed)[..., None]
        state.pos += system.dt * state.vel * (1 - state.fixed)[..., None]
        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="DirectEuler.initialize")
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        The Direct Euler integrator does not require a specific initialization step.


        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The original `State` and `System` objects.
        """
        return state, system


__all__ = ["DirectEuler"]
