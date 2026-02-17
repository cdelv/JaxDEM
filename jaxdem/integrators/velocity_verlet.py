# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Velocity Verlet Integrator."""

from __future__ import annotations

import jax

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import LinearIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearIntegrator.register("verlet")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VelocityVerlet(LinearIntegrator):
    """
    Implements the Velocity Verlet integration method.
    """

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="VelocityVerlet.step_after_force")
    def step_before_force(state: State, system: System) -> Tuple[State, System]:
        """
        Advances the simulation state by one half-step before the force calculation using the Velocity Verlet scheme.

        The update equations are:

        .. math::
            & v(t + \\Delta t / 2) &= v(t) + \\Delta t a(t) / 2 \\\\
            & r(t + \\Delta t) &= r(t) + \\Delta t v(t + \\Delta t / 2) \\Delta t

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

        Note
        -----
        - This method donates state and system
        """
        accel = state.force / state.mass[..., None]
        state.vel += accel * (1 - state.fixed)[..., None] * system.dt / 2
        state.pos_c += state.vel * system.dt
        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="VelocityVerlet.step_after_force")
    def step_after_force(state: State, system: System) -> Tuple[State, System]:
        """
        Advances the simulation state by one half-step after the force calculation using the Velocity Verlet scheme.

        The update equations are:

        .. math::
            & v(t + \\Delta t) &= v(t + \\Delta t / 2) + \\Delta t a(t) / 2 \\\\

        where:
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

        Note
        -----
        - This method donates state and system
        """
        accel = state.force / state.mass[..., None]
        state.vel += accel * (1 - state.fixed)[..., None] * system.dt / 2
        return state, system


__all__ = ["VelocityVerlet"]
