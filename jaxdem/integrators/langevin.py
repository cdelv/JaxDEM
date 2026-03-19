# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Langevin Integrator"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from . import LinearIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearIntegrator.register("langevin")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Langevin(LinearIntegrator):
    r"""
    TODO: add details

    Parameters
    ----------
    gamma : jax.Array
        Friction coefficient.
    k_B : jax.Array
        Boltzmann constant (set to 1.0 for reduced units).
    temperature : jax.Array
        Target temperature :math:`T`.

    """

    gamma: jax.Array
    k_B: jax.Array
    temperature: jax.Array

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="Langevin.step_before_force")
    def step_before_force(state: State, system: System) -> tuple[State, System]:
        r"""
        TODO: add details

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The updated state and system.

        Note
        -----
        - This method donates state and system

        """
        langevin = cast(Langevin, system.linear_integrator)
        dt = system.dt
        gamma = langevin.gamma
        kT = langevin.k_B * langevin.temperature
        mask = (1 - state.fixed)[..., None]

        accel = state.force / state.mass[..., None]
        state.vel += (dt / 2) * accel * mask

        state.pos_c += (dt / 2) * state.vel

        c1 = jnp.exp(-gamma * dt)
        c2 = jnp.sqrt(kT / state.mass[..., None] * (1.0 - jnp.exp(-2.0 * gamma * dt)))
        system.key, noise_key = jax.random.split(system.key)  # split
        noise = jax.random.normal(noise_key, shape=state.vel.shape, dtype=state.vel.dtype)
        state.vel = (c1 * state.vel + c2 * noise) * mask

        state.pos_c += (dt / 2) * state.vel

        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="Langevin.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        r"""
        TODO: add details

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The updated state and system.

        Note
        -----
        - This method donates state and system

        """
        mask = (1 - state.fixed)[..., None]
        accel = state.force / state.mass[..., None]
        state.vel += (system.dt / 2) * accel * mask
        return state, system


__all__ = ["Langevin"]
