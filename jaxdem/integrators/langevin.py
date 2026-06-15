# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Langevin Integrator"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from . import LinearIntegrator, free_mask

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearIntegrator.register("langevin")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Langevin(LinearIntegrator):
    r"""Langevin thermostat integrator for the translational degrees of freedom.

    Integrates the underdamped Langevin equation

    .. math::
        m\,\dot{\vec{v}} = \vec{F} - m \gamma \vec{v} + \sqrt{2 m \gamma k_B T}\, \vec{\eta}(t)

    using the BAOAB splitting scheme (Leimkuhler & Matthews): half kick (B),
    half drift (A), exact Ornstein-Uhlenbeck update of the velocity (O), half
    drift (A), and a final half kick (B) after the force evaluation. The O-step
    samples the friction and Gaussian noise exactly, driving the system towards
    the canonical distribution at temperature :math:`T`.

    Fixed particles keep their prescribed velocities and are not thermostatted.

    Parameters
    ----------
    gamma : jax.Array
        Friction (collision) coefficient :math:`\gamma` with units of inverse
        time; sets how strongly velocities are damped and rethermalized.
    k_B : jax.Array
        Boltzmann constant (set to 1.0 for reduced units).
    temperature : jax.Array
        Target temperature :math:`T` of the thermostat.

    """

    gamma: jax.Array
    k_B: jax.Array
    temperature: jax.Array

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="Langevin.step_before_force")
    def step_before_force(state: State, system: System) -> tuple[State, System]:
        r"""Perform the BAOA part of the BAOAB step.

        Applies a half kick with the current forces, a half drift, the exact
        Ornstein-Uhlenbeck velocity update

        .. math::
            \vec{v} \leftarrow c_1 \vec{v} + c_2 \vec{\eta}, \qquad
            c_1 = e^{-\gamma \Delta t}, \qquad
            c_2 = \sqrt{\tfrac{k_B T}{m}\left(1 - e^{-2\gamma \Delta t}\right)}

        with :math:`\vec{\eta} \sim \mathcal{N}(0, 1)`, and a second half
        drift. ``system.key`` is split to draw the noise. Fixed particles keep
        their prescribed velocities.

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

        """
        langevin = cast(Langevin, system.linear_integrator)
        dt = system.dt
        gamma = langevin.gamma
        kT = langevin.k_B * langevin.temperature
        mask = free_mask(state)

        accel = state.force / state.mass[..., None]
        state.vel += (dt / 2) * accel * mask

        state.pos_c += (dt / 2) * state.vel

        c1 = jnp.exp(-gamma * dt)
        c2 = jnp.sqrt(kT / state.mass * (1.0 - jnp.exp(-2.0 * gamma * dt)))[..., None]
        system.key, noise_key = jax.random.split(system.key)  # split
        noise = jax.random.normal(
            noise_key, shape=state.vel.shape, dtype=state.vel.dtype
        )
        # O-step: thermostat only the free particles; fixed particles keep their
        # prescribed velocities (multiplying by the mask would permanently zero them).
        state.vel = jnp.where(mask, c1 * state.vel + c2 * noise, state.vel)

        state.pos_c += (dt / 2) * state.vel

        return state, system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="Langevin.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        r"""Perform the final B (half kick) part of the BAOAB step.

        Updates the velocities of free particles with the freshly computed
        forces: :math:`\vec{v} \leftarrow \vec{v} + \tfrac{\Delta t}{2} \vec{F}/m`.

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

        """
        mask = free_mask(state)
        accel = state.force / state.mass[..., None]
        state.vel += (system.dt / 2) * accel * mask
        return state, system


__all__ = ["Langevin"]
