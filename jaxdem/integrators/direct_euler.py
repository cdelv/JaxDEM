# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Direct (semi-implicit / symplectic Euler) Integrator."""

from __future__ import annotations

import jax

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from . import LinearIntegrator, free_mask

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearIntegrator.register("euler")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DirectEuler(LinearIntegrator):
    """Implements the semi-implicit (symplectic) Euler integration method.

    The velocity is updated first and the position update then uses the *new*
    velocity, which makes the scheme symplectic (first-order accurate, but with
    bounded energy error), unlike a true forward Euler step.
    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="DirectEuler.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        r"""Advances the simulation state by one time step after the force calculation using the semi-implicit (symplectic) Euler method.

        The update equations are (note the position update uses the *updated* velocity):

        .. math::
            & v(t + \Delta t) &= v(t) + \Delta t a(t) \\
            & r(t + \Delta t) &= r(t) + \Delta t v(t + \Delta t)

        where:
            - :math:`r` is the particle position (:attr:`jaxdem.State.pos`)
            - :math:`v` is the particle velocity (:attr:`jaxdem.State.vel`)
            - :math:`a` is the particle acceleration computed from forces (:attr:`jaxdem.State.force`)
            - :math:`\Delta t` is the time step (:attr:`jaxdem.System.dt`)

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
        state.vel += (
            state.force * (system.dt / state.mass)[..., None] * free_mask(state)
        )
        state.pos_c += system.dt * state.vel
        return state, system


__all__ = ["DirectEuler"]
