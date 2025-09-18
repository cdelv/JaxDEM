# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Direct (forward) Euler integrator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Tuple

import jax

from . import Integrator

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from ..state import State
    from ..system import System


@Integrator.register("euler")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class DirectEuler(Integrator):
    """Implements the explicit (forward) Euler integration method."""

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    def step(state: "State", system: "System") -> Tuple["State", "System"]:
        state, system = system.domain.shift(state, system)
        state, system = system.collider.compute_force(state, system)
        state = replace(
            state,
            vel=state.vel + system.dt * state.accel * (1 - state.fixed)[..., None],
        )
        state = replace(
            state,
            pos=state.pos + system.dt * state.vel * (1 - state.fixed)[..., None],
        )
        system = replace(
            system, time=system.time + system.dt, step_count=system.step_count + 1
        )
        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        return state, system


__all__ = ["DirectEuler"]
