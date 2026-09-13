# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Optional checked orchestration for simulation steps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .state import State
    from .system import System


class SimulationStatus(IntFlag):
    """JIT-safe status bits; both failure bits may be present."""

    SUCCESS = 0
    SEARCH_OVERFLOW = 1
    NONFINITE = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StepResult:
    """Last accepted state, accepted step count, and persistent failure status."""

    state: State
    system: System
    steps: jax.Array
    status: jax.Array

    def check(self) -> None:
        """Synchronize and raise if any snapshot failed."""
        status = int(jnp.bitwise_or.reduce(jnp.ravel(self.status)))
        if status:
            reasons = []
            if status & SimulationStatus.SEARCH_OVERFLOW:
                reasons.append("spatial search overflow")
            if status & SimulationStatus.NONFINITE:
                reasons.append("nonfinite simulation state")
            raise RuntimeError("Checked simulation stopped: " + ", ".join(reasons))


def simulation_status(state: State, system: System) -> jax.Array:
    """Check particle state, contact history, pending loads, and domain geometry.

    Model parameters can intentionally contain infinity (e.g. disabled plastic
    relaxation). Their physical validity belongs to setup validation; this
    function detects nonfinite particle state, not every invalid parameter.
    """
    values = jax.tree.leaves(
        (
            state,
            system.time,
            system.dt,
            system.domain,
            system.collider,
            system.force_manager,
        )
    )
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in values]))
    overflow = jnp.any(system.search_overflow | system.collider.overflow)
    return overflow.astype(jnp.int32) * int(SimulationStatus.SEARCH_OVERFLOW) | (
        ~finite
    ).astype(jnp.int32) * int(SimulationStatus.NONFINITE)


@jax.jit
def checked_steps(state: State, system: System, n: int) -> StepResult:
    """Single-snapshot implementation; checks stay on device."""
    from .system import _step_once

    initial = StepResult(
        state, system, jnp.asarray(0, jnp.int32), simulation_status(state, system)
    )

    def condition(result: StepResult) -> jax.Array:
        return (result.steps < n) & (result.status == 0)

    def advance(result: StepResult) -> StepResult:
        state, system = _step_once(result.state, result.system)
        status = simulation_status(state, system)
        return jax.lax.cond(
            status == 0,
            lambda: StepResult(state, system, result.steps + 1, status),
            lambda: StepResult(result.state, result.system, result.steps, status),
        )

    return jax.lax.while_loop(condition, advance, initial)
