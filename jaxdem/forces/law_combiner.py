# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Composite force model that sums multiple force laws."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Tuple, cast
from functools import partial

if TYPE_CHECKING:
    from ..state import State
    from ..system import System

from . import ForceModel


@ForceModel.register("lawcombiner")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LawCombiner(ForceModel):
    """Sum a tuple of elementary force laws."""

    laws: Tuple[ForceModel, ...] = field(default=(), metadata={"static": True})

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        return tuple(
            sorted({p for lw in self.laws for p in lw.required_material_properties})
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LawCombiner.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> Tuple[jax.Array, jax.Array]:
        rij = system.domain.displacement(pos[i], pos[j], system)
        force = jnp.zeros_like(rij)
        torque = jnp.zeros(rij.shape[:-1] + state.ang_vel.shape[-1:])
        combiner = cast(LawCombiner, system.force_model)
        for law in combiner.laws:
            f, t = law.force(i, j, pos, state, system)
            force += f
            torque += t
        return force, torque

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LawCombiner.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        e = jnp.zeros(state.N)
        combiner = cast(LawCombiner, system.force_model)
        for law in combiner.laws:
            e += law.energy(i, j, pos, state, system)
        return e


__all__ = ["LawCombiner"]
