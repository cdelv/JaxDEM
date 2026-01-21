# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Composite force model that sums multiple force laws."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING
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

    laws: Tuple["ForceModel", ...] = field(default=(), metadata={"static": True})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_material_properties",
            tuple(
                sorted({p for lw in self.laws for p in lw.required_material_properties})
            ),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LawCombiner.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: "State",
        system: "System",
    ) -> Tuple[jax.Array, jax.Array]:
        force = jnp.zeros_like(state.pos[i])
        torque = jnp.zeros_like(state.angVel[i])
        for law in system.force_model.laws:
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
        state: "State",
        system: "System",
    ) -> jax.Array:
        e = jnp.zeros(state.N)
        for law in system.force_model.laws:
            e += law.energy(i, j, pos, state, system)
        return e


__all__ = ["LawCombiner"]
