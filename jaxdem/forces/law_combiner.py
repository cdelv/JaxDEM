# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Composite force model that sums multiple force laws."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Tuple
from functools import partial

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
    def force(i, j, state, system):
        force = jnp.zeros_like(jnp.take(state.pos, i, axis=-2))
        torque = jnp.zeros_like(jnp.take(state.angVel, i, axis=-2))
        for law in system.force_model.laws:
            f, t = law.force(i, j, state, system)
            force = force + f
            torque = torque + t
        return force, torque

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LawCombiner.energy")
    def energy(i, j, state, system):
        e = 0.0
        for law in system.force_model.laws:
            e = e + law.energy(i, j, state, system)
        return e


__all__ = ["LawCombiner"]
