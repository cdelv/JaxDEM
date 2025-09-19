# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Composite force model that sums multiple force laws."""

from __future__ import annotations

import jax

from dataclasses import dataclass, field
from typing import Tuple

from . import ForceModel


@ForceModel.register("lawcombiner")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class LawCombiner(ForceModel):
    """Sum a tuple of elementary force laws."""

    required_material_properties: Tuple[str, ...] = field(
        default=(), metadata={"static": True}
    )
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
    def force(i, j, state, system):
        return jax.tree.reduce(
            lambda a, b: a + b,
            tuple(law.force(i, j, state, system) for law in system.force_model.laws),
        )

    @staticmethod
    @jax.jit
    def energy(i, j, state, system):
        e = 0.0
        for law in system.force_model.laws:
            e = e + law.energy(i, j, state, system)
        return e


__all__ = ["LawCombiner"]
