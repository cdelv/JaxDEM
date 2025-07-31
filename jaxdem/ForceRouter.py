# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Tuple

from .Forces import ForceModel

@ForceModel.register("lawcombiner")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LawCombiner(ForceModel):
    """Sum a tuple of elementary laws."""
    required_material_properties: Tuple[str, ...] = field(default=(), metadata={"static": True})
    laws: Tuple["ForceModel", ...] = field(default=(), metadata={"static": True})

    def __post_init__(self):
        object.__setattr__(
            self, "required_material_properties",
            tuple(sorted({p for lw in self.laws for p in lw.required_material_properties}))
        )

    # change to tree_map + reduce
    @staticmethod
    @jax.jit
    def force(i, j, state, system):
        out = jnp.zeros_like(state.pos[0])
        for law in system.force_model.laws:     # python loop, traced once
            out = out + law.force(i, j, state, system)
        return out

    # change to tree_map + reduce
    @staticmethod
    @jax.jit
    def energy(i, j, state, system):
        e = 0.0
        for law in system.force_model.laws:
            e = e + law.energy(i, j, state, system)
        return e


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceRouter(ForceModel):
    """
    Static (S×S) table that maps species pairs to a ForceModel.
    """
    table: Tuple[Tuple["ForceModel", ...], ...] = field(default=(()))
    required_material_properties: Tuple[str, ...] = field(default=(), metadata={"static": True})

    def __post_init__(self):
        req = {p for row in self.table for law in row for p in law.required_material_properties}
        object.__setattr__(self, "required_material_properties", tuple(sorted(req)))

    @staticmethod
    def from_dict(S: int, mapping: dict[Tuple[int, int], ForceModel]):
        empty = LawCombiner()                   # zero-force default
        m = [[empty for _ in range(S)] for _ in range(S)]
        for (a, b), law in mapping.items():
            m[a][b] = m[b][a] = law
        return ForceRouter(table=tuple(tuple(r) for r in m))

    @staticmethod
    @jax.jit
    def force(i, j, state, system):
        si, sj = int(state.species_id[i]), int(state.species_id[j])
        law = system.force_model.table[si][sj]
        return law.force(i, j, state, system)

    @staticmethod
    @jax.jit
    def energy(i, j, state, system):
        si, sj = int(state.species_id[i]), int(state.species_id[j])
        law = system.force_model.table[si][sj]
        return law.energy(i, j, state, system)