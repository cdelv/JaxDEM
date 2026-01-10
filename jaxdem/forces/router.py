# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Force model router selecting laws based on species pairs."""

from __future__ import annotations

import jax

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

from . import ForceModel
from .law_combiner import LawCombiner


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceRouter(ForceModel):
    """Static species-to-force lookup table."""

    table: Tuple[Tuple["ForceModel", ...], ...] = field(default=(()))

    def __post_init__(self) -> None:
        req = {
            p
            for row in self.table
            for law in row
            for p in law.required_material_properties
        }
        object.__setattr__(self, "required_material_properties", tuple(sorted(req)))

    @staticmethod
    @partial(jax.named_call, name="ForceRouter.from_dict")
    def from_dict(S: int, mapping: dict[Tuple[int, int], ForceModel]) -> "ForceRouter":
        empty = LawCombiner()  # zero-force default
        m = [[empty for _ in range(S)] for _ in range(S)]
        for (a, b), law in mapping.items():
            m[a][b] = m[b][a] = law
        return ForceRouter(table=tuple(tuple(r) for r in m))

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceRouter.force")
    def force(
        i: int,
        j: int,
        state: "State",
        system: "System",
    ) -> "Array":
        si, sj = int(state.species_id[i]), int(state.species_id[j])
        law = system.force_model.table[si][sj]
        return law.force(i, j, state, system)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceRouter.energy")
    def energy(
        i: int,
        j: int,
        state: "State",
        system: "System",
    ) -> "jax.Array":
        si, sj = int(state.species_id[i]), int(state.species_id[j])
        law = system.force_model.table[si][sj]
        return law.energy(i, j, state, system)


__all__ = ["ForceRouter"]
