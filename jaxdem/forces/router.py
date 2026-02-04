# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Force model router selecting laws based on species pairs."""

from __future__ import annotations

import jax

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Tuple, cast
from functools import partial

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

_jit = cast(Callable[..., Any], jax.jit)
_named_call = cast(Callable[..., Any], jax.named_call)

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
    @partial(_named_call, name="ForceRouter.from_dict")
    def from_dict(S: int, mapping: dict[Tuple[int, int], ForceModel]) -> "ForceRouter":
        empty = LawCombiner()  # zero-force default
        m: list[list[ForceModel]] = [[empty for _ in range(S)] for _ in range(S)]
        for (a, b), law in mapping.items():
            m[a][b] = m[b][a] = law
        return ForceRouter(table=tuple(tuple(r) for r in m))

    @staticmethod
    @_jit
    @partial(_named_call, name="ForceRouter.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: "State",
        system: "System",
    ) -> jax.Array:
        si, sj = int(state.species_id[i]), int(state.species_id[j])
        router = cast(ForceRouter, system.force_model)
        law = router.table[si][sj]
        return law.force(i, j, pos, state, system)

    @staticmethod
    @_jit
    @partial(_named_call, name="ForceRouter.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: "State",
        system: "System",
    ) -> jax.Array:
        si, sj = int(state.species_id[i]), int(state.species_id[j])
        router = cast(ForceRouter, system.force_model)
        law = router.table[si][sj]
        return law.energy(i, j, pos, state, system)


__all__ = ["ForceRouter"]
