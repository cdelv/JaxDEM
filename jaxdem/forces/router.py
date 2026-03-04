# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force model router selecting laws based on species pairs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Tuple, cast
from functools import partial

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

from . import ForceModel
from .law_combiner import LawCombiner


@ForceModel.register("forcerouter")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceRouter(ForceModel):
    """Static species-to-force lookup table."""

    table: Tuple[Tuple[ForceModel, ...], ...] = field(default=(()))

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                {
                    p
                    for row in self.table
                    for law in row
                    for p in law.required_material_properties
                }
            )
        )

    @staticmethod
    @partial(jax.named_call, name="ForceRouter.from_dict")
    def from_dict(S: int, mapping: dict[Tuple[int, int], ForceModel]) -> ForceRouter:
        empty = LawCombiner()  # zero-force default
        m: list[list[ForceModel]] = [[empty for _ in range(S)] for _ in range(S)]
        for (a, b), law in mapping.items():
            m[a][b] = m[b][a] = law
        return ForceRouter(table=tuple(tuple(r) for r in m))

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceRouter.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> Tuple[jax.Array, jax.Array]:
        router = cast(ForceRouter, system.force_model)
        S = len(router.table)

        all_f = []
        all_t = []
        for a in range(S):
            for b in range(S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)
                f, t = law.force(i, j, pos, state, sys_law)
                all_f.append(f)
                all_t.append(t)

        f_shape = jnp.broadcast_shapes(*(f.shape for f in all_f))
        t_trail = jnp.broadcast_shapes(*(t.shape[-1:] for t in all_t))
        t_shape = f_shape[:-1] + t_trail
        stacked_f = jnp.stack([jnp.broadcast_to(f, f_shape) for f in all_f])
        stacked_t = jnp.stack([jnp.broadcast_to(t, t_shape) for t in all_t])

        si = state.species_id[i]
        sj = state.species_id[j]
        idx = si * S + sj
        n_idx = jnp.arange(stacked_f.shape[1])
        return stacked_f[idx, n_idx], stacked_t[idx, n_idx]

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceRouter.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        router = cast(ForceRouter, system.force_model)
        S = len(router.table)

        all_e = []
        for a in range(S):
            for b in range(S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)
                e = law.energy(i, j, pos, state, sys_law)
                all_e.append(e)

        e_shape = jnp.broadcast_shapes(*(e.shape for e in all_e))
        stacked_e = jnp.stack([jnp.broadcast_to(e, e_shape) for e in all_e])

        si = state.species_id[i]
        sj = state.species_id[j]
        idx = si * S + sj
        n_idx = jnp.arange(stacked_e.shape[-1])
        return stacked_e[idx, n_idx]

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceRouter.stiffness")
    def stiffness(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        router = cast(ForceRouter, system.force_model)
        S = len(router.table)

        all_c = []
        for a in range(S):
            for b in range(S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)
                c = law.stiffness(i, j, pos, state, sys_law)
                all_c.append(c)

        c_shape = jnp.broadcast_shapes(*(c.shape for c in all_c))
        stacked_c = jnp.stack([jnp.broadcast_to(c, c_shape) for c in all_c])

        si = state.species_id[i]
        sj = state.species_id[j]
        idx = si * S + sj
        n_idx = jnp.arange(stacked_c.shape[-1])
        return stacked_c[idx, n_idx]


__all__ = ["ForceRouter"]
