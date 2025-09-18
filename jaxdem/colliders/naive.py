# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Naïve :math:`O(N^2)` collider implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Tuple

import jax

from . import Collider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Collider.register("naive")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class NaiveSimulator(Collider):
    """Compute pairwise forces with a dense all-pairs loop."""

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        rng = jax.lax.iota(dtype=int, size=state.N)
        return jax.vmap(
            lambda i, j, st, sys: jax.vmap(
                sys.force_model.energy, in_axes=(None, 0, None, None)
            )(i, j, st, sys).sum(axis=0),
            in_axes=(0, None, None, None),
        )(rng, rng, state, system)

    @staticmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        rng = jax.lax.iota(dtype=int, size=state.N)
        accel = state.accel + (
            jax.vmap(
                lambda i, j, st, sys: jax.vmap(
                    sys.force_model.force, in_axes=(None, 0, None, None)
                )(i, j, st, sys).sum(axis=0),
                in_axes=(0, None, None, None),
            )(rng, rng, state, system)
            / state.mass[:, None]
        )
        state = replace(state, accel=accel)
        return state, system


__all__ = ["NaiveSimulator"]
