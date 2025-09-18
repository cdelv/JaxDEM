# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Periodic boundary-condition domain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("periodic")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class PeriodicDomain(Domain):
    """A `Domain` implementation enforcing periodic boundary conditions."""

    periodic: bool = True

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
        rij = ri - rj
        return rij - system.domain.box_size * jnp.floor(
            rij / system.domain.box_size + 0.5
        )

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        pos = state.pos - system.domain.box_size * jnp.floor(
            (state.pos - system.domain.anchor) / system.domain.box_size
        )
        state = replace(state, pos=pos)
        return state, system


__all__ = ["PeriodicDomain"]
