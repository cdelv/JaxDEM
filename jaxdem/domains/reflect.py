# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("reflect")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ReflectDomain(Domain):
    """A `Domain` implementation with reflective boundary conditions."""

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        return ri - rj

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        lo = system.domain.anchor + state.rad[:, None]
        hi = system.domain.anchor + system.domain.box_size - state.rad[:, None]

        over_lo = jnp.maximum(0.0, lo - state.pos)
        over_hi = jnp.maximum(0.0, state.pos - hi)

        x_ref = state.pos + 2.0 * over_lo - 2.0 * over_hi

        hit = jnp.logical_or((over_lo > 0), (over_hi > 0))
        sign = 1.0 - 2.0 * hit
        v_ref = state.vel * sign

        return replace(state, pos=x_ref, vel=v_ref), system


__all__ = ["ReflectDomain"]
