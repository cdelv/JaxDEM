# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Unbounded (free) simulation domain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("free")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class FreeDomain(Domain):
    """A `Domain` implementation representing an unbounded, free space."""

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        return ri - rj

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        p_min = jnp.min(state.pos - state.rad[..., None], axis=-2)
        p_max = jnp.max(state.pos + state.rad[..., None], axis=-2)
        domain = replace(system.domain, box_size=p_max - p_min, anchor=p_min)
        return state, replace(system, domain=domain)


__all__ = ["FreeDomain"]
