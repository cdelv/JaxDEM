# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Linear spring force model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class SpringForce(ForceModel):
    """Linear spring-like interaction between particles."""

    required_material_properties: Tuple[str, ...] = field(
        default=("young_eff",), metadata={"static": True}
    )

    @staticmethod
    @jax.jit
    def force(i: int, j: int, state: "State", system: "System") -> jax.Array:
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]

        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r2 = jnp.dot(rij, rij)
        r = jnp.sqrt(r2 + jnp.finfo(state.pos.dtype).eps)
        R = state.rad[i] + state.rad[j]
        s = jnp.maximum(0.0, R / r - 1.0)
        return k * s * rij

    @staticmethod
    @jax.jit
    def energy(i: int, j: int, state: "State", system: "System") -> jax.Array:
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]

        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r2 = jnp.dot(rij, rij)
        r = jnp.sqrt(r2)
        R = state.rad[i] + state.rad[j]
        s = jnp.maximum(0.0, R - r)
        return 0.5 * k * s**2


__all__ = ["SpringForce"]
