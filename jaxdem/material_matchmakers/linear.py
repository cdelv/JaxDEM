# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Linear averaging material matchmaker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from . import MaterialMatchmaker

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp


@MaterialMatchmaker.register("linear")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class LinearMaterialMatchmaker(MaterialMatchmaker):
    """Compute the arithmetic mean of two material properties."""

    @staticmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return (prop1 + prop2) / 2


__all__ = ["LinearMaterialMatchmaker"]
