# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Harmonic averaging material matchmaker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from . import MaterialMatchmaker

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp


@MaterialMatchmaker.register("harmonic")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class HarmonicMaterialMatchmaker(MaterialMatchmaker):
    """Compute the harmonic mean of two material properties."""

    @staticmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return 2.0 / (1.0 / prop1 + 1.0 / prop2)


__all__ = ["HarmonicMaterialMatchmaker"]
