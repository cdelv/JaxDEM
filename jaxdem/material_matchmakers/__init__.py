# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Material mix rules and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class MaterialMatchmaker(Factory, ABC):
    """Abstract base class for combining material properties."""

    @staticmethod
    @abstractmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        """Return the effective property resulting from combining ``prop1`` and ``prop2``."""
        raise NotImplementedError


from .harmonic import HarmonicMaterialMatchmaker  # noqa: E402,F401
from .linear import LinearMaterialMatchmaker  # noqa: E402,F401

__all__ = [
    "MaterialMatchmaker",
    "HarmonicMaterialMatchmaker",
    "LinearMaterialMatchmaker",
]
