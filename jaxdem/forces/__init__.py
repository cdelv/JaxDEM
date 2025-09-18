# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Force-law interfaces and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

import jax

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ForceModel(Factory, ABC):
    """Abstract base class for inter-particle force laws."""

    required_material_properties: Tuple[str, ...] = field(
        default=(), metadata={"static": True}
    )
    laws: Tuple["ForceModel", ...] = field(default=(), metadata={"static": True})

    @staticmethod
    @abstractmethod
    @jax.jit
    def force(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """Compute the force acting on particle ``i`` due to ``j``."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def energy(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """Compute the potential energy stored between particles ``i`` and ``j``."""
        raise NotImplementedError


from .law_combiner import LawCombiner  # noqa: E402,F401
from .router import ForceRouter  # noqa: E402,F401
from .spring import SpringForce  # noqa: E402,F401

__all__ = ["ForceModel", "LawCombiner", "ForceRouter", "SpringForce"]
