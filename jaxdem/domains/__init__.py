# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Simulation domains and boundary-condition implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple

import jax
import jax.numpy as jnp

from ..factory import Factory

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover - fallback for older Python
    from typing_extensions import Self  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Domain(Factory, ABC):
    """Base interface describing the simulation domain and its boundaries."""

    box_size: jax.Array
    """Length of the simulation domain along each dimension."""

    anchor: jax.Array
    """Anchor position (minimum coordinate) of the simulation domain."""

    periodic: ClassVar[bool] = False
    """Whether the domain enforces periodic boundary conditions."""

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: Optional[jax.Array] = None,
        anchor: Optional[jax.Array] = None,
    ) -> Self:
        """Factory helper returning a domain with sensible defaults."""
        if box_size is None:
            box_size = jnp.ones(dim, dtype=float)
        box_size = jnp.asarray(box_size, dtype=float)

        if anchor is None:
            anchor = jnp.zeros_like(box_size, dtype=float)
        anchor = jnp.asarray(anchor, dtype=float)

        assert box_size.shape == anchor.shape
        return cls(box_size=box_size, anchor=anchor)

    @staticmethod
    @abstractmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
        """Return the displacement vector between two positions."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """Apply boundary conditions to the system state."""
        raise NotImplementedError


from .free import FreeDomain  # noqa: E402,F401
from .periodic import PeriodicDomain  # noqa: E402,F401
from .reflect import ReflectDomain  # noqa: E402,F401

__all__ = ["Domain", "FreeDomain", "PeriodicDomain", "ReflectDomain"]
