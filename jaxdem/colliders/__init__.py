# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Collision-detection interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import jax

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Collider(Factory, ABC):
    """Abstract base class for contact detection and force computation."""

    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """Populate per-particle accelerations from the current configuration."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        """Return the total potential energy of the system."""
        raise NotImplementedError


from .naive import NaiveSimulator  # noqa: E402,F401

__all__ = ["Collider", "NaiveSimulator"]
