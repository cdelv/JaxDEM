# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Time-integration interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import jax

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Integrator(Factory, ABC):
    """Abstract base class for defining the interface for time-stepping."""

    @staticmethod
    @abstractmethod
    @jax.jit
    def step(state: "State", system: "System") -> Tuple["State", "System"]:
        """Advance the simulation by one step using a concrete integrator."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """Perform any integrator-specific initialization."""
        raise NotImplementedError


from .direct_euler import DirectEuler  # noqa: E402,F401

__all__ = ["Integrator", "DirectEuler"]
