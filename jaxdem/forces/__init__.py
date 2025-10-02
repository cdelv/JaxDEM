# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Force-law interfaces and concrete implementations."""

from __future__ import annotations

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ForceModel(Factory, ABC):
    """
    Abstract base class for defining inter-particle force laws and their corresponding potential energies.

    Concrete subclasses implement specific force and energy models, such as
    linear springs, Hertzian contacts, etc.

    Notes
    -----
    - The :meth:`force` and :meth:`energy` methods should correctly handle the
      case where `i` and `j` refer to the same particle (i.e., `i == j`).
      There is no guarantee that self-interaction calls will not occur.

    Example
    -------
    To define a custom force model, inherit from :class:`ForceModel` and implement
    its abstract methods:

    >>> @ForceModel.register("myCustomForce")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True, frozen=True)
    >>> class MyCustomForce(ForceModel):
            ...
    """

    required_material_properties: Tuple[str, ...] = field(
        default=(), metadata={"static": True}
    )
    """
    A static tuple of strings specifying the material properties required by this force model.

    These properties (e.g., 'young_eff', 'restitution', ...) must be present in the
    :attr:`System.mat_table` for the model to function correctly. This is used
    for validation.
    """

    laws: Tuple["ForceModel", ...] = field(default=(), metadata={"static": True})
    """
    A static tuple of other :class:`ForceModel` instances that compose this force model.

    This allows for creating composite force models (e.g., a total force being
    the sum of a spring force and a damping force).
    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def force(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """
        Compute the force vector acting on particle :math:`i` due to particle :math:`j`.

        Parameters
        ----------
        i : int
            Index of the first particle (on which the force is acting).
        j : int
            Index of the second particle (which is exerting the force).
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Force vector acting on particle :math:`i` due to particle :math:`j`. Shape `(dim,)`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def energy(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """
        Compute the potential energy of the interaction between particle :math:`i` and particle :math:`j`.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Scalar JAX array representing the potential energy of the interaction
            between particles :math:`i` and :math:`j`.
        """
        raise NotImplementedError


from .law_combiner import LawCombiner
from .router import ForceRouter
from .spring import SpringForce
from .force_manager import ForceManager

__all__ = ["ForceModel", "LawCombiner", "ForceRouter", "SpringForce", "ForceManager"]
