# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force-law interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceModel(Factory, ABC):
    """Abstract base class for inter-particle force laws and their potential energies.

    Concrete subclasses implement specific force and energy models, such as
    linear springs and Hertzian contacts.

    Notes:
    ------
    - The :meth:`force` and :meth:`energy` methods must handle the case where
      `i` and `j` refer to the same particle (`i == j`). Self-interaction
      calls can occur.

    Example:
    --------
    To define a custom force model, inherit from :class:`ForceModel` and implement
    its abstract methods:

    >>> @ForceModel.register("myCustomForce")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomForce(ForceModel):
            ...

    """

    laws: tuple[ForceModel, ...] = jax.tree.static(default=())
    """
    Static tuple of other :class:`ForceModel` instances that compose this force model.

    Use it to build composite force models, for example a spring force plus a
    damping force.
    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the force and torque on particle :math:`i` from particle :math:`j`.

        Parameters
        ----------
        i : int
            Index of the first particle (on which the interaction acts).
        j : int
            Index of the second particle (which exerts the interaction).
        pos : jax.Array
            Particle positions.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            A tuple ``(force, torque)`` where ``force`` has shape ``(dim,)`` and ``torque`` has shape ``(1,)`` in 2D or ``(3,)`` in 3D.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        """Compute the potential energy of the interaction between particle :math:`i` and particle :math:`j`.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        pos : jax.Array
            Particle positions.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Scalar potential energy of the interaction between particles
            :math:`i` and :math:`j`.

        """
        raise NotImplementedError

    @property
    def requires_history(self) -> bool:
        """Whether this force model needs persistent pair history."""
        return False

    def init_history(self, shape: tuple[int, ...]) -> Any:
        """Initialize the history variables for this force model.

        Parameters
        ----------
        shape : tuple[int, ...]
            The expected shape for pair-wise quantities, typically `(..., N, max_neighbors)`.

        Returns
        -------
        Any
            A PyTree of initialized JAX arrays, or None by default.
        """
        return None

    @staticmethod
    @jax.jit
    def force_and_history(
        i: int, j: int, pos: jax.Array, state: State, system: System, history: Any
    ) -> tuple[jax.Array, jax.Array, Any]:
        """Compute the force and torque, and update history.

        By default, this calls `force` and returns `history` unchanged.
        """
        f, t = system.force_model.force(i, j, pos, state, system)
        return f, t, history

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """Names of the material properties this force model needs.

        Each name (for example 'young_eff' or 'restitution') must be present
        in :attr:`System.mat_table`. Used for validation.
        """
        return ()


from .cundall_strack import CundallStrackForce
from .force_manager import ForceManager
from .hertz import HertzianForce
from .law_combiner import LawCombiner
from .lennardjones import LennardJones
from .router import ForceRouter
from .spring import FacetFacetSpringForce, SphereFacetSpringForce, SpringForce
from .wca import WCA
from .wca_shifted import WCAShifted

__all__ = [
    "WCA",
    "CundallStrackForce",
    "ForceManager",
    "ForceModel",
    "ForceRouter",
    "HertzianForce",
    "LawCombiner",
    "LennardJones",
    "SpringForce",
    "WCAShifted",
    "SphereFacetSpringForce",
    "FacetFacetSpringForce",
]
