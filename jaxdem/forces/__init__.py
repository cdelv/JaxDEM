# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force-law interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

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

    laws: tuple[ForceModel, ...] = ()
    """
    Tuple of other :class:`ForceModel` instances that compose this force model.

    Use it to build composite force models, for example a spring force plus a
    damping force.
    """

    @property
    def supports_analytical_energy_gradient(self) -> bool:
        """Whether force is the negative analytical gradient of ``energy``.

        Translational components use center positions and rotational components
        use the minimizer's incremental rotation coordinates. Force laws are
        assumed to satisfy this contract by default. Laws that do not must
        override this property with ``False``; they require an explicit target
        for minimization.
        """
        return True

    @property
    def species_capacity(self) -> int | None:
        """Smallest species-table capacity reachable through this law.

        Ordinary laws do not constrain species identifiers. Composite laws
        override this capability so callers need not inspect concrete types.
        """
        return None

    @staticmethod
    @abstractmethod
    @jax.jit
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
        *,
        advance_history: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
        Tuple[jax.Array, jax.Array, jax.Array]
            ``(force, torque, history)``. The history output is unchanged for
            stateless laws and when ``advance_history=False``.

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

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Return conservative radii for a single search snapshot.

        A nonzero pair interaction must fit within the sum of the two bounds.
        Custom finite-range laws must override this method to use accelerated
        colliders; the naive all-pairs collider does not require a bound.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must declare search_radii for spatial search."
        )

    def history_shape(self, dim: int) -> tuple[int, ...]:
        """Trailing shape of one pair's history; stateless laws use ``(0,)``.

        Composite laws flatten child histories internally and restore this
        shape before invoking the child law.
        """
        return (0,)

    def init_history(self, pair_shape: tuple[int, ...], dim: int) -> jax.Array:
        """Initialize the history variables for this force model.

        Parameters
        ----------
        pair_shape : tuple[int, ...]
            Leading shape of pair-wise quantities, typically
            ``(N, max_neighbors)``.
        dim : int
            Spatial dimension used by dimension-dependent history layouts.

        Returns
        -------
        jax.Array
            A zero-filled array with shape
            ``pair_shape + history_shape(dim)``.
        """
        return jnp.zeros(pair_shape + self.history_shape(dim))

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
