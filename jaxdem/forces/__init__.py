# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force-law interfaces."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


_FORCE_OPTIONS = ("history", "advance_history", "quasistatic")


def _declared_options(fn: Callable[..., Any], names: Iterable[str]) -> frozenset[str]:
    """Return which of ``names`` the pair kernel ``fn`` declares as parameters."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - uninspectable callable
        return frozenset(names)
    return frozenset(name for name in names if name in params)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceModel(Factory, ABC):
    """Abstract base class for inter-particle force laws and their potential energies.

    Concrete subclasses implement specific force and energy models, such as
    linear springs and Hertzian contacts.

    A law implements :meth:`force` and :meth:`energy` with only the arguments
    it needs. A stateless, rate-independent law takes
    ``(i, j, pos, state, system)`` and returns ``(force, torque)``. A law with
    per-pair memory also declares ``history_shape``, takes ``history`` and
    the keyword ``advance_history``, and returns ``(force, torque, history)``.
    A law with velocity-dependent terms takes the keyword ``quasistatic``
    and switches those terms off when it is true. Colliders always call
    :meth:`pair_force` and :meth:`pair_energy`, which pass each law exactly
    the arguments its signature declares.

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
        override this property with ``False``.
        """
        return True

    @property
    def species_capacity(self) -> int | None:
        """Smallest species-table capacity reachable through this law.

        Ordinary laws do not constrain species identifiers. Composite laws
        override this capability so callers need not inspect concrete types.
        """
        return None

    # Which optional arguments each law's kernels declare. Class-level data,
    # filled in by __init_subclass__; not dataclass fields.
    _force_options: ClassVar[frozenset[str]] = frozenset()
    _energy_options: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Two-arg super: dataclass(slots=True) recreates the class, so the
        # zero-arg form's __class__ cell points at the discarded original.
        super(ForceModel, cls).__init_subclass__(**kwargs)
        cls._force_options = _declared_options(cls.force, _FORCE_OPTIONS)
        cls._energy_options = _declared_options(cls.energy, ("history",))

    @staticmethod
    @abstractmethod
    @jax.jit
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the force and torque on particle :math:`i` from particle :math:`j`.

        This is the stateless, rate-independent form. Laws with pair memory
        add ``history: jax.Array`` after ``system`` and the keyword-only
        ``advance_history: bool``, and return ``(force, torque, history)``.
        Laws with velocity-dependent terms add the keyword-only
        ``quasistatic: bool`` and omit those terms when it is true.

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
            ``(force, torque)``.

        """
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        """Compute the pair potential energy.

        Laws whose stored energy depends on pair memory add
        ``history: jax.Array`` after ``system``.

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
        raise NotImplementedError(
            f"{type(system.force_model).__name__} does not implement pair energy."
        )

    def pair_force(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluate :meth:`force` with the arguments this law declares.

        This is the entry point colliders use. ``history`` is passed through
        unchanged for laws without pair memory, and ``advance_history`` and
        ``quasistatic`` reach only laws that declare them.
        """
        options = type(self)._force_options
        kwargs: dict[str, bool] = {}
        if "advance_history" in options:
            kwargs["advance_history"] = advance_history
        if "quasistatic" in options:
            kwargs["quasistatic"] = quasistatic
        if "history" in options:
            return self.force(i, j, pos, state, system, history, **kwargs)
        if history.shape[-1] != 0:
            raise TypeError(
                f"{type(self).__name__} declares pair history but its force "
                "kernel does not take a history argument."
            )
        force, torque = self.force(i, j, pos, state, system, **kwargs)
        return force, torque, history

    def pair_energy(
        self,
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        """Evaluate :meth:`energy`, passing ``history`` only if it is declared."""
        if "history" in type(self)._energy_options:
            return self.energy(i, j, pos, state, system, history)
        return self.energy(i, j, pos, state, system)

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


from .cundall_strack import (
    CundallStrackContactData,
    CundallStrackForce,
    arithmetic_mean_friction,
    geometric_mean_friction,
    get_cundall_strack_contacts,
    maximum_friction,
    minimum_friction,
)
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
    "CundallStrackContactData",
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
    "arithmetic_mean_friction",
    "geometric_mean_friction",
    "get_cundall_strack_contacts",
    "maximum_friction",
    "minimum_friction",
]
