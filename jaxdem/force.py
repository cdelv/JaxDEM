# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining force laws and their corresponding potential energy.
"""

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

from .factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ForceModel(Factory, ABC):
    """
    Abstract base class for defining inter-particle force laws and their potential energies.

    Concrete subclasses implement specific force and energy models, such as
    linear springs, Hertzian contacts, etc.

    Notes
    -----
    - Implementations should be JIT-compilable.
    - The :meth:`force` and :meth:`energy` methods should correctly handle the
      case where `i` and `j` refer to the same particle (i.e., `i == j`).
      There is no guarantee that self-interaction calls will not occur.

    Example
    -------
    To define a custom force model, inherit from :class:`ForceModel` and implement
    its abstract methods:

    >>> @ForceModel.register("myCustomForce")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomForce(ForceModel):
            ...
    """

    required_material_properties: Tuple[str, ...] = field(
        default=(), metadata={"static": True}
    )
    """
    A static tuple of strings specifying the material properties required by this force model.

    These properties (e.g., 'young_eff', 'restitution') must be present in the
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

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------
        This method is called internally by a `Collider` when computing
        total forces:

        >>> force_on_particle_0_from_1 = system.force_model.force(0, 1, state, system)
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

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------
        This method is typically called internally by a `Collider` when computing
        the total potential energy of the system:

        >>> energy_0_1 = system.force_model.energy(0, 1, state, system)
        """
        raise NotImplementedError


@ForceModel.register("spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class SpringForce(ForceModel):
    r"""
    A `ForceModel` implementation for a linear spring-like interaction between particles.

    Notes
    -----
    - The 'effective Young's modulus' (:math:`k_{eff,\; ij}`) is retrieved from the
      :attr:`system.System.mat_table.young_eff` based on the material IDs of the interacting particles.
    - The force is zero if :math:`i == j`.
    - A small epsilon is added to the squared distance (:math:`r^2`) before taking the square root
      to prevent division by zero or NaN issues when particles are perfectly co-located.


    The penetration :math:`\delta` (overlap) between two particles :math:`i` and :math:`j` is:

    .. math::
        \delta = (R_i + R_j) - r

    where :math:`R_i` and :math:`R_j` are the radii of particles :math:`i` and :math:`j` respectively,
    and :math:`r = ||r_{ij}||` is the distance between their centers.

    The scalar overlap :math:`s` is defined as:

    .. math::
        s = \max \left(0, \frac{R_i + R_j}{r} - 1 \right)

    The force :math:`F_{ij}` acting on particle :math:`i` due to particle :math:`j` is:

    .. math::
        F_{ij} = k_{eff,\; ij} s r_{ij}

    The potential energy :math:`E_{ij}` of the interaction is:

    .. math::
        E_{ij} = \frac{1}{2} k_{eff,\; ij} s^2

    where :math:`k_{eff,\; ij}` is the effective Young's modulus for the particle pair.

    Example
    -------
    To use :class:`SpringForce` in a simulation, specify it as the `force_model`
    when creating your :class:`System`:

    >>> system = jaxdem.System.create(dim=3, force_model_type="spring", force_model_kw={})

    For this force model, the typical :class:`MaterialMatchmaker`: type is "harmonic".
    """

    required_material_properties: Tuple[str, ...] = field(
        default=("young_eff",), metadata={"static": True}
    )

    @staticmethod
    @jax.jit
    def force(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """
        Compute linear spring-like interaction force acting on particle :math:`i` due to particle :math:`j`.

        Returns zero when :math:`i = j`.

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
            Force vector acting on particle :math:`i` due to particle :math:`j`.
        """
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]

        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r2 = jnp.dot(rij, rij)
        r = jnp.sqrt(
            r2 + jnp.finfo(state.pos.dtype).eps
        )  # Adding epsilon for numerical stability
        R = state.rad[i] + state.rad[j]
        s = jnp.maximum(0.0, R / r - 1.0)
        return k * s * rij

    @staticmethod
    @jax.jit
    def energy(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """
        Compute linear spring-like interaction potential energy between particle :math:`i` and particle :math:`j`.

        Returns zero when :math:`i = j`.

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
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]

        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r2 = jnp.dot(rij, rij)
        r = jnp.sqrt(r2)
        R = state.rad[i] + state.rad[j]
        s = jnp.maximum(0.0, R - r)
        return 0.5 * k * s**2
