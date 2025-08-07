# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining colliders. Colliders perform contact detection and compute forces.
"""

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Tuple, TYPE_CHECKING

from .factory import Factory

if TYPE_CHECKING:
    from .state import State
    from .system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Collider(Factory["Collider"], ABC):
    r"""
    The base interface for defining how contact detection and force computations are performed in a simulation.

    Concrete subclasses of `Collider` implement the specific algorithms for calculating the interactions.

    Notes
    -----
    All abstract methods in `Collider` (and their implementations in subclasses)
    must be compatible with JAX transformations (`jax.jit`, `jax.vmap`, etc.).
    They are expected to work seamlessly in both 2D and 3D simulations.

    Self-interaction (i.e., calling the force/energy computation for `i=j`) is allowed,
    and the underlying `force_model` is responsible for correctly handling or
    ignoring this case.

    Example
    -------
    To define a custom collider, inherit from `Collider`, register it and implement its abstract methods:

    >>> @Collider.register("CustomCollider")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class CustomCollider(Collider):
            ...

    Then, instantiate it:

    >>> jaxdem.Collider.create("CustomCollider", **custom_collider_kw)
    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Abstract method to compute the total force acting on each particle in the simulation.

        Implementations should calculate inter-particle forces based on the current
        `state` and `system` configuration, then update the `accel` attribute of the
        `state` object with the resulting total acceleration for each particle.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated `State` object (with computed accelerations)
            and the `System` object.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------
        This method is typically called internally by the `System`'s step function:

        >>> state, system = system.collider.compute_force(state, system)
        """
        raise NotImplemented

    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        """
        Abstract method to compute the total potential energy of the system.

        Implementations should calculate the sum of all potential energies
        present in the system based on the current `state` and `system` configuration.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            A scalar JAX array representing the total potential energy of each particle.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------

        >>> total_potential_energy = system.collider.compute_potential_energy(state, system)
        >>> print(f"Total potential energy per particle: {total_potential_energy:.4f}")
        """
        raise NotImplemented


@Collider.register("naive")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NaiveSimulator(Collider):
    r"""
    Implementation that computes forces and potential energies using a naive :math:`O(N^2)` all-pairs interaction loop.

    Notes
    -----
    Due to its :math:`O(N^2)` complexity, `NaiveSimulator` is suitable for simulations
    with a relatively small number of particles. For larger systems, a more
    efficient spatial partitioning collider should be used. However, thhis collider should be the fastest
    option for small systems (:math:`<10^3` spheres)

    Example
    -------
    """

    @staticmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        r"""
        Computes the total force on each particle using a naive :math:`O(N^2)` all-pairs loop.

        This method iterates over all particle pairs (i, j) and sums the forces
        computed by the `system.force_model`.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated `State` object with computed accelerations
            and the `System` object.

        """
        Range = jax.lax.iota(dtype=int, size=state.N)
        state = replace(
            state,
            accel=state.accel
            + (
                jax.vmap(
                    lambda i: jax.vmap(
                        lambda j: system.force_model.force(i, j, state, system)
                    )(Range).sum(axis=0)
                )(Range)
                / state.mass[:, None]
            ),
        )
        return state, system

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        r"""
        Computes the total potential energy of the system using a naive :math:`O(N^2)` all-pairs loop.

        This method sums the potential energy contributions from all particle pairs (i, j)
        as computed by the `system.force_model`.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            A scalar JAX array representing the total potential energy of the system.
        """
        Range = jax.lax.iota(dtype=int, size=state.N)
        return jax.vmap(
            lambda i: jax.vmap(
                lambda j: system.force_model.energy(i, j, state, system)
            )(Range).sum(axis=0)
        )(Range)
