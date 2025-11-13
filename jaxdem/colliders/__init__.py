# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Collision-detection interfaces and implementations."""

from __future__ import annotations

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Collider(Factory, ABC):
    r"""
    The base interface for defining how contact detection and force computations are performed in a simulation.

    Concrete subclasses of `Collider` implement the specific algorithms for calculating the interactions.

    Notes
    -----
    Self-interaction (i.e., calling the force/energy computation for `i=j`) is allowed,
    and the underlying `force_model` is responsible for correctly handling or
    ignoring this case.

    Example
    -------
    To define a custom collider, inherit from `Collider`, register it and implement its abstract methods:

    >>> @Collider.register("CustomCollider")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True, frozen=True)
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

        Implementations should calculate inter-particle forces and torques based on the current
        `state` and `system` configuration, then update the `accel` and `angAccel` attributes of the
        `state` object with the resulting total accelerations for each particle.

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
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        """
        Abstract method to compute the total potential energy of the system.

        Implementations should calculate the sum per particle of all potential energies
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

        Example
        -------

        >>> potential_energy = system.collider.compute_potential_energy(state, system)
        >>> print(f"Potential energy per particle: {potential_energy:.4f}")
        >>> print(potential_energy.shape") # (N, 1)
        """
        raise NotImplementedError


from .naive import NaiveSimulator
from .cell_list import CellList

__all__ = ["Collider", "NaiveSimulator", "CellList"]
