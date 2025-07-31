# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Collider(Factory["Collider"], ABC):
    """
    This class serves as a factory for the force calculation, 

    Attributes
    ----------
    None (interface class)

    Methods
    -------
    compute_force(state: State, system: System) -> Tuple[State, System]
    compute_potential_energy(state: State, system: System) -> jax.Array
    """
    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Compute the forces acting between particles in the simulation.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            A tuple containing the State with computed accelerations

        Notes
        -----
        The method has to be compatible with jax.jit
        """
        raise NotImplemented

    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        """
        Compute the total potential energy of one particles due to the interaction with other particles in the simulation.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        jax.Array
            Array with potential energies per particle.

        Notes
        -----
        The method has to be compatible with jax.jit
        """
        raise NotImplemented

@Collider.register("naive")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NaiveSimulator(Collider):
    """
    This simulator computes forces between all pairs of particles using a naive double nested loop. 

    Methods
    -------
    compute_force(state: State, system: System) -> Tuple[State, System]
        Compute forces between all particle pairs using a nested double for loop.

    Notes
    -----
    The NaiveSimulator has O(N^2) computational complexity, making it 
    unsuitable for large numbers of particles. For small systems, 
    the overhead of the other methods makes this method worth it.
    """
    @staticmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Compute forces between all particle pairs using a nested double for loop.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            Updated state with computed particle accelerations.

        Notes
        -----
        This implementation has O(N^2) computational complexity, making it 
        unsuitable for large numbers of particles. For small systems, 
        the overhead of the other methods makes this method worth it.
        """
        Range = jax.lax.iota(dtype=int, size=state.N)
        state.accel = jax.vmap(
            lambda i: jax.vmap(
                lambda j: system.force_model.calculate_force(i, j, state, system)
            )(Range).sum(axis=0)
        )(Range)/state.mass[:, None]
        return state, system

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        """
        Compute total potential energy per particle using a nested double for loop.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        jax.Array
            Array with potential energies per particle.

        Notes
        -----
        This implementation has O(N^2) computational complexity, making it 
        unsuitable for large numbers of particles. For small systems, 
        the overhead of the other methods makes this method worth it.
        """
        Range = jax.lax.iota(dtype=int, size=state.N)
        return jax.vmap(
            lambda i: jax.vmap(
                lambda j: system.force_model.calculate_energy(i, j, state, system)
            )(Range).sum(axis=0)
        )(Range)