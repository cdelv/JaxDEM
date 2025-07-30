# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceModel(Factory["ForceModel"], ABC):
    """
    This class provides the protocol for computing forces and energies between pairs of particles.

    Attributes
    ----------
    None (interface class)

    Methods
    -------
    calculate_force(i: int, j: int, state: State, system: System) -> jax.Array
        Abstract method to compute the force between two specific particles.
    
    calculate_energy(i: int, j: int, state: State, system: System) -> jax.Array
        Abstract method to compute the potential energy of interaction between two particles.

    Notes
    -----
    - Implementations should be JIT-compilable and work with JAX's transformation functions.
    - The force and energy methods should correctly handle the case i = j. There is no guaranty that i = j will be or not be called.
    """
    @staticmethod
    @abstractmethod
    @jax.jit
    def calculate_force(i: int, j: int, state: 'State', system: 'System') -> jax.Array:
        """
        Compute the force between two specific particles in the simulation.

        This method calculates the force acting on particle i due to particle j, 
        considering their current state, and the system's properties.

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
        jnp.ndarray
            Force vector acting on particle i due to particle j.
            The vector's dimension matches the simulation's dimensionality (2D or 3D).

        Notes
        -----
        - The method must be jax.jit compatible.
        """
        raise NotImplemented

    @staticmethod
    @abstractmethod
    @jax.jit
    def calculate_energy(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """
        Compute the potential energy of the interaction between particles i and j.

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
            Potential energy of the interaction between particles i and j.

        Notes
        -----
        - The method must be jax.jit compatible.
        """
        raise NotImplemented

@jax.tree_util.register_dataclass
@dataclass(slots=True)
@ForceModel.register("spring")
class SpringForce(ForceModel):
    """
    Linear spring-like interaction between particles

    Attributes
    ----------
    None

    Methods
    -------
    calculate_force(i: int, j: int, state: State, system: System) -> jax.Array
        Computes a spring-like force based on particle overlap.
    
    calculate_energy(i: int, j: int, state: State, system: System) -> jax.Array
        Calculates the potential energy of particle interaction.
    """
    @staticmethod
    @jax.jit
    def calculate_force(i: int, j: int, state: 'State', system: 'System') -> jax.Array:
        """
        Compute linear spring-like interaction force acting on particle i due to particle j.
        Returns zero when i = j.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        state : State
        system : System

        Returns
        -------
        jax.Array
            Force vector acting on particle i due to particle j.
        """
        #k_i = system.material.youngs_modulus[i]
        #k_j = system.material.youngs_modulus[j]
        k = 100.0 #system.material_matchmaker.get_effective_property(k_i, k_j)

        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r = jnp.linalg.norm(rij)
        s = jnp.maximum(0.0, (state.rad[i] + state.rad[j])/(r + jnp.finfo(state.pos.dtype).eps) - 1.0)
        return k * s * rij

    @staticmethod
    @jax.jit
    def calculate_energy(i: int, j: int, state: 'State', system: 'System') -> jax.Array:
        """
        Compute linear spring-like interaction energy acting on particle i due to particle j.
        Returns zero when i = j.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        state : State
        system : System

        Returns
        -------
        jax.Array
            Potential energy of particle i due to particle j.
        """
        #k_i = system.material.youngs_modulus[i]
        #k_j = system.material.youngs_modulus[j]
        k = 100.0 #system.material_matchmaker.get_effective_property(k_i, k_j)

        r_ij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r = jnp.linalg.norm(r_ij)
        s = jnp.maximum(0.0, (state.rad[i] + state.rad[j])/(r + jnp.finfo(state.pos.dtype).eps) - 1.0)
        return 0.5 * k * s * s