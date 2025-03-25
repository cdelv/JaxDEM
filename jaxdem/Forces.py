# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from functools import partial

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Factory import Factory
    from .State import State
    from .System import System

class ForceModel(Factory, ABC):
    """
    Abstract base class defining the interface for calculating inter-particle forces.

    This class provides the protocol for computing forces and energies between particles,
    Force models can be registered and instantiated dynamically using the Factory pattern.

    Attributes
    ----------
    None

    Methods
    -------
    calculate_force(i: int, j: int, state: State, system: System) -> jnp.ndarray
        Abstract method to compute the force between two specific particles.
    
    calculate_energy(i: int, j: int, state: State, system: System) -> float
        Abstract method to compute the potential energy of interaction between two particles.

    Notes
    -----
    - Implementations should be JIT-compilable and work with JAX's transformation functions.
    - The force and energy methods should correcly handle the case i = j and return vec{0} for forces
    unless self interaction is dessired. There is no guaranty i = j will be called.
    """

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def calculate_force(i: int, j: int, state: 'State', system: 'System') -> jnp.ndarray:
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
        ...

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def calculate_energy(i: int, j: int, state: 'State', system: 'System') -> float:
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
        float
            Potential energy of the interaction between particles i and j.

        Notes
        -----
        - The method must be jax.jit compatible.
        """
        ...

@ForceModel.register("spring")
class SpringForce(ForceModel):
    """
    Linear spring-like interaction between particles

    Attributes
    ----------
    None

    Methods
    -------
    calculate_force(i: int, j: int, state: State, system: System) -> jnp.ndarray
        Computes a spring-like force based on particle overlap.
    
    calculate_energy(i: int, j: int, state: State, system: System) -> float
        Calculates the potential energy of particle interaction.
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    def calculate_force(i: int, j: int, state: 'State', system: 'System') -> jnp.ndarray:
        """
        Compute linear spring-like interaction
         acting on particle i due to particle j.
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
        jnp.ndarray
            Force vector acting on particle i due to particle j.
        """
        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r = jnp.linalg.norm(rij)
        s = jnp.maximum(0.0, (state.rad[i] + state.rad[j])/(r + jnp.finfo(state.pos.dtype).eps) - 1.0)
        return system.k * s * rij # change to system.material

    @staticmethod
    @partial(jax.jit, inline=True)
    def calculate_energy(i: int, j: int, state: 'State', system: 'System') -> float:
        """
        Calculate potential energy for particle interaction using spring-like model.

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
        float
            Potential energy of particle interaction.
        """
        r_ij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r = jnp.linalg.norm(r_ij)
        s = jnp.maximum(0.0, (state.rad[i] + state.rad[j])/(r + jnp.finfo(state.pos.dtype).eps) - 1.0)
        return 0.5 * system.k * s * s  # Quadratic energy based on overlap