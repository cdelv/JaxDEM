# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

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
@dataclass(slots=True)
class ForceModel(Factory["ForceModel"], ABC):
    """
    This class provides the protocol for computing forces and energies between pairs of particles.

    Attributes
    ----------
    None (interface class)

    Methods
    -------
    force(i: int, j: int, state: State, system: System) -> jax.Array
        Abstract method to compute the force between two specific particles.
    
    energy(i: int, j: int, state: State, system: System) -> jax.Array
        Abstract method to compute the potential energy of interaction between two particles.

    Notes
    -----
    - Implementations should be JIT-compilable and work with JAX's transformation functions.
    - The force and energy methods should correctly handle the case i = j. There is no guarantee that i == j will not be called..
    """
    required_material_properties: Tuple[str, ...] = field(default=(), metadata={"static": True})
    laws: Tuple["ForceModel", ...] = field(default=(), metadata={"static": True})

    @staticmethod
    @abstractmethod
    @jax.jit
    def force(i: int, j: int, state: 'State', system: 'System') -> jax.Array:
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def energy(i: int, j: int, state: "State", system: "System") -> jax.Array:
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
        raise NotImplementedError

@ForceModel.register("spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SpringForce(ForceModel):
    """
    Linear spring-like interaction between particles

    Attributes
    ----------
    None

    Methods
    -------
    force(i: int, j: int, state: State, system: System) -> jax.Array
        Computes a spring-like force based on particle overlap.
    
    energy(i: int, j: int, state: State, system: System) -> jax.Array
        Calculates the potential energy of particle interaction.
    """
    required_material_properties: Tuple[str, ...] = field(default=("young_eff",), metadata={"static": True})

    @staticmethod
    @jax.jit
    def force(i: int, j: int, state: 'State', system: 'System') -> jax.Array:
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
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]

        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r2 = jnp.dot(rij, rij)
        r  = jnp.sqrt(r2 + jnp.finfo(state.pos.dtype).eps)
        s = jnp.maximum(0.0, (state.rad[i] + state.rad[j])/r - 1.0)
        return k * s * rij

    @staticmethod
    @jax.jit
    def energy(i: int, j: int, state: 'State', system: 'System') -> jax.Array:
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
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]

        rij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r2 = jnp.dot(rij, rij)
        r  = jnp.sqrt(r2 + jnp.finfo(state.pos.dtype).eps)
        s = jnp.maximum(0.0, (state.rad[i] + state.rad[j])/r - 1.0)
        return 0.5 * k * s**2