# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax
import jax.numpy as jnp

from typing import ClassVar, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Domain(Factory["Domain"], ABC):
    """
    Abstract base class representing the simulation domain.

    Domains define the spatial boundaries and coordinate transformation rules.

    The Domain class defines how:
        - Particle displacements are calculated
        - Boundary conditions are handled

    Attributes
    ----------
    box_size : jax.Array
        Size of the computational domain in each dimension.
    anchor : jax.Array
        Reference point (origin) of the computational domain.

    Methods
    -------
    displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array
        Calculate the displacement vector between two particles.
    
    shift(r: jax.Array, system: System) -> Tuple["State", "System"]
        Adjusts particle positions based on domain-specific boundary conditions.

    Notes
    -----
    - Must Support 2D and 3D domains.
    - Must be jit compatible
    """
    box_size: jax.Array
    anchor: jax.Array
    periodic: ClassVar[bool] = False
       
    @staticmethod
    @abstractmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
        """
        Calculate the displacement vector between two particles and defines the interface of the method.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle.
        rj : jax.Array
            Position vector of the second particle.
        system : System

        Returns
        -------
        jax.Array
            Displacement vector between particles, transformed by domain rules.
        """
        raise NotImplemented

    @staticmethod
    @abstractmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Adjusts particle positions based on domain-specific boundary conditions.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        State
            The updated State after adjusting the particle positions 
            based on domain-specific boundary conditions.
        System
        """
        raise NotImplemented

@Domain.register("free")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FreeDomain(Domain):
    """
    A domain with unbounded, Euclidean space.

    Attributes
    ----------
    Inherits from Domain base class.

    Methods
    -------
    displacement(ri, rj, system)
        Calculates simple Euclidean displacement between particles.
    
    shift(state, system)
        Does not apply
    """
    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        """
        Calculate direct Euclidean displacement between two vectors.

        Parameters
        ----------
        ri : jax.Array
        rj : jax.Array
        system : System

        Returns
        -------
        jax.Array
            Vector difference between particle positions.
        """
        return ri - rj

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Does not apply

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        State
            Unchanged state.
        System
        """
        return state, system

@Domain.register("reflect")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ReflectDomain(Domain):
    """
    A domain with reflective boundaries in each dimension.

    Attributes
    ----------
    Inherits from Domain base class.

    Methods
    -------
    displacement(ri, rj, system)
        Calculates simple Euclidean displacement between particles.
    
    shift(state, system)
        Applies reflective boundary conditions (bounce-back)
    """
    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        """
        Calculate Euclidean displacement between two vectors.

        Parameters
        ----------
        ri : jax.Array
        rj : jax.Array
        system : System

        Returns
        -------
        jax.Array
            Vector difference between particle positions.
        """
        return ri - rj

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Applies reflective boundary conditions (bounce-back).

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        State
            The updated State after adjusting the particle positions 
            based on reflective boundary conditions.
        System
        """
        lower_bound = system.domain.anchor + state.rad[:, None]
        upper_bound = system.domain.anchor + system.domain.box_size - state.rad[:, None]
        outside_lower = state.pos < lower_bound
        outside_upper = state.pos > upper_bound
        state.vel = jnp.where(outside_lower + outside_upper, -state.vel, state.vel)
        reflected_pos = jnp.where(outside_lower, 2.0 * lower_bound - state.pos, state.pos)
        reflected_pos = jnp.where(outside_upper, 2.0 * upper_bound - reflected_pos, reflected_pos)
        state.pos = reflected_pos
        return state, system

@Domain.register("periodic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PeriodicDomain(Domain):
    """
    A domain with periodic boundary conditions.

    Attributes
    ----------
    Inherits from Domain base class.

    Methods
    -------
    displacement(ri, rj, system)
        Calculates displacement considering periodic boundary wrapping.
    
    shift(state, system)
        Apply the periodic boundary conditions.
    """
    periodic: ClassVar[bool] = True

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
        """
        Calculate displacement with periodic boundary conditions.

        Parameters
        ----------
        ri : jax.Array
        rj : jax.Array
        system : System

        Returns
        -------
        jax.Array
            Displacement vector.
        """
        rij = (ri - system.domain.anchor) - (rj - system.domain.anchor)
        return rij - system.domain.box_size * jnp.round(rij / system.domain.box_size)

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Computes the position shifts to apply the periodic boundary conditions.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        State
           The updated State after adjusting the particle positions 
            based on periodic boundary conditions.
        System
        """
        state.pos -= system.domain.box_size * jnp.floor((state.pos - system.domain.anchor) / system.domain.box_size)
        return state, system