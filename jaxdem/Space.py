# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from functools import partial

from jaxdem.Factory import Factory
from jaxdem.System import System
from jaxdem.State import State

class Domain(Factory, ABC):
    """
    Abstract base class representing the domains of the simulation.

    Domains define the spatial boundaries and coordinate transformation rules 
    for particle interactions.

    The Domain class defines how:
    - Particle displacements are calculated
    - Boundary conditions are handled

    Attributes
    ----------
    box_size : jnp.ndarray
        Size of the computational domain in each dimension.
    anchor : jnp.ndarray
        Reference point (origin) of the computational domain.

    Methods
    -------
    displacement(ri: jnp.ndarray, rj: jnp.ndarray, system: System) -> jnp.ndarray
        Calculate the displacement vector between two particles.
    
    shift(r: jnp.ndarray, system: System) -> jnp.ndarray
        Adjusts particle positions based on domain-specific boundary conditions.

    Notes
    -----
    - Supports 2D and 3D domains.
    - Enables different boundary condition implementations
    - Can be extended to create custom spatial metrics
    """

    def __init__(self, dim: int = 3, box_size: Optional[jnp.ndarray] = None, anchor: Optional[jnp.ndarray] = None):
        """
        Parameters
        ----------
        box_size : jnp.ndarray, optional
            Size of the computational domain in each dimension.
        anchor : jnp.ndarray, optional
            Reference point (origin) of the computational domain.
        """
        self.box_size = box_size
        if self.box_size is None:
            self.box_size = jnp.ones(dim)

        self.anchor = anchor
        if self.anchor is None:
            self.anchor = jnp.zeros(dim)

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Calculate the displacement vector between two particles and defines the interface of the method.

        Parameters
        ----------
        ri : jnp.ndarray
            Position vector of the first particle.
        rj : jnp.ndarray
            Position vector of the second particle.
        system : System

        Returns
        -------
        jnp.ndarray
            Displacement vector between particles, transformed by domain rules.
        """
        ...

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def shift(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Adjusts particle positions based on domain-specific boundary conditions.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated State and System after adjusting the particle positions 
            based on domain-specific boundary conditions.
        """
        ...

@Domain.register("free")
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
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Calculate direct Euclidean displacement between two vectors.

        Parameters
        ----------
        ri : jnp.ndarray
        rj : jnp.ndarray
        system : System

        Returns
        -------
        jnp.ndarray
            Vector difference between particle positions.
        """
        return ri - rj

    @staticmethod
    @partial(jax.jit, inline=True)
    def shift(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Does not apply

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            Unchanged state.
        """
        return state, system


@Domain.register("reflect")
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
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Calculate Euclidean displacement between two vectors.

        Parameters
        ----------
        ri : jnp.ndarray
        rj : jnp.ndarray
        system : System

        Returns
        -------
        jnp.ndarray
            Vector difference between particle positions.
        """
        return ri - rj

    @staticmethod
    @partial(jax.jit, inline=True)
    def shift(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Applies reflective boundary conditions (bounce-back).

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            Unchanged state.
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

    @staticmethod
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Calculate displacement with periodic boundary conditions.

        Parameters
        ----------
        ri : jnp.ndarray
        rj : jnp.ndarray
        system : System

        Returns
        -------
        jnp.ndarray
            Displacement vector.
        """
        rij = (ri - system.domain.anchor) - (rj - system.domain.anchor)
        return rij - system.domain.box_size * jnp.round(rij / system.domain.box_size)

    @staticmethod
    @partial(jax.jit, inline=True)
    def shift(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Computes position shifts to apply the periodic boundary conditions.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            State with applied periodic boundary conditions.
        """
        state.pos -= system.domain.box_size * jnp.floor((state.pos - system.domain.anchor) / system.domain.box_size)
        return state, system