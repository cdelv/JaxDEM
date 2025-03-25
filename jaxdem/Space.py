# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from typing import Tuple
from abc import ABC, abstractmethod
from functools import partial

from jaxdem.Factory import Factory
from jaxdem.System import System

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
        Compute the spatial shift for particles within the domain.

    Notes
    -----
    - Supports both 2D and 3D simulation domains
    - Enables different boundary condition implementations
    - Can be extended to create custom spatial metrics
    """

    def __init__(self, box_size: jnp.ndarray = jnp.ones(3), anchor: jnp.ndarray = jnp.zeros(3)):
        """
        Parameters
        ----------
        box_size : jnp.ndarray, optional
            Size of the computational domain in each dimension.
        anchor : jnp.ndarray, optional
            Reference point (origin) of the computational domain.
        """
        self.box_size = box_size
        self.anchor = anchor

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Calculate the displacement vector between two particles and defines the interface the subclasses need to follow.

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
    def shift(r:jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Compute spatial shift for particles within the domain and defines the interface the subclasses need to follow.

        Adjusts particle positions based on domain-specific boundary conditions

        Parameters
        ----------
        r : jnp.ndarray
            Position vector of a particle.
        system : System

        Returns
        -------
        jnp.ndarray
            Shift vector to be applied to the particle's position.
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
    
    shift(r, system)
        Returns zero.
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
    def shift(r:jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Returns zero

        Parameters
        ----------
        r : jnp.ndarray
        system : System

        Returns
        -------
        jnp.ndarray
            Zero vector.
        """
        return jnp.zeros_like(r)

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
    
    shift(r, system)
        Computes position shifts to apply the periodic boundary conditions.
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
    def shift(r:jnp.ndarray, system: 'System') -> jnp.ndarray:
        """
        Computes position shifts to apply the periodic boundary conditions.

        Parameters
        ----------
        r : jnp.ndarray
        system : System

        Returns
        -------
        jnp.ndarray
            Shift vector to reposition particles within domain.
        """
        return system.domain.box_size * jnp.floor((r - system.domain.anchor) / system.domain.box_size)