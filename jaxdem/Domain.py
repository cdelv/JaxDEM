# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from typing import Optional
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass, field

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class Domain(Factory, ABC):
    """
    Abstract base class representing the simulation domain.

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
    dim: int = field(default=3, metadata={"static": True})
    periodic: bool = field(default=False, metadata={"static": True})
    box_size: Optional[jax.Array] = None
    anchor:Optional[jax.Array] = None

    def __post_init__(self):
        if self.box_size is None:
            self.box_size = jnp.ones(self.dim)
        
        if self.anchor is None:
            self.anchor = jnp.zeros(self.dim)
       
    @classmethod
    @abstractmethod
    @partial(jax.jit, static_argnames=("cls"))
    def displacement(cls, ri: jax.Array, rj: jax.Array, system: "System") -> jnp.ndarray:
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
        raise NotImplemented

    @classmethod
    @abstractmethod
    @partial(jax.jit, static_argnames=("cls"))
    def shift(cls, state: "State", system: "System") -> "State":
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
        """
        raise NotImplemented

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
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
    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def displacement(cls, ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
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

    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def shift(cls, state: "State", system: "System") -> "State":
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
        """
        return state


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
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
    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def displacement(cls, ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
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

    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def shift(cls, state: "State", system: "System") -> "State":
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
        """
        lower_bound = system.domain.anchor + state.rad[:, None]
        upper_bound = system.domain.anchor + system.domain.box_size - state.rad[:, None]
        outside_lower = state.pos < lower_bound
        outside_upper = state.pos > upper_bound
        state.vel = jnp.where(outside_lower + outside_upper, -state.vel, state.vel)
        reflected_pos = jnp.where(outside_lower, 2.0 * lower_bound - state.pos, state.pos)
        reflected_pos = jnp.where(outside_upper, 2.0 * upper_bound - reflected_pos, reflected_pos)
        state.pos = reflected_pos
        return state

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
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
    periodic: bool = field(default=True, metadata={"static": True})

    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def displacement(cls, ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
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

    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def shift(cls, state: "State", system: "System") -> "State":
        """
        Computes position shifts to apply the periodic boundary conditions.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        State
           The updated State after adjusting the particle positions 
            based on periodic boundary conditions.
        """
        state.pos -= system.domain.box_size * jnp.floor((state.pos - system.domain.anchor) / system.domain.box_size)
        return state