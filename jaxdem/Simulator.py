# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Tuple

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

@jax.tree_util.register_dataclass
@dataclass
class Simulator(Factory, ABC):
    """
    This class serves as a factory for different simulators, 
    providing a standard interface for force computation and time-stepping.

    Attributes
    ----------
    None (interface class)

    Methods
    -------
    compute_force(cls, state: State, system: System) -> Tuple[State, System]
        Compute forces acting between particles in the simulation.
    
    step(cls, state: State, system: System) -> Tuple[State, System]
        Advance the simulation state by one time step.
    """
    @classmethod
    @abstractmethod
    @partial(jax.jit, static_argnames=("cls"))
    def compute_force(cls, state: 'State', system: 'System') -> Tuple['State', 'System']:
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
        ...

    @classmethod
    @abstractmethod
    @partial(jax.jit, static_argnames=("cls"))
    def step(cls, state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Advance the simulation state by one time step.

        This method combines the force computation and time integration, 
        representing a single time step in the simulation. Any additional
        operations required in a time step are also done here.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated State and System.

        Notes
        -----
        The method has to be compatible with jax.jit
        """
        ...


@Simulator.register("naive")
@jax.tree_util.register_dataclass
@dataclass
class NaiveSimulator(Simulator):
    """
    This simulator computes forces between all pairs of particles using a naive
    double nested loop. 

    Methods
    -------
    compute_force(cls, state: State, system: System) -> Tuple[State, System]
        Compute forces between all particle pairs using a nested double for loop.
    
    step(cls, state: State, system: System) -> Tuple[State, System]
        Perform a complete simulation step by computing forces and integrating the state.

    Notes
    -----
    The NaiveSimulator has O(N^2) computational complexity, making it 
    unsuitable for large numbers of particles. For small systems, 
    the overhead of the other methods makes this method worth it.
    """
    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def compute_force(cls, state: 'State', system: 'System') -> Tuple['State', 'System']:
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
                lambda j: 
                    system.force_model.calculate_force(i, j, state, system)
            )(Range).sum(axis=0)
        )(Range)/state.mass[:, None]
        return state, system

    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def step(cls, state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Perform a complete simulation step by computing forces and integrating the state.

        This method performs:
        1. Forces calculation between all particles
        2. State integration using the system's specified time integrator

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            Updated state after the step.
        """
        #state, system = cls.compute_force(state, system)
        state, system = system.integrator.step(state, system)
        #state = system.domain.shift(state, system)
        return state, system