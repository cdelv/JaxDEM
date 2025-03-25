# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax

from typing import Tuple
from abc import ABC, abstractmethod
from functools import partial

from jaxdem.Factory import Factory
from jaxdem.State import State
from jaxdem.System import System

class Simulator(Factory, ABC):
    """
    Abstract base class defining the interface for simulation strategies.

    This class serves as a factory for different simulation approaches, 
    providing a standard interface for force computation and time-stepping 
    algorithms.

    Attributes
    ----------
    None (interface class)

    Methods
    -------
    compute_force(state: State, system: System) -> Tuple[State, System]
        Compute forces acting between particles in the simulation.
    
    step(state: State, system: System) -> Tuple[State, System]
        Advance the simulation state by one time step.
    """
    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def compute_force(state: 'State', system: 'System') -> Tuple['State', 'System']:
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

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Notes
        -----
        The method has to be compatible with jax.jit
        """
        ...

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Advance the simulation state by one time step.

        This method combines force computation and time integration, 
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

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Notes
        -----
        The method has to be compatible with jax.jit
        """
        ...

@Simulator.register("naive")
class NaiveSimulator(Simulator):
    """
    This simulator computes forces between all pairs of particles using a 
    straightforward nested vectorized computation. 

    Methods
    -------
    compute_force(state: State, system: System) -> Tuple[State, System]
        Compute forces between all particle pairs using a nested double for loop.
    
    step(state: State, system: System) -> Tuple[State, System]
        Perform a complete simulation step by computing forces and integrating the state.

    Notes
    -----
    The NaiveSimulator has O(N^2) computational complexity, making it 
    unsuitable for large numbers of particles. For small systems, 
    the overhead of the other methods makes this mehtod worth it.
    """
    @staticmethod
    @partial(jax.jit, inline=True)
    def compute_force(state: 'State', system: 'System') -> Tuple['State', 'System']:
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
        the overhead of the other methods makes this mehtod worth it.
        """
        state.accel = jax.vmap(
            lambda i: jax.vmap(
                lambda j: 
                    system.force_model.calculate_force(i, j, state, system)
            )(jax.lax.iota(dtype=int, size=state.N)).sum(axis=0)
        )(jax.lax.iota(dtype=int, size=state.N))/state.mass[:, None]
        return state, system

    @staticmethod
    @partial(jax.jit, inline=True)
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Perform a complete simulation step by computing forces and integrating the state.

        This method cperforms:
        1. Computing forces between all particles
        2. Integrating the state using the system's specified time integrator

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            Updated state after the step.
        """
        state, system = NaiveSimulator.compute_force(state, system)
        state, system = system.integrator.step(state, system)
        return state, system