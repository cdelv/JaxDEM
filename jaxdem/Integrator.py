# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax

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
class Integrator(Factory, ABC):
    """
    This class serves as a factory and provides a standard interface for time-stepping. 
    Subclasses must implement the `step` method to define how particle states are updated over time.

    Attributes
    ----------
    None (interface class)

    Methods
    -------
    step(cls, state: State, system: System) -> Tuple[State, System]
        Abstract method to be implemented by subclasses, defining how to 
        advance the simulation state by one time step.
    """

    @classmethod
    @abstractmethod
    @partial(jax.jit, static_argnames=("cls"))
    def step(cls, state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Advance the simulation state by one time step using a specific 
        numerical integration method.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated State and System after one 
            time step of integration.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Notes
        -----
        This method must be compatible with jax.jit.
        """
        ...

@jax.tree_util.register_dataclass
@dataclass
@Integrator.register("euler")
class DirectEuler(Integrator):
    """
    Direct Euler integration method.

    Methods
    -------
    step(cls, state: State, system: System) -> Tuple[State, System]
        Perform a single time step using the Direct Euler method.
    """
    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def step(cls, state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Direct Euler integration method.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            The updated state and system after one time step.

        Notes
        -----
        Direct Euler:
        - Increments velocity: v(t+dt) = v(t) + a(t) * dt
        - Updates position: x(t+dt) = x(t) + v(t+dt) * dt
        """
        state.vel += system.dt * state.accel
        state.pos += system.dt * state.vel
        return state, system