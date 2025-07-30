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
@dataclass(slots=True)
class Integrator(Factory, ABC):
    """
    This class serves as a factory and provides a standard interface for time-stepping. 
    Subclasses must implement the `step` method to define how particle states are updated over time.

    Attributes
    ----------
    None (interface class)

    Methods
    -------
    step(state: State, system: System) -> Tuple[State, System]
        Abstract method to be implemented by subclasses, defining how to advance the simulation state by one time step.
    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
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
        raise NotImplementedError

@Integrator.register("euler")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DirectEuler(Integrator):
    """
    Direct Euler integration method.
    """
    @staticmethod
    @jax.jit
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
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
        #state, system = system.Collider.compute_forces()
        state.vel += system.dt * state.accel
        state.pos += system.dt * state.vel
        #state, system = system.Domain.shift()
        return state, system