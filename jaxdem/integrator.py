# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from .factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state import State
    from .system import System
    
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Integrator(Factory["Integrator"], ABC):
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

    Notes
    -----
    - Must Support 2D and 3D domains.
    - Must be jit compatible
    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def step(state: "State", system: "System") -> Tuple["State", "System"]:
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

    @staticmethod
    @abstractmethod
    @jax.jit
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Some integration methods require an initialization step, like Leap Frog. 
        This function implements the interface for initialization.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated State and System 

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
    Direct-Euler integration method.
    """
    @staticmethod
    @jax.jit
    def step(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Direct-Euler integration method.

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
        state, system = system.domain.shift(state, system)
        state, system = system.collider.compute_force(state, system)
        state.vel += system.dt * state.accel * (1 - state.fixed)[..., None]
        state.pos += system.dt * state.vel * (1 - state.fixed)[..., None]
        return state, system

    @staticmethod
    @jax.jit
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Direct-Euler does not need initialization method.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            The updated state and system
        """
        return state, system