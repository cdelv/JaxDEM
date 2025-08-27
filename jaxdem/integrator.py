# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining time integrators. The time integrator performs one simulation step.
"""

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Tuple

from .factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Integrator(Factory, ABC):
    """
    Abstract base class for defining the interface for time-stepping.

    Notes
    -----
    - Must Support 2D and 3D domains.
    - Must be jit compatible

    Example
    -------
    To define a custom integrator, inherit from :class:`Integrator` and implement its abstract methods:

    >>> @Integrator.register("myCustomIntegrator")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomIntegrator(Integrator):
            ...
    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def step(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Advance the simulation state by one time step using a specific numerical integration method.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated State and System after one time step of integration.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------

        >>> state, system = system.integrator.step(state, system)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Some integration methods require an initialization step, for example LeapFrog.
        This function implements the interface for the initialization.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated State and System after the initialization.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------

        >>> state, system = system.integrator.initialize(state, system)
        """
        raise NotImplementedError


@Integrator.register("euler")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class DirectEuler(Integrator):
    """
    Implements the explicit (forward) Euler integration method.

    Notes
    -----
    - This method performs the following updates:
        1.  Applies boundary conditions using :meth:`system.domain.shift`.
        2.  Computes forces and accelerations using :meth:`system.collider.compute_force`.
        3.  Updates velocities based on current acceleration.
        4.  Updates positions based on the newly updated velocities.
    - Particles with `state.fixed` set to `True` will have their velocities
      and positions unaffected by the integration step.
    """

    @staticmethod
    @jax.jit
    def step(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Advances the simulation state by one time step using the Direct Euler method.

        The update equations are:

        .. math::
            & v(t + \\Delta t) &= v(t) + \\Delta t a(t) \\\\
            & r(t + \\Delta t) &= r(t) + \\Delta t v(t + \\Delta t)

        where:
            - :math:`r` is the particle position (:attr:`state.pos`)
            - :math:`v` is the particle velocity (:attr:`state.vel`)
            - :math:`a` is the particle acceleration (:attr:`state.accel`)
            - :math:`\\Delta t` is the time step (:attr:`system.dt`)

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The updated state and system after one time step.
        """
        state, system = system.domain.shift(state, system)
        state, system = system.collider.compute_force(state, system)
        state = replace(
            state,
            vel=state.vel + system.dt * state.accel * (1 - state.fixed)[..., None],
        )
        state = replace(
            state,
            pos=state.pos + system.dt * state.vel * (1 - state.fixed)[..., None],
        )
        return state, system

    @staticmethod
    @jax.jit
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        The Direct Euler integrator does not require a specific initialization step.


        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The original `State` and `System` objects.
        """
        return state, system
