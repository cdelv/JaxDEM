# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Time-integration interfaces and implementations."""

from __future__ import annotations

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Integrator(Factory, ABC):
    """
    Abstract base class for defining the interface for time-stepping.


    Example
    -------
    To define a custom integrator, inherit from :class:`Integrator` and implement its abstract methods:

    >>> @Integrator.register("myCustomIntegrator")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True, frozen=True)
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

        Notes
        -----
        - This method performs the following updates:
            1.  Applies boundary conditions using :meth:`system.domain.shift`.
            2.  Computes forces and accelerations using :meth:`system.collider.compute_force`.
            3.  Updates velocities based on current acceleration.
            4.  Updates positions based on the newly updated velocities.
        - Particles with `state.fixed` set to `True` will have their velocities
          and positions unaffected by the integration step.

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

        Example
        -------

        >>> state, system = system.integrator.initialize(state, system)
        """
        raise NotImplementedError


from .direct_euler import DirectEuler

__all__ = ["Integrator", "DirectEuler"]
