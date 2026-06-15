# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Time-integration interfaces and implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit(inline=True)
def free_mask(state: State) -> jax.Array:
    """Per-particle mask selecting non-fixed (free) particles.

    Returns a boolean array of shape `(..., N, 1)` that is `True` for free
    particles and `False` for fixed ones. Multiplying a `(..., N, dim)` update
    by this mask zeroes the update on fixed particles while leaving their
    prescribed velocities untouched. This is the single idiom integrators
    should use to respect :attr:`jaxdem.State.fixed`.
    """
    return (~state.fixed)[..., None]


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Integrator(Factory, ABC):
    """Abstract base class for defining the interface for time-stepping.

    Example:
    --------
    To define a custom integrator, inherit from :class:`Integrator` and implement its abstract methods:

    >>> @Integrator.register("myCustomIntegrator")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomIntegrator(Integrator):
            ...

    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="Integrator.step_before_force")
    def step_before_force(state: State, system: System) -> tuple[State, System]:
        """Advance the simulation state before the force evaluation.

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

        """
        return state, system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="Integrator.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        """Advance the simulation state after the force computation by one time step.

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

        """
        return state, system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="Integrator.initialize")
    def initialize(state: State, system: System) -> tuple[State, System]:
        """Some integration methods require an initialization step, for example LeapFrog.
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
        return state, system


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LinearIntegrator(Integrator):
    """Namespace for translation/linear-time integrators.

    Purpose
    -------
    Groups integrators that update linear state (e.g., position and velocity).
    Concrete methods (e.g., DirectEuler) should subclass this to register via
    the Factory and to signal they operate on linear kinematics.
    """


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class RotationIntegrator(Integrator):
    """Namespace for rotation/angular-time integrators.

    Purpose
    -------
    Groups integrators that update angular state (e.g., orientation, angular
    velocity). Concrete methods (e.g., DirectEulerRotation) should subclass
    this to register via the Factory and to signal they operate on rotational
    kinematics.
    """


LinearIntegrator.register("")(LinearIntegrator)
RotationIntegrator.register("")(RotationIntegrator)


from .direct_euler import DirectEuler
from .langevin import Langevin
from .spiral import Spiral
from .velocity_verlet import VelocityVerlet
from .velocity_verlet_rescaling import VelocityVerletRescaling
from .velocity_verlet_spiral import VelocityVerletSpiral
from .vicsek import VicsekExtrinsic, VicsekIntrinsic

__all__ = [
    "DirectEuler",
    "Langevin",
    "LinearIntegrator",
    "free_mask",
    "RotationIntegrator",
    "Spiral",
    "VelocityVerlet",
    "VelocityVerletRescaling",
    "VelocityVerletSpiral",
    "VicsekExtrinsic",
    "VicsekIntrinsic",
]
