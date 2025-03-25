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

class Integrator(Factory, ABC):
    """
    Abstract class defining the interface for force calculation models.
    Subclasses must implement `calculate_force`.
    """
    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """Calculate the energy of the interaction between particles i and j."""
        ...

@Integrator.register("euler")
class DirectEuler(Integrator):
    @staticmethod
    @partial(jax.jit, inline=True)
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
        state.vel += system.dt * state.accel
        state.pos += system.dt * state.vel
        state.pos -= system.domain.shift(state.pos, system)
        return state, system