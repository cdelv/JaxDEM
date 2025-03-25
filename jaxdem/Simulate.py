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
    Abstract class defining the interface for force calculation models.
    Subclasses must implement `calculate_force`.
    """
    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def compute_force(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """Calculate the force between particles i and j."""
        ...

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """Calculate the energy of the interaction between particles i and j."""
        ...

@Simulator.register("naive")
class NaiveSimulator(Simulator):
    @staticmethod
    @partial(jax.jit, inline=True)
    def compute_force(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """Calculate the force between particles i and j."""
        state.accel = jax.vmap(
            lambda i: jax.vmap(
                lambda j: 
                    system.force_model.calculate_force(i, j, state, system)
            )(jax.lax.iota(dtype=int, size=state.N)).sum(axis=0)
        )(jax.lax.iota(dtype=int, size=state.N))
        return state, system

    @staticmethod
    @partial(jax.jit, inline=True)
    def step(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """Calculate the energy of the interaction between particles i and j."""
        state, system = NaiveSimulator.compute_force(state, system)
        state, system = system.integrator.step(state, system)
        return state, system