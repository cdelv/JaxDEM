# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from functools import partial

from jaxdem.Factory import Factory
from jaxdem.State import State
from jaxdem.System import System

class ForceModel(Factory, ABC):
    """
    Abstract class defining the interface for force calculation models.
    Subclasses must implement `calculate_force`.
    """
    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def calculate_force(i: int, j: int, state: 'State', system: 'System') -> jnp.ndarray:
        """Calculate the force between particles i and j."""
        ...

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def calculate_energy(i: int, j: int, state: 'State', system: 'System') -> float:
        """Calculate the energy of the interaction between particles i and j."""
        ...

@ForceModel.register("spring")
class SpringForce(ForceModel):
    """A valid force model that implements `calculate_force`."""

    @staticmethod
    @partial(jax.jit, inline=True)
    def calculate_force(i: int, j: int, state: 'State', system: 'System') -> jnp.ndarray:
        r_ij = system.domain.displacement(state.pos[i], state.pos[j], system)
        r = jnp.linalg.norm(r_ij)
        s = jnp.maximum(0.0, (state.rad[i] + state.rad[j])/(r + jnp.finfo(state.pos.dtype).eps) - 1.0)
        return system.k * s * r_ij

    @staticmethod
    @partial(jax.jit, inline=True)
    def calculate_energy(i: int, j: int, state: 'State', system: 'System') -> float:
        """Calculate the energy of the interaction between particles i and j."""
        ...