# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from typing import Tuple
from abc import ABC, abstractmethod
from functools import partial

from jaxdem.Factory import Factory
from jaxdem.System import System

class Domain(Factory, ABC):
    """
    Abstract class defining the interface for force calculation models.
    Subclasses must implement `calculate_force`.
    """
    def __init__(self, box_size: jnp.ndarray = jnp.ones(3), anchor: jnp.ndarray = jnp.zeros(3)):
        self.box_size = box_size
        self.anchor = anchor

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        """Calculate the displacement vector for the given metric"""
        ...

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def shift(r:jnp.ndarray, system: 'System') -> jnp.ndarray:
        """Calculate the displacement vector for the given metric"""
        ...

@Domain.register("free")
class FreeDomain(Domain):
    @staticmethod
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        return ri - rj

    @staticmethod
    @partial(jax.jit, inline=True)
    def shift(r:jnp.ndarray, system: 'System') -> jnp.ndarray:
        return jnp.zeros_like(r)

@Domain.register("periodic")
class PeriodicDomain(Domain):
    @staticmethod
    @partial(jax.jit, inline=True)
    def displacement(ri:jnp.ndarray, rj: jnp.ndarray, system: 'System') -> jnp.ndarray:
        rij = (ri - system.domain.anchor) - (rj - system.domain.anchor)
        return rij - system.domain.box_size * jnp.round(rij / system.domain.box_size)

    @staticmethod
    @partial(jax.jit, inline=True)
    def shift(r:jnp.ndarray, system: 'System') -> jnp.ndarray:
        return system.domain.box_size * jnp.floor((r - system.domain.anchor) / system.domain.box_size)