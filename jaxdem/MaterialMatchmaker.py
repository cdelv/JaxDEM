# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

@jax.tree_util.register_dataclass
@dataclass
class MaterialMatchmaker(Factory, ABC):
    """
    .
    """
    @classmethod
    @abstractmethod
    @partial(jax.jit, static_argnames=("cls"))
    def get_effective_property(cls, prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        raise NotImplemented

@jax.tree_util.register_dataclass
@dataclass
@MaterialMatchmaker.register("linear")
class LinearMaterialMatchmaker(MaterialMatchmaker):
    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def get_effective_property(cls, prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return (prop1 + prop2)/2

@jax.tree_util.register_dataclass
@dataclass
@MaterialMatchmaker.register("harmonic")
class HarmonicMaterialMatchmaker(MaterialMatchmaker):
    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def get_effective_property(cls, prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return 2.0/(1.0/prop1 + 1.0/prop2)