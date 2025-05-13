# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
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
        ...

    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def get_effective_material(cls, m1: "Material", m2: "Material") -> "Material":
        assert type(m1) is type(m2), f"Can't mix {type(m1)} with {type(m2)}"

        values = {}
        for f in fields(m1):
            if f.name in ["_registry"]:
                continue 

            v1 = getattr(m1, f.name)
            v2 = getattr(m2, f.name)

            values[f.name] = cls.get_effective_property(v1, v2)

        return type(m1)(**values)

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