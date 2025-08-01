# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .factory import Factory

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MaterialMatchmaker(Factory["MaterialMatchmaker"], ABC):
    """Class for defining how to mix different materials"""
    @staticmethod
    @abstractmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        raise NotImplementedError

@MaterialMatchmaker.register("linear")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LinearMaterialMatchmaker(MaterialMatchmaker):
    @staticmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return (prop1 + prop2)/2


@MaterialMatchmaker.register("harmonic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class HarmonicMaterialMatchmaker(MaterialMatchmaker):
    @staticmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return 2.0/(1.0/prop1 + 1.0/prop2)