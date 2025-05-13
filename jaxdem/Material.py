# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

@jax.tree_util.register_dataclass
@dataclass
class Material(Factory, ABC):
    """Base material class"""


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
@Material.register("elastic")
class ElasticMaterial(Material):
    """Elastic material"""
    youngs_modulus: jax.Array = jnp.asarray([1e3], dtype=float)
    poissons_ratio: jax.Array = jnp.asarray([0.2], dtype=float)
    friction: jax.Array = jnp.asarray([0.1], dtype=float)


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
@Material.register("Lennard-Jones")
class LennardJonesMaterial(Material):
    """Lennard Jones material"""
    sigma: jax.Array = jnp.asarray([1.0], dtype=float)
    epsilon: jax.Array = jnp.asarray([0.1], dtype=float)

    def __post_init__(self):
        self.sigma = jnp.asarray(self.sigma, dtype=float)
        self.epsilon = jnp.asarray(self.epsilon, dtype=float)