# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC
from dataclasses import dataclass, fields

from .Factory import Factory

@jax.tree_util.register_dataclass
@dataclass
class Material(Factory, ABC):
    """Base material class"""

    @classmethod
    def combine_materials(cls, *materials: "Material") -> "Material":
        """Combine multiple Material instances by concatenating all fields"""
        merged_data = {}
        
        for field in fields(cls):
            merged_data[field.name] = jnp.concatenate([getattr(mat, field.name) for mat in materials], axis=0)

        return cls(**merged_data)

    @property
    def N(self) -> int:
        # complete
        return 0

    @property
    def is_valid(self) -> int | bool:
        """
        Validate that the state has the expected structure.

        Returns
        -------
        bool
            True if the state is valid; otherwise, False.
        """
        valid = True

        # Complete

        return valid


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
@Material.register("elastic")
class ElasticMaterial(Material):
    """Elastic material"""
    youngs_modulus: jax.Array = jnp.asarray([1e3], dtype=float)
    poissons_ratio: jax.Array = jnp.asarray([0.2], dtype=float)
    friction: jax.Array = jnp.asarray([0.1], dtype=float)

    def __post_init__(self):
        self.youngs_modulus = jnp.asarray(self.youngs_modulus, dtype=float)
        self.poissons_ratio = jnp.asarray(self.poissons_ratio, dtype=float)
        self.friction = jnp.asarray(self.friction, dtype=float)

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