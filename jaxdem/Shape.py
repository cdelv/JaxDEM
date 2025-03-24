# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp 

class shape(ABC):
    """
    Abstract class for representing particle shapes.
    """
    @abstractmethod
    def __init__(self):
        pass

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class sphere(shape):
    """
    A geometric data for representation a sphere.

    Attributes
    ----------
    rad : jnp.ndarray
        The radius of the sphere.
    """
    rad: jnp.ndarray = jnp.asarray(1.0, dtype=float)

    def __post_init__(self):
        self.rad = jnp.asarray(self.rad, dtype=float)
