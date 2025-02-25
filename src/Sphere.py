# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax 
import jax.numpy as jnp

from dataclasses import dataclass

from jaxdem import Shape

@dataclass
class Sphere(Shape):
    """
    A geometric data for representation a sphere.

    Attributes
    ----------
    rad : jnp.ndarray
        The radius of the sphere.
    """
    _rad: jnp.ndarray = jnp.asarray(1.0, dtype=float)

    def __post_init__(self):
        self._rad = jnp.asarray(self._rad, dtype=float)

    @property
    def rad(self):
        return self._rad

