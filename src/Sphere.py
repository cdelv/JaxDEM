# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax 
import jax.numpy as jnp

from functools import partial

from jaxdem import Shape

class Sphere(Shape):
    """
    A geometric data for representation a sphere.

    Attributes
    ----------
    rad : jnp.ndarray
        The radius of the sphere.
    """
    def __init__(self, rad: float = 1.0):
        """
        Initialize a Sphere instance.

        Parameters
        ----------
        rad : float, optional
            The radius of the sphere. Default is 1.0.
        """
        self._rad = jnp.asarray(rad, dtype=float)

    @property
    @partial(jax.jit, static_argnums=(0,))
    def rad(self):
        """
        Get the radius of the sphere.

        Returns
        -------
        jnp.ndarray
            The radius of the sphere as a JAX array.
        """
        return self._rad
