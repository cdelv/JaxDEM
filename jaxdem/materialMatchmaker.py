# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Tooling for mixing materials of the same type. The MaterialMatchmaker defines how to compute the effective material property.
"""

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .factory import Factory


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MaterialMatchmaker(Factory["MaterialMatchmaker"], ABC):
    """
    Abstract base class for defining how to combine (mix) material properties.

    Notes
    -----
    - Implementations should be JIT-compilable and work with JAX's transformation functions.
    - These matchmakers are used by the :class:`MaterialTable` to pre-compute interaction matrices.
    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        """
        Abstract method to compute the effective property value from two individual material properties.

        Concrete implementations define the specific mixing rule

        Parameters
        ----------
        prop1 : jax.Array
            The property value from the first material. Can be a scalar or an array.
        prop2 : jax.Array
            The property value from the second material. Can be a scalar or an array.

        Returns
        -------
        jax.Array
            A JAX array representing the effective property, computed from `prop1` and `prop2`
            according to the matchmaker's specific rule.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.
        """

        raise NotImplementedError


@MaterialMatchmaker.register("linear")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LinearMaterialMatchmaker(MaterialMatchmaker):
    """
    A `MaterialMatchmaker` implementation that computes the effective property
    as the arithmetic mean (linear average) of two properties:

    .. math::
        P_{eff} = \\frac{P_1 + P_2}{2}

    where :math:`P_1` and :math:`P_2` are the property values from the two materials.
    """

    @staticmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return (prop1 + prop2) / 2


@MaterialMatchmaker.register("harmonic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class HarmonicMaterialMatchmaker(MaterialMatchmaker):
    """
    A `MaterialMatchmaker` implementation that computes the effective property
    as the harmonic mean of two properties:

    .. math::
        P_{eff} = \\frac{2}{\\frac{1}{P_1} + \\frac{1}{P_2}}

    where :math:`P_1` and :math:`P_2` are the property values from the two materials.
    """

    @staticmethod
    @jax.jit
    def get_effective_property(prop1: jax.Array, prop2: jax.Array) -> jax.Array:
        return 2.0 / (1.0 / prop1 + 1.0 / prop2)
