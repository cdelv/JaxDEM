# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Material mix rules and implementations."""

from __future__ import annotations

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from ..factory import Factory


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MaterialMatchmaker(Factory, ABC):
    """
    Abstract base class for defining how to combine (mix) material properties.

    Notes
    -----
    - These matchmakers are used by the :class:`jaxdem.MaterialTable` to pre-compute interaction matrices.

    Example
    -------
    To define a custom matchmaker, inherit from :class:`MaterialMatchmaker` and implement
    its abstract methods:

    >>> @MaterialMatchmaker.register("myCustomForce")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomMatchmaker(MaterialMatchmaker):
            ...
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
        """
        raise NotImplementedError


from .harmonic import HarmonicMaterialMatchmaker
from .linear import LinearMaterialMatchmaker

__all__ = [
    "MaterialMatchmaker",
    "HarmonicMaterialMatchmaker",
    "LinearMaterialMatchmaker",
]
