# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of some variations of elastic materials.
"""
from __future__ import annotations

import jax

from dataclasses import dataclass

from . import Material


@Material.register("elastic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Elastic(Material):
    """
    Example
    -------
    >>> import jaxdem as jdem
    >>> elastic_steel = jdem.Material.create("elastic", young=2.0e11, poisson=0.3)
    """

    young: float
    poisson: float


@Material.register("elasticfrict")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ElasticFriction(Material):
    """
    Example
    -------
    >>> import jaxdem as jdem
    >>> frictional_rubber = jdem.Material.create("elasticfrict", young=1.0e7, poisson=0.49, mu=0.5)
    """

    young: float
    poisson: float
    mu: float


__all__ = ["Elastic", "ElasticFriction"]
