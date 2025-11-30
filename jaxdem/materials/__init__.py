# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining materials and the MaterialTable.
"""
from __future__ import annotations

import jax

from dataclasses import dataclass

from ..factory import Factory


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Material(Factory):
    """
    Abstract base class for defining materials.

    Concrete subclasses of `Material` should define scalar or vector fields (e.g., `young`, `poisson`, `mu`)
    that represent specific physical properties of a material. These fields are
    then collected and managed by the :class:`MaterialTable`.

    Notes
    -----
    - Each field defined in a concrete `Material` subclass will become a named property in the :attr:`MaterialTable.props` dictionary.

    Example
    -------
    To define a custom material, inherit from `Material`

    >>> @Material.register("my_custom_material")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True, frozen=True)
    >>> class MyCustomMaterial(Material):
            ...
    """

    density: float


from .materialTable import MaterialTable
from .elasticMats import Elastic, ElasticFriction


__all__ = ["Material", "MaterialTable", "ElasticFriction", "Elastic"]
