# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Interface for defining materials and the MaterialTable."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from ..factory import Factory


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Material(Factory):
    """Abstract base class for materials.

    Concrete subclasses of `Material` define scalar or vector fields (e.g., `young`, `poisson`, `mu`)
    for the physical properties of a material. The :class:`MaterialTable`
    collects and manages these fields.

    Notes:
    ------
    - Each field of a concrete `Material` subclass becomes a named property in the :attr:`MaterialTable.props` dictionary.

    Example:
    --------
    To define a custom material, inherit from `Material`

    >>> @Material.register("my_custom_material")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomMaterial(Material):
            ...

    """

    density: float


from .elastic_mats import Elastic, ElasticFriction
from .lj_mats import LJMaterial
from .material_table import MaterialTable

__all__ = ["Elastic", "ElasticFriction", "LJMaterial", "Material", "MaterialTable"]
