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
    """Abstract base class for defining materials.

    Concrete subclasses of `Material` should define scalar or vector fields (e.g., `young`, `poisson`, `mu`)
    that represent specific physical properties of a material. These fields are
    then collected and managed by the :class:`MaterialTable`.

    Notes:
    ------
    - Each field defined in a concrete `Material` subclass will become a named property in the :attr:`MaterialTable.props` dictionary.

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
