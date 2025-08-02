# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining materials and the MaterialTable. The MaterialTable creates a SoA container for the materials. Different material types can be used if the force laws supports them.
"""

from dataclasses import dataclass, fields
from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from .factory import Factory
from .materialMatchmaker import MaterialMatchmaker

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Material(Factory["Material"]):
    """Base class – concrete subclasses define their own scalar fields."""

@Material.register("elastic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Elastic(Material):
    young: float
    poisson: float

@Material.register("elasticfrict")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ElasticFriction(Material):
    young: float
    poisson: float
    mu: float

@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MaterialTable:
    """
    Aligned property arrays + pre-mixed pair matrices.
    Scalar:  table.young[i]
    Mixed :  table.young_eff[i, j]
    """
    props: Dict[str, jax.Array]            # key → (M,)
    pair:  Dict[str, jax.Array]            # key → (M, M)
    matcher: MaterialMatchmaker

    @staticmethod
    def from_materials(mats: Sequence[Material], *,
        matcher: MaterialMatchmaker,
        fill: float = 0.0,
    ) -> "MaterialTable":
        all_keys = {f.name for m in mats for f in fields(m)}
        scalars: Dict[str, list[float]] = {k: [] for k in all_keys}
        for m in mats:
            for k in all_keys:
                scalars[k].append(getattr(m, k, fill))

        props = {k: jnp.asarray(v, dtype=float) for k, v in scalars.items()}
        pair  = {
            f"{k}_eff": matcher.get_effective_property(a[:, None], a[None, :])
            for k, a in props.items()
        }
        return MaterialTable(props=props, pair=pair, matcher=matcher)

    def __getattr__(self, item):
        if item in self.props:
            return self.props[item]
        if item in self.pair:
            return self.pair[item]
        raise AttributeError(item)

    def __len__(self):                        
        return next(iter(self.props.values())).shape[0]


# add and merge methods. Return the corresponding material ID when adding or merging