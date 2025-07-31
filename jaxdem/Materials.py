# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

from dataclasses import dataclass, fields
from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from .MaterialMatchmaker import MaterialMatchmaker

@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Material:
    """Marker base class – concrete subclasses define their own fields."""

@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ElasticMat(Material):
    young: float
    poisson: float

@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ElasticFrictMat(Material):
    young: float
    poisson: float
    mu: float

# ------------------------------------------------------------------ #
# 2.  MaterialTable                                                  #
# ------------------------------------------------------------------ #
@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MaterialTable:
    """
    Container holding *aligned* property arrays and the pre-mixed pair matrices.

    Access scalar property i via `table.young[i]`; access mixed property via `table.young_eff[i, j]`. Field names as in the Material class.
    """
    # scalar arrays (1-D)
    props: Dict[str, jax.Array]            # key → (M,)
    # pair matrices (2-D)
    pair:  Dict[str, jax.Array]            # key → (M, M)
    # match-maker used to build the table
    matcher: MaterialMatchmaker

    # ---------------------- construction helpers ------------------- #
    @staticmethod
    def from_materials(mats: Sequence[Material], *, matcher: MaterialMatchmaker, fill: float = 0.0) -> "MaterialTable":
        """
        Build a table from an *ordered* list of material objects.
        Missing scalar properties are filled with `fill`.
        """
        # ------ gather all scalar field names ---------------------- #
        all_keys: set[str] = set()
        for m in mats:
            all_keys.update(f.name for f in fields(m))

        # ------ build (M,) arrays ---------------------------------- #
        scalars: Dict[str, list[float]] = {k: [] for k in all_keys}
        for m in mats:
            for k in all_keys:
                scalars[k].append(getattr(m, k, fill))

        props_1d = {k: jnp.asarray(v, dtype=float) for k, v in scalars.items()}

        # ------ pre-mix pair matrices ------------------------------ #
        pair_2d = {
            f"{k}_eff": matcher.get_effective_property(a[:, None], a[None, :])
            for k, a in props_1d.items()
        }

        return MaterialTable(props=props_1d, pair=pair_2d, matcher=matcher)

    # ---------------------- convenience look-ups ------------------- #
    def __getattr__(self, item):
        if item in self.props:
            return self.props[item]
        if item in self.pair:
            return self.pair[item]
        raise AttributeError(item)

    # ---------------------- add / merge ---------------------------- #
    def add(self, mat: Material) -> "MaterialTable":
        """
        Return a new table that appends *mat* to the current list.
        """
        mats = [self.material(i) for i in range(len(self))] + [mat]
        return MaterialTable.from_materials(mats, matcher=self.matcher)

    @staticmethod
    def merge(t1: "MaterialTable", t2: "MaterialTable") -> "MaterialTable":
        """
        Stack two tables vertically (materials in t2 get new indices).
        Match-maker must be identical.
        """
        assert type(t1.matcher) is type(t2.matcher), "different match-makers"
        mats = [t1.material(i) for i in range(len(t1))] + \
               [t2.material(i) for i in range(len(t2))]
        return MaterialTable.from_materials(mats, matcher=t1.matcher)

    # ---------------------- helpers -------------------------------- #
    def __len__(self):
        # length of any 1-D property
        return next(iter(self.props.values())).shape[0]

    def material(self, idx: int) -> Material:
        """
        Re-assemble a Material object of the *minimal* concrete subclass
        that owns exactly the present scalar fields (for debug / IO).
        """
        data = {k: float(a[idx]) for k, a in self.props.items()}
        NT = dataclass(  # type: ignore[misc]
            type(f"Mat{idx}", (Material,), {}),
            slots=True,
        )(**data)
        return NT