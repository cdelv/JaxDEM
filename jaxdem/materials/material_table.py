# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""The MaterialTable stores materials in a structure of arrays (SoA). Materials of different types can share a table when the force law supports it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ..material_matchmakers import MaterialMatchmaker

if TYPE_CHECKING:  # pragma: no cover
    from . import Material


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MaterialTable:
    """A container for material properties, organized as Structures of Arrays (SoA)
    and pre-computed effective pair properties.

    The table gives direct access to the scalar properties of each material
    and to the pre-computed effective properties for material pairs.

    Notes:
    ------
    - Access scalar properties directly with dot notation (e.g., `material_table.young`).
    - Access effective pair properties directly with dot notation
      (e.g., `material_table.young_eff`).

    Example:
    --------
    Creating a `MaterialTable` from multiple material types:

    >>> import jax.numpy as jnp
    >>> import jaxdem as jdem
    >>>
    >>> # Define different material instances
    >>> mat1 = jdem.Material.create("elastic", density=2500.0, young=1.0e4, poisson=0.3)
    >>> mat2 = jdem.Material.create("elasticfrict", density=7800.0, young=2.0e4, poisson=0.4, mu=0.5, e=1.0)
    >>>
    >>> # Create a MaterialTable using a linear matcher
    >>> matcher_instance = jdem.MaterialMatchmaker.create("linear")
    >>> mat_table = jdem.MaterialTable.from_materials(
    >>>     [mat1, mat2],
    >>>     matcher=matcher_instance
    >>> )

    """

    props: dict[str, jax.Array]
    """
    A dictionary mapping scalar material property names (e.g., "young", "poisson", "mu")
    to JAX arrays. Each array has shape `(M,)`, where `M` is the total number
    of distinct material types present in the table.
    """

    pair: dict[str, jax.Array]  # key → (M, M)
    """
    A dictionary mapping effective pair property names (e.g., "young_eff", "mu_eff")
    to JAX arrays. Each array has shape `(M, M)` and holds the effective
    property for interactions between any two material types (M_i, M_j).
    """

    matcher: MaterialMatchmaker
    """
    The :class:`jaxdem.MaterialMatchmaker` instance that computed the
    effective pair properties stored in the :attr:`pair` dictionary.
    """

    @staticmethod
    @partial(jax.named_call, name="MaterialTable.from_materials")
    def from_materials(
        mats: Sequence[Material],
        *,
        matcher: MaterialMatchmaker | None = None,
        fill: float = 0.0,
    ) -> MaterialTable:
        """Construct a :class:`MaterialTable` from a sequence of :class:`Material` instances.

        Parameters
        ----------
        mats : Sequence[Material]
            A sequence of concrete :class:`Material` instances. Each instance
            represents a distinct material type in the simulation. The order in
            this sequence defines their material IDs (0 to `len(mats)-1`).
        matcher : MaterialMatchmaker
            The :class:`jaxdem.MaterialMatchmaker` instance used to compute
            effective pair properties (e.g., harmonic mean, arithmetic mean).
            If `None`, defaults to the harmonic matchmaker.
        fill : float, optional
            Fill value for material properties that a `Material` subclass does
            not define. For example, if an :class:`Elastic` material appears
            with an :class:`ElasticFriction` material, `mu` takes this value.
            Defaults to 0.0.

        Returns
        -------
        MaterialTable
            A new `MaterialTable` instance containing the scalar properties and
            pre-computed effective pair properties for all provided materials.

        Raises
        ------
        TypeError
            If `mats` is not a sequence of `Material` instances.

        """
        all_keys = {f.name for m in mats for f in fields(m)}
        scalars: dict[str, list[float]] = {k: [] for k in all_keys}
        for m in mats:
            for k in all_keys:
                scalars[k].append(getattr(m, k, fill))

        if matcher is None:
            matcher = MaterialMatchmaker.create("harmonic")

        props = {k: jnp.asarray(v, dtype=float) for k, v in scalars.items()}
        pair = {
            f"{k}_eff": matcher.get_effective_property(a[:, None], a[None, :])
            for k, a in props.items()
        }
        return MaterialTable(props=props, pair=pair, matcher=matcher)

    @partial(jax.named_call, name="MaterialTable.__getattr__")
    def __getattr__(self, item: str) -> jax.Array:
        """Give direct attribute access to scalar and effective pair properties.

        Parameters
        ----------
        item : str
            The name of the attribute to access (e.g., "young", "young_eff").

        Returns
        -------
        jax.Array
            The JAX array for the requested scalar or effective pair property.

        Raises
        ------
        AttributeError
            If `item` is not a scalar property in :attr:`props` or an effective pair property in :attr:`pair`.

        """
        # Guard against infinite recursion: `__getattr__` is only called when normal
        # attribute lookup fails, which happens for the dataclass fields themselves
        # during object construction (e.g. `copy.deepcopy`, `pickle`) before they are
        # set. Accessing `self.props` below would then recurse forever.
        if item in ("props", "pair", "matcher") or item.startswith("__"):
            raise AttributeError(item)
        if item in self.props:
            return self.props[item]
        if item in self.pair:
            return self.pair[item]
        raise AttributeError(item)

    @partial(jax.named_call, name="MaterialTable.__len__")
    def __len__(self) -> int:
        """Return the number of distinct material types in the table.

        Returns
        -------
        int
            The number of materials, `M`. This equals the length of any scalar property array.

        """
        return next(iter(self.props.values())).shape[0]

    # TODO: add and merge methods similar to State, returning the corresponding material ID when adding or merging.
    # Will need to handle the underlying Dict[str, jax.Array] structures and recompute pair properties.
    # This might require some JAX array manipulations within the `props` and `pair` dictionaries.

    # Example placeholders for future methods:
    # @staticmethod
    # def merge(table1: MaterialTable, table2: MaterialTable) -> MaterialTable:
    #    """Merges two MaterialTable instances."""
    #    # Logic would involve combining props, then recomputing pair based on the combined set
    #    # and ensuring material IDs are consistent if coming from different tables.
    #    pass

    @property
    def metadata(self) -> dict[str, Any]:
        """MaterialTable configuration parameters for serialization and restoration."""
        return {
            "num_materials": len(self),
            "prop_keys": list(self.props.keys()),
            "pair_keys": list(self.pair.keys()),
            "matcher_type": self.matcher.type_name,
        }


__all__ = ["MaterialTable"]
