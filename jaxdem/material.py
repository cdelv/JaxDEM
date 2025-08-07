# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining materials and the MaterialTable. The MaterialTable creates a SoA container for the materials. Different material types can be used if the force laws supports them.
"""

from dataclasses import dataclass, fields  # Add field import if not present
from typing import Dict, Sequence, Tuple, ClassVar

import jax
import jax.numpy as jnp

from .factory import Factory
from .materialMatchmaker import MaterialMatchmaker


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Material(Factory["Material"]):
    """
    Abstract base class for defining materials.

    Concrete subclasses of `Material` should define scalar fields (e.g., `young`, `poisson`, `mu`)
    that represent specific physical properties of a material. These fields are
    then collected and managed by the :class:`MaterialTable`.

    Notes
    -----
    - Each field defined in a concrete `Material` subclass will become a named property in the :attr:`MaterialTable.props` dictionary.
    """

    ...


@Material.register("elastic")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Elastic(Material):
    """
    A concrete `Material` implementation for elastic properties.

    This material type defines properties relevant for elastic interactions,
    such as Young's modulus and Poisson's ratio.


    Example
    -------
    >>> import jaxdem as jdem
    >>> elastic_steel = jdem.Material.create("elastic", young=2.0e11, poisson=0.3)
    """

    young: float
    poisson: float


@Material.register("elasticfrict")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ElasticFriction(Material):
    """
    A concrete `Material` implementation for elastic properties with friction.

    This material type extends :class:`Elastic` by adding a coefficient
    of friction, making it suitable for models that include frictional contact.

    Example
    -------
    >>> import jaxdem as jdem
    >>> frictional_rubber = jdem.Material.create("elasticfrict", young=1.0e7, poisson=0.49, mu=0.5)
    """

    young: float
    poisson: float
    mu: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MaterialTable:
    """
    A container for material properties, organized as Structures of Arrays (SoA)
    and pre-computed effective pair properties.

    This class centralizes material data, allowing efficient access to scalar
    properties for individual materials and pre-calculated effective properties
    for material-pair interactions.

    Notes
    -----
    - Scalar properties can be accessed directly using dot notation (e.g., `material_table.young`).
    - Effective pair properties can also be accessed directly using dot notation
      (e.g., `material_table.young_eff`).

    Example
    -------
    Creating a `MaterialTable` from multiple material types:

    >>> import jax.numpy as jnp
    >>> import jaxdem as jdem
    >>>
    >>> # Define different material instances
    >>> mat1 = jdem.Material.create("elastic", young=1.0e4, poisson=0.3)
    >>> mat2 = jdem.Material.create("elasticfrict", young=2.0e4, poisson=0.4, mu=0.5)
    >>>
    >>> # Create a MaterialTable using a linear matcher
    >>> matcher_instance = jdem.MaterialMatchmaker.create("linear")
    >>> mat_table = matcher_instance.from_materials(
    >>>     [mat1, mat2],
    >>>     matcher=matcher_instance
    >>> )
    """

    props: Dict[str, jax.Array]
    """
    A dictionary mapping scalar material property names (e.g., "young", "poisson", "mu")
    to JAX arrays. Each array has shape `(M,)`, where `M` is the total number
    of distinct material types present in the table.
    """

    pair: Dict[str, jax.Array]  # key → (M, M)
    """
    A dictionary mapping effective pair property names (e.g., "young_eff", "mu_eff")
    to JAX arrays. Each array has shape `(M, M)`, representing the effective
    property for interactions between any two material types (M_i, M_j).
    """

    matcher: MaterialMatchmaker
    """
    The :class:`MaterialMatchmaker` instance that was used to compute the
    effective pair properties stored in the :attr:`pair` dictionary.
    """

    @staticmethod
    def from_materials(
        mats: Sequence[Material],
        *,
        matcher: MaterialMatchmaker,
        fill: float = 0.0,
    ) -> "MaterialTable":
        """
        Constructs a :class:`MaterialTable` from a sequence of :class:`Material` instances.

        Parameters
        ----------
        mats : Sequence[Material]
            A sequence of concrete :class:`Material` instances. Each instance
            represents a distinct material type in the simulation. The order in
            this sequence defines their material IDs (0 to `len(mats)-1`).
        matcher : MaterialMatchmaker
            The :class:`MaterialMatchmaker` instance to be used for computing
            effective pair properties (e.g., harmonic mean, arithmetic mean).
        fill : float, optional
            A fill value used for material properties that are not defined in a
            specific `Material` subclass (e.g., if an :class:`Elastic` material
            is provided when an :class:`ElasticFriction` is expected, `mu`
            would be filled with this value). Defaults to 0.0.

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
        scalars: Dict[str, list[float]] = {k: [] for k in all_keys}
        for m in mats:
            for k in all_keys:
                scalars[k].append(getattr(m, k, fill))

        props = {k: jnp.asarray(v, dtype=float) for k, v in scalars.items()}
        pair = {
            f"{k}_eff": matcher.get_effective_property(a[:, None], a[None, :])
            for k, a in props.items()
        }
        return MaterialTable(props=props, pair=pair, matcher=matcher)

    def __getattr__(self, item: str) -> jax.Array:
        """
        Allows direct attribute access to scalar and effective pair properties.

        Parameters
        ----------
        item : str
            The name of the attribute being accessed (e.g., "young", "young_eff").

        Returns
        -------
        jax.Array
            The JAX array corresponding to the requested scalar or effective pair property.

        Raises
        ------
        AttributeError
            If `item` is not found as a scalar property in :attr:`props` or an effective pair property in :attr:`pair`.
        """
        if item in self.props:
            return self.props[item]
        if item in self.pair:
            return self.pair[item]
        raise AttributeError(item)

    def __len__(self) -> int:
        """
        Returns the number of distinct material types stored in the table.

        Returns
        -------
        int
            The number of materials, `M`. This corresponds to the length of any scalar property array.
        """
        return next(iter(self.props.values())).shape[0]

    # TODO: add and merge methods similar to State, returning the corresponding material ID when adding or merging.
    # Will need to handle the underlying Dict[str, jax.Array] structures and recompute pair properties.
    # This might require some JAX array manipulations within the `props` and `pair` dictionaries.
    # The `MaterialTable` is frozen, so methods would return new instances.

    # Example placeholders for future methods:
    # @staticmethod
    # def merge(table1: "MaterialTable", table2: "MaterialTable") -> "MaterialTable":
    #    """Merges two MaterialTable instances."""
    #    # Logic would involve combining props, then recomputing pair based on the combined set
    #    # and ensuring material IDs are consistent if coming from different tables.
    #    pass

    # def add_materials(self, mats: Sequence[Material], fill: float = 0.0) -> "MaterialTable":
    #    """Adds new materials to the table, returning a new MaterialTable instance."""
    #    # Logic would involve converting mats to a partial table, then merging with self.
    #    pass
