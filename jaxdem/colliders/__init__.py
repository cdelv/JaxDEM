# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Collision-detection interfaces and implementations."""

from __future__ import annotations

import dataclasses
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

import jax
import jax.numpy as jnp

from ..factory import Factory
from ..domains import Domain, SearchGeometry
from ..utils.linalg import norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Collider(Factory, ABC):
    r"""The base interface for contact detection and force computation in a simulation.

    Concrete subclasses of `Collider` implement the interaction algorithms.

    Notes:
    ------
    Self-interaction (calling the force/energy computation for `i=j`) is allowed.
    The `force_model` must handle or ignore this case correctly.

    Example:
    --------
    To define a custom collider, inherit from `Collider`, register it, and implement its abstract methods:

    >>> @Collider.register("CustomCollider")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class CustomCollider(Collider):
            ...

    Then, instantiate it:

    >>> jaxdem.Collider.create("CustomCollider", **custom_collider_kw)

    """

    overflow: jax.Array = field(
        default_factory=lambda: jnp.array(False, dtype=bool), kw_only=True
    )
    """True when a collider overflow occurred."""

    supported_search_geometries: ClassVar[frozenset[SearchGeometry] | None] = None
    """Supported hashed geometries, or ``None`` for displacement-only search."""

    def validate_domain(self, domain: Domain) -> None:
        """Validate construction-time compatibility with ``domain``."""
        supported = self.supported_search_geometries
        if supported is None:
            return
        geometry = domain.search_geometry
        if geometry is None:
            raise ValueError(
                f"{type(domain).__name__} does not declare a search geometry; "
                "custom domains must explicitly set Domain.search_geometry."
            )
        if geometry not in supported:
            raise ValueError(
                f"{type(self).__name__} does not support the "
                f"{geometry.value!r} search geometry."
            )

    def invalidate(self) -> Collider:
        """Return this collider with any search cache marked invalid."""
        return self

    @property
    def stateful(self) -> bool:
        """Whether this collider caches state-size-dependent search data."""
        return False

    @property
    def supports_history(self) -> bool:
        """Whether this collider implements the NeighborList history contract."""
        return False

    @staticmethod
    @jax.jit(inline=True)
    def compute_force(state: State, system: System) -> tuple[State, System]:
        """Compute the total force acting on each particle in the simulation.

        This base implementation is a concrete no-op: it zeroes the ``force``
        and ``torque`` attributes of the ``state`` and returns. It backs the
        ``""`` (empty-string) no-op collider registration for systems whose
        dynamics come only from bonded forces or user force functions.

        Subclasses override it to compute inter-particle forces and torques
        from the current `state` and `system` configuration. They write the
        total force and torque of each particle to the `force` and `torque`
        attributes of the `state` object.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated `State` object (with computed forces) and the `System` object.

        """

        state.force *= 0
        state.torque *= 0
        return state, system

    @staticmethod
    @jax.jit(inline=True)
    def evaluate_force(state: State, system: System) -> tuple[State, System]:
        """Evaluate forces without advancing contact history."""
        return system.collider.compute_force(state, system)

    @staticmethod
    def get_history(
        state: State, system: System, neighbor_list: jax.Array
    ) -> jax.Array:
        """Return initialized history for an explicit pair query."""
        return system.force_model.init_history(neighbor_list.shape, state.dim)

    @staticmethod
    @jax.jit(inline=True)
    def compute_potential_energy(
        state: State, system: System
    ) -> tuple[State, System, jax.Array]:
        """Compute the total (scalar) non-bonded potential energy of the system.

        Implementations sum every pair-interaction contribution defined by
        ``system.force_model`` and return a single scalar. They weight pair
        energies with the standard 0.5 factor, so each pair counts once even
        when the neighbor list visits ``(i, j)`` and ``(j, i)`` separately.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System, jax.Array]
            A tuple of (state, system, potential_energy). The potential_energy
            is a scalar JAX array (shape ``()``) with the total non-bonded
            potential energy of the system.

        Example
        -------

        >>> state, system, potential_energy = system.collider.compute_potential_energy(state, system)
        >>> print(f"Total potential energy: {float(potential_energy):.4f}")
        >>> print(potential_energy.shape)  # ()

        """
        return state, system, jnp.asarray(0.0)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[State, System, jax.Array, jax.Array]:
        """Build a neighbor list for the current collider.

        Neighbor-list-based algorithms and diagnostics use this list.
        Implementations match the cell-list semantics:

        - Return a neighbor list of shape ``(N, max_neighbors)`` padded with ``-1``.
        - Neighbor indices refer to the returned ``state``.
        - Also return an ``overflow`` boolean flag. The flag is True when any
          particle has more than ``max_neighbors`` neighbors within the cutoff.
        """
        raise NotImplementedError

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Build a cross-neighbor list between two sets of positions.

        For each point in ``pos_a``, find all neighbors from ``pos_b``
        within the ``cutoff`` distance. Use this to couple different particle
        systems or to compute interactions between distinct sets of objects.

        The default implementation runs a naive :math:`O(N_A \times N_B)`
        all-pairs search. Subclasses can override it with faster algorithms.

        Parameters
        ----------
        pos_a : jax.Array
            Query positions, shape ``(N_A, dim)``.
        pos_b : jax.Array
            Database positions, shape ``(N_B, dim)``.
        system : System
            The configuration of the simulation (used for domain displacement).
        cutoff : float
            Search radius.
        max_neighbors : int
            Maximum number of neighbors to store per query point.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            A tuple containing:

            - ``neighbor_list``: Array of shape ``(N_A, max_neighbors)`` containing
              indices into ``pos_b``, padded with ``-1``.
            - ``overflow``: Boolean flag. True when any query point has more
              than ``max_neighbors`` neighbors within the cutoff.

        """
        if max_neighbors == 0:
            n_a = pos_a.shape[0]
            empty = jnp.empty((n_a, 0), dtype=int)
            return empty, jnp.asarray(False)

        n_b = pos_b.shape[0]
        iota_b = jax.lax.iota(dtype=int, size=n_b)
        cutoff_sq = jnp.asarray(cutoff, dtype=pos_a.dtype) ** 2

        def per_query(pos_ai: jax.Array) -> tuple[jax.Array, jax.Array]:
            dr = system.domain.displacement(pos_ai, pos_b, system)
            dist_sq = norm2(dr)
            valid = dist_sq <= cutoff_sq
            num_neighbors = jnp.sum(valid)
            overflow_flag = num_neighbors > max_neighbors
            candidates = jnp.where(valid, iota_b, -1)
            k_eff = min(max_neighbors, n_b)
            topk = jax.lax.top_k(candidates, k_eff)[0]
            if k_eff < max_neighbors:
                topk = jnp.concatenate(
                    [topk, jnp.full((max_neighbors - k_eff,), -1, dtype=topk.dtype)]
                )
            return topk, overflow_flag

        nl, overflows = jax.vmap(per_query)(pos_a)
        return nl, jnp.any(overflows)


# The base class doubles as a no-op collider (zero force/torque, zero
# potential energy) registered under the empty-string key, mirroring the
# integrators' "" registration. Use ``collider_type=""`` for systems whose
# dynamics come exclusively from bonded forces or user force functions.
Collider.register("")(Collider)


@jax.jit(inline=True)
def valid_interaction_mask(
    clump_i: jax.Array,
    clump_j: jax.Array,
    bond_id_i: jax.Array,
    index_j: jax.Array,
    interact_same_bond_id: jax.Array | bool = False,
) -> jax.Array:
    """Pair mask shared by all colliders.

    The mask always disables interactions between particles in the same clump.
    It also disables interactions between bonded particles unless
    ``interact_same_bond_id`` is ``True`` (see
    :attr:`jaxdem.System.interact_same_bond_id`).
    """
    is_bonded = jnp.any(bond_id_i == index_j[..., None], axis=-1)
    mask1 = (clump_i != clump_j).astype(int)
    mask2 = (~is_bonded | interact_same_bond_id).astype(int)
    return mask1 * mask2


def invalidate_collider(collider: Collider) -> Collider:
    """Explicitly invalidate cached topology, radii, or search geometry."""
    return collider.invalidate()


def refresh_collider(
    state: State,
    collider: Collider,
    force_model: Any | None = None,
    *,
    reset_history: bool = False,
) -> Collider:
    """Rebuild a stateful collider for a (possibly resized) state.

    Stateless colliders (``naive``) have no state-size-dependent buffers, so
    this function returns them unchanged. For stateful colliders
    (``CellList``, ``MultiCellList``, ``NeighborList``), it reads the
    ``Create`` signature. It forwards every parameter whose name matches a
    dataclass field on the current collider instance, plus the new ``state``.
    Parameters not stored on the collider (e.g. ``number_density`` and
    ``safety_factor`` on :class:`NeighborList`) use the ``Create`` defaults.

    Use this after editing a state in ways the collider caches cannot track
    (changing the particle count, teleporting particles, rescaling the box).
    Supply ``force_model`` when growing nonempty history so new slots use its
    initializer. Changing particle count with nonempty history, shrinking its
    capacity, or changing particle identities requires ``reset_history=True``.

    Example
    -------
    >>> system.collider = jdem.colliders.refresh_collider(state, system.collider)
    """
    from inspect import signature

    if not collider.stateful:
        return collider

    create_fn = getattr(type(collider), "Create", None)
    if create_fn is None:
        return collider

    def _stored_search_range(c: Any) -> int | None:
        if not hasattr(c, "neighbor_mask"):
            return None
        return int(jnp.max(jnp.abs(c.neighbor_mask)))

    def _stored_create_kwargs(c: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if c.stateful:
            kwargs["state"] = state
        if hasattr(c, "cell_size"):
            kwargs["cell_size"] = c.cell_size
        search_range = _stored_search_range(c)
        if search_range is not None:
            kwargs["search_range"] = search_range
        return kwargs

    if isinstance(collider, NeighborList):
        secondary_collider = collider.secondary_collider
        new_collider = type(collider).Create(
            state=state,
            cutoff=collider.cutoff,
            skin=collider.skin,
            max_neighbors=collider.max_neighbors,
            secondary_collider_type=secondary_collider.type_name,
            secondary_collider_kw=_stored_create_kwargs(secondary_collider),
        )
        old_n = int(collider.neighbor_list.shape[0])
        same_indexing = old_n == state.N
        old_history = collider.history
        if not same_indexing and old_history.shape[-1] > 0 and not reset_history:
            raise ValueError(
                "Changing particle count with nonempty pair history requires "
                "reset_history=True because particle identity cannot be inferred."
            )
        if force_model is not None:
            expected_tail = force_model.history_shape(state.pos.shape[-1])
            if (
                old_history.shape[-len(expected_tail) :] != expected_tail
                and not reset_history
            ):
                raise ValueError(
                    "Changing the force-model history shape requires "
                    "reset_history=True."
                )
        if same_indexing and not reset_history:
            old_neighbors = collider.neighbor_list
            old_width = old_neighbors.shape[-1]
            new_width = new_collider.max_neighbors
            if old_history.shape[-1] > 0 and new_width < old_width:
                raise ValueError(
                    "Shrinking history capacity requires reset_history=True; "
                    "discarded pair memory cannot be preserved without a rebuild."
                )
            if new_width > old_width:
                padding = new_width - old_width
                old_neighbors = jnp.pad(
                    old_neighbors, ((0, 0), (0, padding)), constant_values=-1
                )
                if old_history.shape[-1] > 0:
                    if force_model is None:
                        raise ValueError(
                            "force_model is required to initialize expanded "
                            "NeighborList history capacity"
                        )
                    expanded_history = force_model.init_history(
                        (state.N, new_width), state.pos.shape[-1]
                    )
                    old_history = expanded_history.at[:, :old_width].set(old_history)
                else:
                    old_history = jnp.empty(
                        (state.N, new_width, 0), dtype=old_history.dtype
                    )
            else:
                old_neighbors = old_neighbors[:, :new_width]
                old_history = old_history[:, :new_width]
            return cast(
                Collider,
                dataclasses.replace(
                    new_collider,
                    neighbor_list=old_neighbors,
                    old_pos=collider.old_pos,
                    n_build_times=collider.n_build_times,
                    history=old_history,
                    invalidated=jnp.asarray(True),
                ),
            )
        if force_model is None and old_history.shape[-1] > 0:
            raise ValueError(
                "force_model is required to refresh a resized NeighborList with history"
            )
        history = new_collider.history
        if force_model is not None:
            history = force_model.init_history(
                (state.N, new_collider.max_neighbors), state.pos.shape[-1]
            )
        return cast(Collider, dataclasses.replace(new_collider, history=history))

    kwargs: dict[str, Any] = {}
    for pname in signature(create_fn).parameters:
        if pname in ("cls", "self"):
            continue
        if pname == "state":
            kwargs[pname] = state
        elif pname == "search_range":
            search_range = _stored_search_range(collider)
            if search_range is not None:
                kwargs[pname] = search_range
        elif hasattr(collider, pname):
            kwargs[pname] = getattr(collider, pname)
    return cast(Collider, create_fn(**kwargs))


from .cell_list import DynamicCellList
from .multi_cell_list import DynamicMultiCellList
from .naive import NaiveSimulator
from .neighbor_list import NeighborList

__all__ = [
    "Collider",
    "DynamicCellList",
    "DynamicMultiCellList",
    "NaiveSimulator",
    "NeighborList",
    "invalidate_collider",
    "refresh_collider",
    "valid_interaction_mask",
]
