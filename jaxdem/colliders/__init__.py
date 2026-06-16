# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Collision-detection interfaces and implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

from ..factory import Factory
from ..utils.linalg import norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Collider(Factory, ABC):
    r"""The base interface for defining how contact detection and force computations are performed in a simulation.

    Concrete subclasses of `Collider` implement the specific algorithms for calculating the interactions.

    Notes:
    ------
    Self-interaction (i.e., calling the force/energy computation for `i=j`) is allowed,
    and the underlying `force_model` is responsible for correctly handling or
    ignoring this case.

    Example:
    --------
    To define a custom collider, inherit from `Collider`, register it and implement its abstract methods:

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
    """Boolean flag indicating if a collider overflow occurred."""

    @staticmethod
    @jax.jit(inline=True)
    def compute_force(state: State, system: System) -> tuple[State, System]:
        """Compute the total force acting on each particle in the simulation.

        This base implementation is a concrete no-op: it zeroes the ``force``
        and ``torque`` attributes of the ``state`` and returns. It backs the
        ``""`` (empty-string) no-op collider registration for systems whose
        dynamics come exclusively from bonded forces or user force functions.

        Subclasses override it to calculate inter-particle forces and torques
        based on the current `state` and `system` configuration, then update
        the `force` and `torque` attributes of the `state` object with the
        resulting total force and torque for each particle.

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
    def compute_potential_energy(
        state: State, system: System
    ) -> tuple[State, System, jax.Array]:
        """Compute the total (scalar) non-bonded potential energy of the system.

        Implementations sum every pair-interaction contribution defined by
        ``system.force_model`` and return a single scalar. Pair energies are
        accumulated with the standard 0.5 factor so each pair counts once
        even when the underlying neighbor list visits ``(i, j)`` and
        ``(j, i)`` separately.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System, jax.Array]
            A tuple of (state, system, potential_energy) where potential_energy is a
            scalar JAX array (shape ``()``) — the total non-bonded potential energy of the system.

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

        This is primarily used by neighbor-list-based algorithms and diagnostics.
        Implementations should match the cell-list semantics:

        - Returns a neighbor list of shape ``(N, max_neighbors)`` padded with ``-1``.
        - Neighbor indices must refer to the returned (possibly sorted) ``state``.
        - Also returns an ``overflow`` boolean flag (True if any particle exceeded
          ``max_neighbors`` neighbors within the cutoff).
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

        For each point in ``pos_a``, finds all neighbors from ``pos_b``
        within the given ``cutoff`` distance. This is useful for coupling
        different particle systems or computing interactions between
        distinct sets of objects.

        The default implementation uses a naive :math:`O(N_A \times N_B)`
        all-pairs search. Subclasses may override this with more efficient
        algorithms.

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
            - ``overflow``: Boolean flag indicating if any query point exceeded
              ``max_neighbors`` neighbors within the cutoff.

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
    unique_id_j: jax.Array,
    interact_same_bond_id: jax.Array | bool = False,
) -> jax.Array:
    """Pair mask shared by all colliders.

    Interactions are always disabled for particles in the same clump.
    Interactions for particles connected by a bond are disabled unless
    ``interact_same_bond_id`` is ``True`` (see
    :attr:`jaxdem.System.interact_same_bond_id`).
    """
    is_bonded = jnp.any(bond_id_i == unique_id_j[..., None], axis=-1)
    mask1 = (clump_i != clump_j).astype(int)
    mask2 = (~is_bonded | interact_same_bond_id).astype(int)
    return mask1 * mask2


def refresh_collider(state: State, collider: Collider) -> Collider:
    """Rebuild a stateful collider for a (possibly resized) state.

    Stateless colliders (``naive``) have no state-size-dependent buffers and
    are returned unchanged. Stateful colliders (``CellList``,
    ``MultiCellList``, ``NeighborList``) are rebuilt by introspecting their
    ``Create`` signature and forwarding any parameter whose name is also a
    dataclass field on the current collider instance (plus the new
    ``state``). Parameters not stored on the collider (e.g.
    ``number_density`` and ``safety_factor`` on :class:`NeighborList`) fall
    back to ``Create``'s own defaults.

    Use this after editing a state in ways the collider caches cannot track
    (changing the particle count, teleporting particles, rescaling the box).

    Example
    -------
    >>> system.collider = jdem.colliders.refresh_collider(state, system.collider)
    """
    from inspect import signature

    stateful = {"neighborlist", "celllist", "multicelllist"}
    if collider.type_name.lower() not in stateful:
        return collider

    create_fn = getattr(type(collider), "Create", None)
    if create_fn is None:
        return collider

    def _stored_search_range(c: Any) -> int | None:
        if not hasattr(c, "neighbor_mask"):
            return None
        return int(jnp.max(jnp.abs(c.neighbor_mask)))

    def _stored_create_kwargs(c: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"state": state}
        if hasattr(c, "cell_size"):
            kwargs["cell_size"] = c.cell_size
        search_range = _stored_search_range(c)
        if search_range is not None:
            kwargs["search_range"] = search_range
        return kwargs

    if collider.type_name.lower() == "neighborlist":
        secondary_collider = collider.secondary_collider  # type: ignore[attr-defined]
        new_collider = cast(
            Collider,
            type(collider).Create(  # type: ignore[attr-defined]
                state=state,
                cutoff=collider.cutoff,  # type: ignore[attr-defined]
                skin=collider.skin,  # type: ignore[attr-defined]
                max_neighbors=collider.max_neighbors,  # type: ignore[attr-defined]
                secondary_collider_type=secondary_collider.type_name,
                secondary_collider_kw=_stored_create_kwargs(secondary_collider),
            ),
        )
        if getattr(collider, "history", None) is not None:
            # We don't have access to ForceModel to initialize properly here!
            # Wait, history is just a PyTree of arrays.
            # We can't easily recreate history without the force model!
            pass
        return new_collider

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
    "refresh_collider",
    "valid_interaction_mask",
]
