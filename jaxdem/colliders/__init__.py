# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Collision-detection interfaces and implementations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ..factory import Factory
from ..utils.linalg import norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Collider(Factory, ABC):
    r"""
    The base interface for defining how contact detection and force computations are performed in a simulation.

    Concrete subclasses of `Collider` implement the specific algorithms for calculating the interactions.

    Notes
    -----
    Self-interaction (i.e., calling the force/energy computation for `i=j`) is allowed,
    and the underlying `force_model` is responsible for correctly handling or
    ignoring this case.

    Example
    -------
    To define a custom collider, inherit from `Collider`, register it and implement its abstract methods:

    >>> @Collider.register("CustomCollider")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class CustomCollider(Collider):
            ...

    Then, instantiate it:

    >>> jaxdem.Collider.create("CustomCollider", **custom_collider_kw)
    """

    @staticmethod
    @jax.jit(donate_argnames=("state", "system"), inline=True)
    def compute_force(state: State, system: System) -> Tuple[State, System]:
        """
        Abstract method to compute the total force acting on each particle in the simulation.

        Implementations should calculate inter-particle forces and torques based on the current
        `state` and `system` configuration, then update the `force` and `torque` attributes of the
        `state` object with the resulting total force and torque for each particle.

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

        Note
        -----
        - This method donates state and system
        """
        state.force *= 0
        state.torque *= 0
        return state, system

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        """
        Abstract method to compute the total potential energy of the system.

        Implementations should calculate the sum per particle of all potential energies
        present in the system based on the current `state` and `system` configuration.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            A scalar JAX array representing the total potential energy of each particle.

        Example
        -------

        >>> potential_energy = system.collider.compute_potential_energy(state, system)
        >>> print(f"Potential energy per particle: {potential_energy:.4f}")
        >>> print(potential_energy.shape)  # (N,)
        """
        return jnp.zeros_like(state.mass)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple[State, System, jax.Array, jax.Array]:
        """
        Build a neighbor list for the current collider.

        This is primarily used by neighbor-list-based algorithms and diagnostics.
        Implementations should match the cell-list semantics:

        - Returns a neighbor list of shape ``(N, max_neighbors)`` padded with ``-1``.
        - Neighbor indices must refer to the returned (possibly sorted) ``state``.
        - Also returns an ``overflow`` boolean flag (True if any particle exceeded
          ``max_neighbors`` neighbors within the cutoff).
        """
        raise NotImplementedError

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple[jax.Array, jax.Array]:
        r"""
        Build a cross-neighbor list between two sets of positions.

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

        def per_query(pos_ai: jax.Array) -> Tuple[jax.Array, jax.Array]:
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


Collider.register("")(Collider)


@jax.jit(inline=True)
def valid_interaction_mask(
    clump_i: jax.Array,
    clump_j: jax.Array,
    bond_i: jax.Array,
    bond_j: jax.Array,
    interact_same_bond_id: jax.Array,
) -> jax.Array:
    """
    Pair mask shared by all colliders.

    Interactions are always disabled for particles in the same clump.
    Interactions for particles with equal ``bond_id`` are controlled by
    ``interact_same_bond_id``.
    """
    return (clump_i != clump_j) * (interact_same_bond_id | (bond_i != bond_j))


from .naive import NaiveSimulator
from .cell_list import StaticCellList, DynamicCellList
from .neighbor_list import NeighborList

# from .sweep_and_prune import SweepAndPrune

__all__ = [
    "Collider",
    "NaiveSimulator",
    "StaticCellList",
    "DynamicCellList",
    "NeighborList",
    "valid_interaction_mask",
]
