# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Collision-detection interfaces and implementations."""

from __future__ import annotations

from abc import ABC
import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

import jax
import jax.numpy as jnp

from ..domains import Domain, SearchGeometry
from ..factory import Factory
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

    def history_pair_shape(
        self, state_shape: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        """Leading shape of the stored pair history for ``state_shape``.

        ``None`` means this collider has no persistent pair memory, so it can
        only run stateless force laws.
        """
        return None

    def refresh(
        self,
        state: State,
        force_model: Any | None = None,
        *,
        reset_history: bool = False,
    ) -> Collider:
        """Rebuild this collider for a (possibly resized) ``state``.

        A stateless collider returns itself. A collider with search caches
        is rebuilt through its ``Create`` signature: every parameter whose
        name matches a dataclass field on this instance is forwarded, plus
        the new ``state``. Colliders with pair history override this to
        carry their history across, using ``force_model`` to initialize new
        slots; ``reset_history`` discards it instead.
        """
        from inspect import signature

        create_fn = getattr(type(self), "Create", None)
        if not self.stateful or create_fn is None:
            return self
        kwargs: dict[str, Any] = {}
        for pname in signature(create_fn).parameters:
            if pname == "state":
                kwargs[pname] = state
            elif pname == "search_range" and hasattr(self, "neighbor_mask"):
                kwargs[pname] = int(jnp.max(jnp.abs(self.neighbor_mask)))
            elif pname not in ("cls", "self") and hasattr(self, pname):
                kwargs[pname] = getattr(self, pname)
        return cast(Collider, create_fn(**kwargs))

    def reindex_history(
        self, state: State, system: System, new_to_old: jax.Array
    ) -> System:
        """Transfer this collider's stored pair history onto a reindexed state.

        ``system.collider`` must already be allocated for ``state`` and
        ``new_to_old[k]`` is the original index of particle ``k``. Pairs that
        survive keep their history; new pairs and padding start from the
        force law's initial history. Colliders without pair memory return
        ``system`` unchanged.
        """
        return system

    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
    def compute_force(
        state: State,
        system: System,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[State, System]:
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
        advance_history : bool, optional
            Advance stored pair history over this step. Colliders without
            pair memory ignore it.
        quasistatic : bool, optional
            Forwarded to the force law. True omits velocity-dependent terms.

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
        return system.collider.compute_force(state, system, advance_history=False)

    @staticmethod
    def get_history(
        state: State, system: System, neighbor_list: jax.Array
    ) -> jax.Array:
        """Return stored history for an explicit ``(N, k)`` pair query.

        Colliders without pair memory return the force law's initial history.
        """
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
    """Rebuild a collider for a (possibly resized) state.

    Use this after editing a state in ways the collider caches cannot track
    (changing the particle count, teleporting particles, rescaling the box).
    Supply ``force_model`` when growing nonempty history so new slots use its
    initializer. Changing particle count with nonempty history, shrinking its
    capacity, or changing particle identities requires ``reset_history=True``.
    See :meth:`Collider.refresh`.

    Example
    -------
    >>> system.collider = jdem.colliders.refresh_collider(state, system.collider)
    """
    return collider.refresh(state, force_model, reset_history=reset_history)


def advance_history(
    state_old: State, system_old: System, state_new: State, system_new: System
) -> tuple[State, System]:
    """Advance pair history for a change of configuration without time passing.

    The motion from ``(state_old, system_old)`` to ``(state_new, system_new)``
    is expressed as velocities over ``system_new.dt``: body translation and
    rotation increments divided by ``dt`` for the particles, and the domain's
    image rates for moving periodic images. The force model is evaluated once
    at the new configuration with ``advance_history=True`` and
    ``quasistatic=True``, so ``dt`` cancels exactly and no velocity-
    dependent force acts.

    Used by affine box deformation (:meth:`Domain.shear`,
    :meth:`Domain.resize`) and by relaxation. Dynamics does not call it: the
    integrator's own velocities advance history inside ``compute_force``.

    This relies on the ``quasistatic`` contract of :meth:`ForceModel.force`:
    the law produces no velocity-dependent force, and its history update
    depends on velocity only through ``velocity * dt``. A law that breaks
    either rule gives a ``dt``-dependent result here.

    Returns
    -------
    state : State
        ``state_new`` with the rate-independent forces and torques of the new
        configuration.
    system : System
        ``system_new`` with the advanced collider history and updated overflow
        flag. The domain and ``dt`` are those of ``system_new``.

    This is a trace-time no-op for stateless force laws.
    """
    from ..utils.quaternion import Quaternion

    if system_new.force_model.history_shape(state_new.dim) == (0,):
        return state_new, system_new
    dt = system_new.dt
    vel = (state_new.pos_c - state_old.pos_c) / dt
    rotvec = Quaternion.to_rotvec(state_new.q @ Quaternion.conj(state_old.q))
    if state_new.dim == 2:
        rotvec = rotvec[..., 2:]
    moving_state = dataclasses.replace(state_new, vel=vel, ang_vel=rotvec / dt)
    moving_system = dataclasses.replace(
        system_new, domain=system_new.domain.rates_between(system_old.domain, dt)
    )
    moving_state, advanced = moving_system.collider.compute_force(
        moving_state, moving_system, advance_history=True, quasistatic=True
    )
    state = dataclasses.replace(
        state_new, force=moving_state.force, torque=moving_state.torque
    )
    system = dataclasses.replace(
        system_new,
        collider=advanced.collider,
        search_overflow=system_new.search_overflow | advanced.collider.overflow,
    )
    return state, system


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
    "advance_history",
    "invalidate_collider",
    "refresh_collider",
    "valid_interaction_mask",
]
