# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Contact forces, stress, coordination, friction, and rattler analysis.

Diagnostics use the configured collider and evaluate the force law without
advancing contact history. Scalar and per-particle summaries traverse pairs
directly. Individual contacts are collected only for analyses that need them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..colliders import valid_interaction_mask
from ._pair_candidates import (
    _check_search,
    _collect_pair_candidates,
    _prepare_system,
    _record_overflow,
    _validate_state,
)
from .linalg import cross

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


class ContactData(NamedTuple):
    """Directed active contacts for one configuration, without padding.

    Attributes
    ----------
    pair_ids : jax.Array
        Sphere indices ``(i, j)``, shape ``(M, 2)``. Each entry describes
        the force and torque on ``i`` from ``j``. Reciprocal interactions
        have entries in both directions.
    forces : jax.Array
        Pair forces, shape ``(M, dim)``.
    torques : jax.Array
        Pair torques about the clump COM of ``i``, shape ``(M, 1 | 3)``.
        Includes the force law's intrinsic torque and the member offset moment.
    displacements : jax.Array
        Domain-aware displacement from sphere ``j`` to sphere ``i``,
        shape ``(M, dim)``.

    Notes
    -----
    An entry is active when its force or torque is nonzero. Skin-only
    candidates and excluded interactions are absent. This is a snapshot;
    reuse it only with the configuration from which it was collected.
    """

    pair_ids: jax.Array
    forces: jax.Array
    torques: jax.Array
    displacements: jax.Array


class GroupContactData(NamedTuple):
    """Sparse directed group interactions, with one row per contacting pair.

    Attributes
    ----------
    group_ids : jax.Array
        Sorted group labels, including groups with no contacts.
    pair_ids : jax.Array
        Original group labels ``(I, J)``, shape ``(K, 2)``.
    forces : jax.Array
        Total force on group ``I`` from group ``J``, shape ``(K, dim)``.
    torques : jax.Array
        Total moment about the source group centroid, shape ``(K, 1 | 3)``.
        Includes intrinsic torques and force lever arms.
    displacements : jax.Array
        Domain-aware displacement from centroid ``J`` to centroid ``I``,
        shape ``(K, dim)``. For clumps these are COM displacements.
    friction : jax.Array
        Magnitude ratio ``|F_t| / |F_n|`` relative to the group-centroid
        axis, shape ``(K,)``. A purely tangential nonzero force has ratio
        infinity. Zero net force has ratio zero. Coincident centroids with
        nonzero net force have ratio NaN because the axis is undefined.
    sphere_counts : jax.Array
        Number of distinct participating spheres on each side, shape ``(K, 2)``.
    contact_counts : jax.Array
        Number of directed constituent contacts contributing to each row.
    """

    group_ids: jax.Array
    pair_ids: jax.Array
    forces: jax.Array
    torques: jax.Array
    displacements: jax.Array
    friction: jax.Array
    sphere_counts: jax.Array
    contact_counts: jax.Array


def _pair_values(
    state: State,
    system: System,
    i: jax.Array,
    j: jax.Array,
    history: jax.Array,
    valid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate one pair block with moments about the source clump COM."""
    pos = state.pos
    i = jnp.broadcast_to(i, j.shape)
    safe_i = jnp.clip(i, 0, state.N - 1)
    safe_j = jnp.clip(j, 0, state.N - 1)
    force, torque, _ = jax.vmap(
        lambda a, b, h: system.force_model.force(
            a, b, pos, state, system, h, advance_history=False
        )
    )(safe_i, safe_j, history)
    force = jnp.where(valid[:, None], force, 0.0)
    torque = jnp.where(valid[:, None], torque, 0.0)
    torque = torque + cross(state._pos_p_rot[safe_i], force)
    displacement = system.domain.displacement(pos[safe_i], pos[safe_j], system)
    return force, torque, jnp.where(valid[:, None], displacement, 0.0)


def _summary_values(
    state: State,
    system: System,
    i: jax.Array,
    j: jax.Array,
    history: jax.Array,
    valid: jax.Array,
    quantity: str,
) -> jax.Array:
    force, _, displacement = _pair_values(state, system, i, j, history, valid)
    if quantity == "count":
        return (jnp.any(force != 0, axis=-1) & valid).astype(int)
    virial = displacement[:, :, None] * force[:, None, :]
    return jnp.where((valid & (i < j))[:, None, None], virial, 0.0)


def _cell_summary(
    acc: jax.Array,
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: State,
    valid: jax.Array,
    *,
    system: System,
    quantity: str,
) -> jax.Array:
    history = system.force_model.init_history(j.shape, state.dim)
    return acc + _summary_values(state, system, i, j, history, valid > 0, quantity)


@partial(jax.jit, static_argnames=("quantity",))
def _reduce_contacts(
    state: State, system: System, quantity: str
) -> tuple[System, jax.Array]:
    """Reduce the collider's native traversal without collecting a pair array."""
    from ..colliders import (
        Collider,
        DynamicCellList,
        DynamicMultiCellList,
        NeighborList,
    )
    from ..colliders._neighbor_cache import _row_reduce, _valid_pairs
    from ..colliders.cell_list import _traverse_pairs as cell_traverse
    from ..colliders.multi_cell_list import _traverse_pairs as multi_traverse

    system = _prepare_system(state, system)
    collider = system.collider
    initial = (
        jnp.asarray(0, dtype=int)
        if quantity == "count"
        else jnp.zeros((state.dim, state.dim), dtype=state.pos_c.dtype)
    )
    if not state.N or type(collider) is Collider:
        rows = jnp.zeros((state.N, *initial.shape), dtype=initial.dtype)
        overflow = jnp.asarray(False)
    elif isinstance(collider, NeighborList):

        def evaluate(i: jax.Array, neighbors: jax.Array, slots: jax.Array) -> jax.Array:
            j, valid = _valid_pairs(state, system, i, neighbors)
            return _summary_values(
                state, system, i, j, collider.history[slots], valid, quantity
            )

        rows = _row_reduce(collider, initial, evaluate)
        overflow = collider.overflow
    elif isinstance(collider, (DynamicCellList, DynamicMultiCellList)):
        reach = jnp.max(system.force_model.search_radii(state, system), initial=0.0)
        search_range = jnp.maximum(jnp.max(jnp.abs(collider.neighbor_mask)), 1)
        cell_size = jnp.maximum(collider.cell_size, 2 * reach / search_range)
        traverse = (
            multi_traverse
            if isinstance(collider, DynamicMultiCellList)
            else cell_traverse
        )
        rows, overflow = traverse(
            state,
            system,
            cell_size,
            collider.neighbor_mask,
            partial(_cell_summary, system=system, quantity=quantity),
            initial,
        )
    else:
        neighbors = jnp.arange(state.N)
        history = system.force_model.init_history(neighbors.shape, state.dim)

        def row(i: jax.Array) -> jax.Array:
            valid = valid_interaction_mask(
                state.clump_id[i],
                state.clump_id,
                state.bond_id[i],
                neighbors,
                system.interact_same_bond_id,
            ).astype(bool)
            return _summary_values(
                state, system, i, neighbors, history, valid, quantity
            ).sum(axis=0)

        rows = jax.lax.map(row, neighbors, batch_size=min(state.N, 32))
        overflow = jnp.asarray(False)
    rows = jnp.where(overflow, -1 if quantity == "count" else jnp.nan, rows)
    return _record_overflow(system, overflow), rows


@jax.jit
def _evaluate_contacts(
    state: State,
    system: System,
    pairs: jax.Array,
    history: jax.Array,
    valid: jax.Array,
) -> ContactData:
    if not pairs.shape[0]:
        return _empty_contacts(state)

    def evaluate(args: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, ...]:
        pair, pair_history, pair_valid = args
        f, t, dr = _pair_values(
            state, system, pair[:1], pair[1:], pair_history[None], pair_valid[None]
        )
        return f[0], t[0], dr[0]

    force, torque, displacement = jax.lax.map(
        evaluate, (pairs, history, valid), batch_size=min(pairs.shape[0], 4096)
    )
    return ContactData(pairs, force, torque, displacement)


def _empty_contacts(state: State) -> ContactData:
    return ContactData(
        jnp.empty((0, 2), dtype=int),
        jnp.empty((0, state.dim), dtype=state.pos_c.dtype),
        jnp.empty((0, state.ang_vel.shape[-1]), dtype=state.pos_c.dtype),
        jnp.empty((0, state.dim), dtype=state.pos_c.dtype),
    )


def get_contacts(state: State, system: System) -> tuple[State, System, ContactData]:
    """Collect active directed contacts from the configured collider.

    NeighborList uses its current cache, rebuilding when invalid. Cell and
    naive colliders use an automatically sized packed output buffer. No dense
    per-particle neighbor array is constructed and no query capacity is needed.
    The input state and contact history are not advanced. Use the returned
    system to retain any refreshed cache.

    This host-side operation returns variable-length arrays. Reuse the returned
    ContactData through the ``contacts`` keyword of other diagnostics while
    the configuration is unchanged. Pressure and count reductions can instead
    traverse directly without collecting contacts.

    Raises
    ------
    ValueError
        Search results are incomplete or pair forces/torques are nonfinite.
    """
    system, pairs, valid, history = _collect_pair_candidates(state, system)
    data = _evaluate_contacts(state, system, pairs, history, valid)
    active = valid & (
        jnp.any(data.forces != 0, axis=1) | jnp.any(data.torques != 0, axis=1)
    )
    data = jax.tree.map(lambda x: x[active], data)
    if not bool(
        jnp.all(jnp.isfinite(data.forces)) & jnp.all(jnp.isfinite(data.torques))
    ):
        raise ValueError("Contact forces and torques must be finite.")
    order = jnp.lexsort((data.pair_ids[:, 1], data.pair_ids[:, 0]))
    return state, system, jax.tree.map(lambda x: x[order], data)


def _contacts_or_collect(
    state: State, system: System, contacts: ContactData | None
) -> tuple[State, System, ContactData]:
    _validate_state(state)
    return (
        get_contacts(state, system) if contacts is None else (state, system, contacts)
    )


def compute_contact_stress_tensor(
    state: State,
    system: System,
    *,
    volume: float | jax.Array | None = None,
    contacts: ContactData | None = None,
) -> tuple[State, System, jax.Array]:
    r"""Return the sphere-contact virial stress ``sum(i < j, rij outer Fij) / V``.

    ``rij`` points from sphere ``j`` to ``i`` using the domain's image rule;
    ``Fij`` acts on ``i``. Compression has positive diagonal stress. Clump
    contributions use the constituent sphere displacements. Volume defaults
    to the product of the current domain's box lengths.

    This reduction is JIT-compatible and does not collect a contact list.
    An optional ContactData snapshot avoids reevaluating the force law.
    History and particle dynamics are unchanged. An incomplete search raises
    on the host; compiled calls return NaN and set the returned system's
    search-overflow flag.
    """
    _validate_state(state)
    if contacts is None:
        system, rows = _reduce_contacts(state, system, "stress")
        _check_search(system)
        virial = rows.sum(axis=0)
    else:
        i, j = contacts.pair_ids.T
        virial = jnp.einsum("ni,nj->nij", contacts.displacements, contacts.forces)
        virial = jnp.where((i < j)[:, None, None], virial, 0.0).sum(axis=0)
    if volume is None:
        volume = jnp.prod(system.domain.box_size)
    return state, system, virial / volume


def compute_contact_pressure(
    state: State,
    system: System,
    *,
    volume: float | jax.Array | None = None,
    contacts: ContactData | None = None,
) -> tuple[State, System, jax.Array]:
    """Return contact pressure, ``trace(stress) / dim``, positive in compression.

    Search settings and history come from the system. See
    :func:`compute_contact_stress_tensor` for normalization and overflow rules.
    """
    state, system, stress = compute_contact_stress_tensor(
        state, system, volume=volume, contacts=contacts
    )
    return state, system, jnp.trace(stress) / state.dim


def count_sphere_contacts(
    state: State, system: System, *, contacts: ContactData | None = None
) -> tuple[State, System, jax.Array]:
    """Return force-bearing contact counts per constituent sphere, shape ``(N,)``.

    The direct reduction is JIT-compatible and allocates no contact list.
    Incomplete searches raise on the host; compiled calls return -1 counts
    and set the returned system's search-overflow flag.
    """
    _validate_state(state)
    if contacts is None:
        system, counts = _reduce_contacts(state, system, "count")
        _check_search(system)
    else:
        counts = (
            jnp.zeros(state.N, dtype=int)
            .at[contacts.pair_ids[:, 0]]
            .add(jnp.any(contacts.forces != 0, axis=1).astype(int))
        )
    return state, system, counts


def _clump_count(state: State) -> int:
    return int(jnp.max(state.clump_id, initial=-1)) + 1


def count_vertex_contacts(
    state: State, system: System, *, contacts: ContactData | None = None
) -> tuple[State, System, jax.Array]:
    """Count force-bearing sphere contacts per clump, indexed by clump ID.

    Each reciprocal physical contact increments both participating clumps.
    Different constituent contacts between the same clumps count separately.
    """
    state, system, counts = count_sphere_contacts(state, system, contacts=contacts)
    result = jax.ops.segment_sum(
        counts, state.clump_id, num_segments=_clump_count(state)
    )
    return state, system, result


def _group_labels(state: State, group_by: str | jax.Array) -> np.ndarray:
    if isinstance(group_by, str) and group_by == "bond_id":
        parent = np.arange(state.N)

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        bonds = np.asarray(state.bond_id)
        for i, row in enumerate(bonds):
            for j in row:
                if j >= 0:
                    if j >= state.N:
                        raise ValueError(
                            "Bond adjacency contains an invalid particle index."
                        )
                    a, b = root(i), root(int(j))
                    parent[max(a, b)] = min(a, b)
        labels = np.array([root(i) for i in range(state.N)], dtype=int)
        return np.unique(labels, return_inverse=True)[1]
    labels = np.asarray(
        getattr(state, group_by) if isinstance(group_by, str) else group_by
    )
    if labels.shape != (state.N,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Group labels must be a one-dimensional integer array of length N."
        )
    if np.any(labels < 0):
        raise ValueError("Group labels must be nonnegative.")
    return labels


@jax.jit
def _pair_friction(forces: jax.Array, displacements: jax.Array) -> jax.Array:
    """Return ``|r cross F| / |r dot F|`` without normalizing the pair axis."""
    normal = jnp.abs(jnp.sum(forces * displacements, axis=-1))
    tangent = jnp.linalg.norm(cross(displacements, forces), axis=-1)
    ratio = jnp.where(normal > 0, tangent / jnp.where(normal > 0, normal, 1), jnp.inf)
    ratio = jnp.where(jnp.any(displacements != 0, axis=-1), ratio, jnp.nan)
    return jnp.where(jnp.any(forces != 0, axis=-1), ratio, 0.0)


def get_group_contacts(
    state: State,
    system: System,
    *,
    group_by: str | jax.Array = "clump_id",
    contacts: ContactData | None = None,
) -> tuple[State, System, GroupContactData]:
    """Aggregate contact forces, moments, geometry, and sphere participation.

    ``group_by`` is an integer particle-label array or a state attribute name.
    ``clump_id`` groups rigid bodies. ``bond_id`` groups connected components
    of the bond adjacency, treating bonds as undirected. Isolated particles
    each form a component. Group labels need not be consecutive.

    Group contact existence means at least one constituent force or torque is
    nonzero. Net-force cancellation does not erase the interaction or
    its sphere participation counts. Centroids are unwrapped relative to one
    group member before averaging, then compared with the domain's image rule.
    Output storage scales with the groups and contacting pairs.
    """
    state, system, data = _contacts_or_collect(state, system, contacts)
    labels = _group_labels(state, group_by)
    group_ids, members = np.unique(labels, return_inverse=True)
    ng = len(group_ids)
    pairs = np.asarray(data.pair_ids)
    active = np.any(np.asarray(data.forces) != 0, axis=1) | np.any(
        np.asarray(data.torques) != 0, axis=1
    )
    gi, gj = members[pairs[:, 0]], members[pairs[:, 1]]
    mask = active & (gi != gj)
    slots = np.flatnonzero(mask)
    group_pairs, inverse, counts = np.unique(
        np.column_stack((gi[mask], gj[mask])),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    k = group_pairs.shape[0]
    force = jax.ops.segment_sum(
        data.forces[slots], jnp.asarray(inverse), num_segments=k
    )
    sphere_counts = np.zeros((k, 2), dtype=int)
    for side in (0, 1):
        participation = np.unique(np.column_stack((inverse, pairs[mask, side])), axis=0)
        sphere_counts[:, side] = np.bincount(participation[:, 0], minlength=k)

    if ng:
        _, first = np.unique(members, return_index=True)
        anchors = state.pos_c[first]
        member_slots = jnp.asarray(members)
        dr = system.domain.displacement(state.pos_c, anchors[member_slots], system)
        center = (
            anchors
            + jax.ops.segment_sum(dr, member_slots, num_segments=ng)
            / jnp.asarray(np.bincount(members), dtype=state.pos_c.dtype)[:, None]
        )
        displacement = system.domain.displacement(
            center[group_pairs[:, 0]], center[group_pairs[:, 1]], system
        )
        lever = system.domain.displacement(
            state.pos_c[pairs[slots, 0]], center[gi[slots]], system
        )
    else:
        displacement = jnp.empty((0, state.dim), dtype=state.pos_c.dtype)
        lever = jnp.empty((0, state.dim), dtype=state.pos_c.dtype)
    torque = jax.ops.segment_sum(
        data.torques[slots] + cross(lever, data.forces[slots]),
        jnp.asarray(inverse),
        num_segments=k,
    )
    result = GroupContactData(
        jnp.asarray(group_ids),
        jnp.asarray(group_ids[group_pairs]),
        force,
        torque,
        displacement,
        _pair_friction(force, displacement),
        jnp.asarray(sphere_counts),
        jnp.asarray(counts),
    )
    return state, system, result


def count_clump_contacts(
    state: State, system: System, *, contacts: ContactData | None = None
) -> tuple[State, System, jax.Array]:
    """Count distinct force-bearing neighboring clumps, indexed by clump ID.

    A neighboring clump counts once even when several constituent contacts
    exist or their forces cancel. This differs from vertex contact counts.
    """
    state, system, data = _contacts_or_collect(state, system, contacts)
    pair_ids = np.asarray(data.pair_ids)
    labels = np.asarray(state.clump_id)
    force_bearing = np.any(np.asarray(data.forces) != 0, axis=1)
    pairs = labels[pair_ids[force_bearing]]
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    result = np.bincount(pairs[:, 0], minlength=_clump_count(state))
    return state, system, jnp.asarray(result)


def _prune_rattlers(
    group_i: np.ndarray,
    group_j: np.ndarray,
    group_ids: np.ndarray,
    rows: np.ndarray,
    zc: int,
    dof: int,
    check_rank: bool,
    rank_tol: float | None,
) -> tuple[jax.Array, jax.Array]:
    if isinstance(zc, bool) or not isinstance(zc, (int, np.integer)) or zc < 0:
        raise ValueError("zc must be a nonnegative integer.")
    if rank_tol is not None and (not math.isfinite(rank_tol) or rank_tol < 0):
        raise ValueError("contact_rank_tol must be finite and nonnegative.")
    active = np.ones(len(group_ids), dtype=bool)
    order = np.argsort(group_i, kind="stable")
    counts = np.bincount(group_i, minlength=len(group_ids))
    starts = np.concatenate(([0], np.cumsum(counts)))
    while np.any(active):
        keep = active[group_i] & active[group_j]
        counts = np.bincount(group_i[keep], minlength=len(group_ids))
        remove = active & ((counts < zc) | (counts == 0))
        if check_rank:
            for group in np.flatnonzero(active & ~remove):
                indices = order[starts[group] : starts[group + 1]]
                block = rows[indices[keep[indices]]]
                if np.linalg.matrix_rank(block, tol=rank_tol) < dof:
                    remove[group] = True
        if not np.any(remove):
            break
        active[remove] = False
    if group_ids.size and not np.any(active):
        warnings.warn("No valid particles remain after rattler pruning.", stacklevel=3)
    return jnp.asarray(group_ids[~active]), jnp.asarray(group_ids[active])


def get_clump_rattler_ids(
    state: State,
    system: System,
    *,
    zc: int | None = None,
    check_contact_rank: bool = False,
    contact_rank_tol: float | None = None,
    contacts: ContactData | None = None,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Return rattler and non-rattler clump IDs after iterative contact pruning.

    Clumps with fewer than ``zc`` active constituent contacts are removed,
    followed by clumps disconnected or under-coordinated by those removals.
    ``zc`` defaults to ``dim + angular_dof + 1``. The optional rank criterion
    requires contact force/torque rows to span ``dim + angular_dof`` dimensions.
    Torques include the force law's intrinsic moment and the clump lever arm.
    ``contact_rank_tol`` is the absolute singular-value threshold for that check.

    This host-side analysis does not remove particles or alter contact history.
    Use :func:`remove_rattlers` to construct a reduced state afterward.
    """
    state, system, data = _contacts_or_collect(state, system, contacts)
    group_ids, groups = np.unique(np.asarray(state.clump_id), return_inverse=True)
    pairs = np.asarray(data.pair_ids)
    force, torque = np.asarray(data.forces), np.asarray(data.torques)
    scale = np.linalg.norm(force, axis=1)
    scale = np.where(scale > 0, scale, np.linalg.norm(torque, axis=1))
    rows = (
        np.concatenate((force, torque), axis=1) / np.where(scale > 0, scale, 1)[:, None]
    )
    dof = state.dim + state.ang_vel.shape[-1]
    rattlers, non_rattlers = _prune_rattlers(
        groups[pairs[:, 0]],
        groups[pairs[:, 1]],
        group_ids,
        rows,
        dof + 1 if zc is None else zc,
        dof,
        check_contact_rank,
        contact_rank_tol,
    )
    return state, system, rattlers, non_rattlers


def get_sphere_rattler_ids(
    state: State,
    system: System,
    *,
    zc: int | None = None,
    check_contact_rank: bool = False,
    contact_rank_tol: float | None = None,
    contacts: ContactData | None = None,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Return rattler and non-rattler sphere indices after iterative pruning.

    ``zc`` defaults to ``dim + 1`` force-bearing contacts. The optional rank
    check requires the normalized force rows to span ``dim`` dimensions;
    ``contact_rank_tol`` specifies its absolute singular-value threshold.
    This is a translational sphere criterion. Clump force/torque analysis is
    provided by :func:`get_clump_rattler_ids`.
    """
    state, system, data = _contacts_or_collect(state, system, contacts)
    force = np.asarray(data.forces)
    scale = np.linalg.norm(force, axis=1)
    active = scale > 0
    pairs = np.asarray(data.pair_ids)[active]
    rows = force[active] / scale[active, None]
    rattlers, non_rattlers = _prune_rattlers(
        pairs[:, 0],
        pairs[:, 1],
        np.arange(state.N),
        rows,
        state.dim + 1 if zc is None else zc,
        state.dim,
        check_contact_rank,
        contact_rank_tol,
    )
    return state, system, rattlers, non_rattlers


def remove_rattlers(
    state: State, system: System, rattler_clump_ids: jax.Array
) -> tuple[State, System]:
    """Remove all spheres belonging to rattler clumps and rebuild a matching system.

    The function drops the rattler spheres from the state and re-indexes
    its ``clump_id`` and ``bond_id`` arrays. The returned system is a
    :func:`dataclasses.replace` copy of the input, so it preserves every
    field (``domain``, ``mat_table``, integrators, user hooks, ``dt``,
    ``time``, and any future additions to :class:`System`) by default.
    Only the state-size-dependent fields change:

    * The function rebuilds ``collider`` via its :meth:`Create` method for
      stateful colliders (``NeighborList``, cell lists) and passes it
      through unchanged for stateless ones (``naive``). It recovers
      Create's config kwargs from the current collider via introspection
      (see :func:`jaxdem.colliders.refresh_collider`).
      ``NeighborList`` transfers the complete history of each surviving
      directed pair through the particle index mapping, including tangential
      displacement and previous contact normals. New pairs use the force
      law's history initializer. Force evaluation does not advance history.
    * The function rebuilds ``force_manager`` so that its per-particle
      buffers (``external_force``, ``external_force_com``,
      ``external_torque``) are sized for the reduced state. ``gravity``,
      ``force_functions``, ``energy_functions``, and ``is_com_force``
      stay unchanged.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        Current system. The rebuilt system carries over all fields except
        the state-size-dependent ones listed above.
    rattler_clump_ids : jax.Array
        1-D array of clump IDs to remove.

    Returns
    -------
    state : State
        New state with rattler spheres removed and IDs re-indexed.
    system : System
        New system with matching state shape.

    Notes
    -----
    **DP / bonded force models.** When ``system.bonded_force_model`` is a
    :class:`DeformableParticleModel`, its topology arrays (``elements``,
    ``edges``, ``element_adjacency``, …) reference vertices by
    particle array. :func:`remove_rattlers` re-indexes ``bond_id`` in
    the state but does **not** remap the bonded-model topology, because
    the correct behavior is ambiguous (an element that partially
    straddles removed vertices could be dropped, re-triangulated, or
    flagged). The function emits a warning when a bonded model is
    present. Users with DP systems should handle the topology remap
    manually.

    **Custom collider settings.** Any collider Create-kwarg whose name
    is not a field on the current collider (e.g. ``number_density`` and
    ``safety_factor`` on :class:`NeighborList`) gets Create's default
    value. Preserving those settings requires explicitly supplying them
    when reconstructing the collider.
    """
    from ..colliders import refresh_collider
    from ..forces.force_manager import ForceManager

    # 1. State update.
    keep = ~jnp.isin(state.clump_id, rattler_clump_ids)
    idx = jnp.where(keep)[0]
    new_state = jax.tree.map(lambda x: x[idx], state)
    N_new = idx.shape[0]
    _, new_clump_id = jnp.unique(new_state.clump_id, return_inverse=True, size=N_new)
    new_state.clump_id = new_clump_id

    # ``bond_id`` is an adjacency list of neighbor *indices* padded with
    # -1, not dense labels, so it must be remapped through an old-idx ->
    # new-idx table (padding and removed neighbors stay -1).
    new_idx = jnp.arange(N_new, dtype=int)
    idx_remap = jnp.full((state.N,), -1, dtype=int)
    idx_remap = idx_remap.at[idx].set(new_idx)
    old_bond_id = new_state.bond_id
    keep_bond = old_bond_id >= 0
    new_state.bond_id = jnp.where(
        keep_bond, idx_remap[jnp.where(keep_bond, old_bond_id, 0)], -1
    )

    # 2. Rebuild the collider (if stateful).
    new_collider = refresh_collider(
        new_state, system.collider, system.force_model, reset_history=True
    )

    # 3. Rebuild the force manager. Its force_functions / energy_functions /
    # is_com_force static tuples — including any bonded-model
    # force_and_energy_fns appended by the original System.create — are
    # preserved, while the per-particle external_force / external_torque
    # buffers are resized to the new state.
    fm = system.force_manager
    fm_entries = [
        (fm.force_functions[i], fm.energy_functions[i], fm.is_com_force[i])
        for i in range(len(fm.force_functions))
    ]
    new_force_manager = ForceManager.create(
        state_shape=new_state.shape,
        gravity=fm.gravity,
        force_functions=fm_entries,
    )

    # 4. Warn about DP/bonded-model topology going stale.
    if system.bonded_force_model is not None:
        warnings.warn(
            "remove_rattlers does not remap bonded_force_model topology. "
            "If removed vertices are referenced by the bonded model's "
            "elements / edges / adjacencies, the returned system will be "
            "inconsistent; remap the topology manually.",
            stacklevel=2,
        )

    # 5. Use dataclasses.replace so every other System field (including
    # any added later) is preserved automatically.
    new_system = replace(
        system,
        collider=new_collider,
        force_manager=new_force_manager,
    )

    # Refresh forces after reindexing without repeating integrator setup.
    from ..colliders import NeighborList

    if isinstance(new_system.collider, NeighborList):
        from ..colliders._neighbor_cache import reindex_history

        new_system = reindex_history(new_state, new_system, system.collider, idx)
        new_state, new_system = new_system.collider.compute_force(
            new_state, new_system, advance_history=False
        )
    else:
        new_state, new_system = new_system.collider.compute_force(new_state, new_system)
    new_state, new_system = new_system.force_manager.apply(new_state, new_system)
    new_system = replace(
        new_system,
        search_overflow=new_system.search_overflow | new_system.collider.overflow,
    )

    return new_state, new_system


__all__ = [
    "ContactData",
    "GroupContactData",
    "compute_contact_pressure",
    "compute_contact_stress_tensor",
    "count_clump_contacts",
    "count_sphere_contacts",
    "count_vertex_contacts",
    "get_clump_rattler_ids",
    "get_contacts",
    "get_group_contacts",
    "get_sphere_rattler_ids",
    "remove_rattlers",
]
