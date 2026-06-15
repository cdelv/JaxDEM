# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions for analyzing particle contacts and identifying rattlers."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from .linalg import norm, unit

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


def get_pair_forces_and_ids(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Compute pairwise contact forces and their associated particle IDs.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition containing the collider and force model.
    cutoff : float, optional
        Neighbor search cutoff distance. Defaults to ``3 * max(rad)``.
    max_neighbors : int, optional
        Maximum number of neighbors per particle (default 100).

    Returns
    -------
    state : State
        Potentially updated state (after neighbor-list rebuild).
    system : System
        Potentially updated system.
    pair_ids : jax.Array
        ``(N * max_neighbors, 2)`` array of ``(i, j)`` sphere index pairs —
        the full neighbor-list grid, *including* the padding rows where
        ``j == -1`` (whose force is zero). Filter on ``pair_ids[:, 1] != -1``
        to keep only real pairs.
    forces : jax.Array
        ``(N * max_neighbors, dim)`` array of pairwise force vectors, one
        per pair (zero for padding rows).

    """
    if cutoff is None:
        cutoff = float(jnp.max(state.rad)) * 3.0
    if max_neighbors is None:
        max_neighbors = 100

    state, system, nl, overflow = system.collider.create_neighbor_list(
        state, system, cutoff, max_neighbors
    )
    if overflow:
        raise ValueError("Neighbor list overflowed. Increase max_neighbors.")

    sphere_ids = jax.lax.iota(dtype=int, size=state.N)
    pos = state.pos

    def per_pair_force(i: jnp.ndarray, neighbors: jnp.ndarray) -> jnp.ndarray:
        def per_neighbor_force(j_id: jnp.ndarray) -> jnp.ndarray:
            valid = j_id != -1
            safe_j = jnp.maximum(j_id, 0)
            f, _ = system.force_model.force(i, safe_j, pos, state, system)
            return f * valid

        return jax.vmap(per_neighbor_force)(neighbors)

    neigh_force = jax.vmap(per_pair_force)(sphere_ids, nl)

    n_neighbors = nl.shape[1]
    i_ids = jnp.repeat(sphere_ids[:, None], n_neighbors, axis=1).ravel()
    j_ids = nl.ravel()
    neigh_force = neigh_force.reshape(-1, state.dim)

    return state, system, jnp.column_stack((i_ids, j_ids)), neigh_force


def compute_contact_stress_tensor(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    volume: float | jax.Array | None = None,
) -> tuple[State, System, jax.Array]:
    r"""Compute the contact virial stress tensor.

    The stress is computed from force-bearing sphere-sphere contacts as

    .. math::
        \sigma = \frac{1}{V}\sum_{i < j} \mathbf{r}_{ij}\otimes\mathbf{F}_{ij},

    where ``F_ij`` is the force on sphere ``i`` from sphere ``j`` and
    ``r_ij`` is the domain-aware displacement from ``j`` to ``i``. This
    convention gives positive diagonal stress, and therefore positive
    pressure, for repulsive compression.

    For rigid clumps, the sum remains over the vertex-sphere contacts
    because those are the force-bearing contacts in the simulation.

    Parameters
    ----------
    state, system, cutoff, max_neighbors
        Same as :func:`get_pair_forces_and_ids`.
    volume : float or jax.Array, optional
        Volume/area used for normalization. Defaults to
        ``prod(system.domain.box_size)``.

    Returns
    -------
    state : State
        Potentially updated state (after neighbor-list rebuild).
    system : System
        Potentially updated system.
    stress : jax.Array
        ``(dim, dim)`` contact stress tensor.
    """
    state, system, pair_ids, forces = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )

    i = pair_ids[:, 0]
    j = pair_ids[:, 1]
    safe_j = jnp.maximum(j, 0)

    pos = state.pos
    rij = system.domain.displacement(pos[i], pos[safe_j], system)

    valid = (j >= 0) & (i < safe_j) & (jnp.sum(forces * forces, axis=-1) > 0)
    virial = jnp.einsum("ni,nj->nij", rij, forces)
    virial = jnp.sum(virial * valid[:, None, None], axis=0)

    if volume is None:
        volume = jnp.prod(system.domain.box_size)

    return state, system, virial / volume


def compute_contact_pressure(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    volume: float | jax.Array | None = None,
) -> tuple[State, System, jax.Array]:
    """Compute scalar contact pressure from the contact stress tensor.

    Pressure is ``trace(stress) / dim`` and is positive for repulsive
    compression under :func:`compute_contact_stress_tensor`'s sign convention.
    """
    state, system, stress = compute_contact_stress_tensor(
        state, system, cutoff, max_neighbors, volume
    )
    return state, system, jnp.trace(stress) / state.dim


def _generalized_contact_force_rows(
    state: State,
    system: System,
    pair_ids: jax.Array,
    forces: jax.Array,
    normalize: bool = True,
) -> jax.Array:
    """Return ``[force, torque]`` contact rows for the first sphere in each pair.

    The torque lever arm runs from the clump center of mass to the *contact
    point* (sphere center plus ``rad * n_hat`` toward the other sphere), so
    tangential (frictional) force components produce the correct torque rows.
    """
    if normalize:
        forces = forces / norm(forces)[:, None]

    i = pair_ids[:, 0]
    j = pair_ids[:, 1]
    pos = state.pos
    # Domain-aware displacement from j to i; the contact point sits at
    # rad_i along the i -> j direction (-rij) from sphere i's center.
    rij = system.domain.displacement(pos[i], pos[j], system)
    n_hat = unit(-rij)
    lever_arms = pos[i] - state.pos_c[i] + state.rad[i][:, None] * n_hat
    torques = jnp.cross(lever_arms, forces)
    if torques.ndim == 1:
        torques = torques[:, None]

    return jnp.concatenate([forces, torques], axis=1)


def _matrix_ranks_by_group(
    rows: jax.Array,
    group_ids: jax.Array,
    n_groups: int,
    tol: float | None = None,
) -> jax.Array:
    """Compute the rank of variable-length row blocks grouped by integer ID."""
    if rows.shape[0] == 0:
        return jnp.zeros(n_groups, dtype=int)

    counts = jnp.bincount(group_ids, length=n_groups)
    max_rows = int(jnp.max(counts))

    order = jnp.argsort(group_ids)
    group_sorted = group_ids[order]
    rows_sorted = rows[order]

    starts = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts[:-1])])
    slot = jnp.arange(rows_sorted.shape[0]) - jnp.repeat(starts, counts)

    padded = jnp.zeros((n_groups, max_rows, rows.shape[1]), dtype=rows.dtype)
    padded = padded.at[group_sorted, slot].set(rows_sorted)

    return jax.vmap(lambda block: jnp.linalg.matrix_rank(block, tol=tol))(padded)


def _iterative_rattler_prune(
    group_i: jax.Array,
    group_j: jax.Array,
    n_groups: int,
    zc: int,
    dof: int,
    rows_fn: Any,
    check_contact_rank: bool,
    contact_rank_tol: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Iteratively remove under-coordinated / under-ranked contact groups.

    ``group_i`` / ``group_j`` are the group ids of the two endpoints of every
    force-bearing contact. ``rows_fn(keep_mask)`` returns the generalized
    force rows of the contacts selected by ``keep_mask`` (used only when
    ``check_contact_rank`` is True). Returns ``(rattler_ids,
    non_rattler_ids)``.
    """
    all_ids = jnp.arange(n_groups)
    rattler_ids = jnp.setdiff1d(
        all_ids, jnp.unique(jnp.concatenate([group_i, group_j]))
    )

    while True:
        remaining = jnp.setdiff1d(all_ids, rattler_ids)
        if len(remaining) == 0:
            break

        keep = jnp.isin(group_i, remaining) & jnp.isin(group_j, remaining)
        active_group_i = group_i[keep]
        groups_in_contacts = jnp.unique(
            jnp.concatenate([active_group_i, group_j[keep]])
        )
        disconnected = jnp.setdiff1d(remaining, groups_in_contacts)
        contact_counts = jnp.bincount(active_group_i, length=n_groups)
        active = jnp.unique(active_group_i)
        under_coordinated = active[contact_counts[active] < zc]
        under_ranked = jnp.asarray([], dtype=active.dtype)
        if check_contact_rank:
            rows = rows_fn(keep)
            contact_ranks = _matrix_ranks_by_group(
                rows, active_group_i, n_groups, contact_rank_tol
            )
            under_ranked = active[contact_ranks[active] < dof]
        new_rattlers = jnp.union1d(
            disconnected,
            jnp.union1d(
                jnp.setdiff1d(under_coordinated, rattler_ids),
                jnp.setdiff1d(under_ranked, rattler_ids),
            ),
        )

        if len(new_rattlers) == 0:
            break

        rattler_ids = jnp.union1d(rattler_ids, new_rattlers)

    non_rattler_ids = jnp.setdiff1d(all_ids, rattler_ids)
    return rattler_ids, non_rattler_ids


def get_clump_rattler_ids(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    zc: int | None = None,
    check_contact_rank: bool = False,
    contact_rank_tol: float | None = None,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Identify rattler clumps by iteratively removing under-coordinated clumps.

    A clump is a rattler if its total vertex-contact count is below the
    coordination threshold *zc*. Optionally, clumps whose contacts do not
    span their rigid-body generalized force space are also treated as
    rattlers.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition.
    cutoff : float, optional
        Neighbor search cutoff distance.
    max_neighbors : int, optional
        Maximum number of neighbors per particle.
    zc : int, optional
        Minimum contact count. Defaults to ``dim + angular_dof + 1`` —
        the mechanical stability threshold for a rigid body. A clump
        with ``dim + angular_dof`` or fewer force-bearing vertex contacts
        is unstable (the tangential softening of each contact at finite
        overlap can give the sub-hessian a negative eigenvalue), so
        ``dim + angular_dof + 1`` non-degenerate contacts are needed for
        a positive-definite local hessian.
    check_contact_rank : bool, optional
        If ``True``, also remove clumps whose active force-bearing contacts
        have generalized force rank below ``dim + angular_dof``.
    contact_rank_tol : float, optional
        Absolute tolerance passed to ``jax.numpy.linalg.matrix_rank`` for
        the optional generalized force rank check.

    Returns
    -------
    state : State
        Potentially updated state.
    system : System
        Potentially updated system.
    rattler_ids : jax.Array
        1-D array of rattler clump IDs.
    non_rattler_ids : jax.Array
        1-D array of non-rattler clump IDs.

    """
    state, system, pair_ids, neigh_force = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )

    force_norm = norm(neigh_force)
    active_force_mask = force_norm > 0
    pair_ids = pair_ids[active_force_mask]
    neigh_force = neigh_force[active_force_mask]

    N_clumps = int(jnp.max(state.clump_id)) + 1

    if zc is None:
        zc = state.ang_vel.shape[-1] + state.dim + 1

    clump_i = state.clump_id[pair_ids[:, 0]]
    clump_j = state.clump_id[pair_ids[:, 1]]

    def rows_fn(keep: jax.Array) -> jax.Array:
        return _generalized_contact_force_rows(
            state, system, pair_ids[keep], neigh_force[keep]
        )

    rattler_ids, non_rattler_ids = _iterative_rattler_prune(
        clump_i,
        clump_j,
        N_clumps,
        zc,
        state.dim + state.ang_vel.shape[-1],
        rows_fn,
        check_contact_rank,
        contact_rank_tol,
    )

    # TODO: add warning here of no remaining particles

    return state, system, rattler_ids, non_rattler_ids


def get_sphere_rattler_ids(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    zc: int | None = None,
    check_contact_rank: bool = False,
    contact_rank_tol: float | None = None,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Identify rattler spheres by iteratively removing under-coordinated particles.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition.
    cutoff : float, optional
        Neighbor search cutoff distance.
    max_neighbors : int, optional
        Maximum number of neighbors per particle.
    zc : int, optional
        Minimum contact count. Defaults to ``dim + 1`` — the mechanical
        stability threshold for a point particle. A sphere with ``dim`` or
        fewer force-bearing contacts is unstable: the tangential softening
        of each contact at finite overlap can give the sub-hessian a
        negative eigenvalue, so ``dim + 1`` non-degenerate contacts are
        needed for a positive-definite local hessian.
    check_contact_rank : bool, optional
        If ``True``, also remove particles whose active force-bearing contacts
        have force-direction rank below ``dim``.
    contact_rank_tol : float, optional
        Absolute tolerance passed to ``jax.numpy.linalg.matrix_rank`` for
        the optional force-direction rank check.

    Returns
    -------
    state : State
        Potentially updated state.
    system : System
        Potentially updated system.
    rattler_ids : jax.Array
        1-D array of rattler sphere indices.
    non_rattler_ids : jax.Array
        1-D array of non-rattler sphere indices.

    """
    state, system, pair_ids, neigh_force = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )

    force_norm = norm(neigh_force)
    active_force_mask = force_norm > 0
    pair_ids = pair_ids[active_force_mask]
    neigh_force = neigh_force[active_force_mask]
    force_norm = force_norm[active_force_mask]

    N = state.N
    if zc is None:
        zc = state.dim + 1

    def rows_fn(keep: jax.Array) -> jax.Array:
        return neigh_force[keep] / force_norm[keep][:, None]

    rattler_ids, non_rattler_ids = _iterative_rattler_prune(
        pair_ids[:, 0],
        pair_ids[:, 1],
        N,
        zc,
        state.dim,
        rows_fn,
        check_contact_rank,
        contact_rank_tol,
    )

    # TODO: add warning here of no remaining particles

    return state, system, rattler_ids, non_rattler_ids


def count_vertex_contacts(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
) -> tuple[State, System, jax.Array]:
    """Count force-bearing vertex-level contacts per clump.

    For each clump, returns the number of sphere-sphere contacts with
    nonzero contact force that involve one of the clump's vertex spheres.
    Each unique physical contact between clumps :math:`I` and :math:`J`
    increments both clump :math:`I`'s and clump :math:`J`'s count by one
    (the neighbor list lists the pair in both directions).

    This is the contact-count quantity entering the Maxwell / isostaticity
    condition: the sum over clumps equals twice the number of distinct
    force-bearing vertex contacts, so the mean over clumps is the average
    coordination number :math:`Z`.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition.
    cutoff : float, optional
        Neighbor search cutoff distance.
    max_neighbors : int, optional
        Maximum number of neighbors per particle.

    Returns
    -------
    state : State
        Potentially updated state.
    system : System
        Potentially updated system.
    contacts : jax.Array
        ``(N_clumps,)`` integer array of force-bearing vertex contacts
        per clump.
    """
    state, system, pair_ids, forces = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )
    force_norm = norm(forces)
    N_clumps = int(jnp.max(state.clump_id)) + 1
    counts = jnp.bincount(
        state.clump_id[pair_ids[:, 0]],
        weights=(force_norm > 0).astype(forces.dtype),
        length=N_clumps,
    )
    return state, system, counts.astype(int)


def count_clump_contacts(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
) -> tuple[State, System, jax.Array]:
    """Count force-bearing clump-level neighbors per clump.

    For every pair of clumps, sums the sphere-sphere contact forces into
    the clump-clump total and marks the pair as "in contact" iff the
    resulting total force has nonzero norm. Returns, per clump, the
    number of such neighbors. This matches the clump-pair convention
    used by :func:`compute_clump_pair_friction`: two clumps count as
    in contact when their net contact interaction is nonzero.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition.
    cutoff : float, optional
        Neighbor search cutoff distance.
    max_neighbors : int, optional
        Maximum number of neighbors per particle.

    Returns
    -------
    state : State
        Potentially updated state.
    system : System
        Potentially updated system.
    contacts : jax.Array
        ``(N_clumps,)`` integer array of force-bearing clump-level
        neighbors per clump.
    """
    state, system, pair_ids, forces = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )
    N_clumps = int(jnp.max(state.clump_id)) + 1
    # Mask out padding (j == -1) and intra-clump pairs; safe-index into
    # clump_id for padding so the scatter target is always valid.
    valid_pad = pair_ids[:, 1] != -1
    safe_j = jnp.maximum(pair_ids[:, 1], 0)
    clump_i = state.clump_id[pair_ids[:, 0]]
    clump_j = state.clump_id[safe_j]
    valid = valid_pad & (clump_i != clump_j)
    if not bool(jnp.any(valid)):
        return state, system, jnp.zeros((N_clumps,), dtype=int)

    # Sparse accumulation over the directed clump pairs that actually appear,
    # instead of a dense (N_clumps, N_clumps, dim) tensor.
    keys = clump_i[valid] * N_clumps + clump_j[valid]
    unique_keys, inverse = jnp.unique(keys, return_inverse=True)
    F_pairs = jax.ops.segment_sum(
        forces[valid], inverse, num_segments=unique_keys.shape[0]
    )
    has_contact = jnp.sum(F_pairs**2, axis=-1) > 0
    counts = jnp.bincount((unique_keys // N_clumps)[has_contact], length=N_clumps)
    return state, system, counts.astype(int)


def remove_rattlers(
    state: State, system: System, rattler_clump_ids: jax.Array
) -> tuple[State, System]:
    """Remove all spheres belonging to rattler clumps and rebuild a matching system.

    The state's rattler spheres are dropped and its ``clump_id`` /
    ``bond_id`` / ``unique_id`` arrays are re-indexed. The returned system
    is a :func:`dataclasses.replace` copy of the input — so every field
    (``domain``, ``mat_table``, integrators, user hooks, ``dt``, ``time``,
    and any future additions to :class:`System`) is preserved by default —
    with only the state-size-dependent fields refreshed:

    * ``collider`` is rebuilt via its :meth:`Create` method for stateful
      colliders (``NeighborList``, cell lists) and
      passed through unchanged for stateless ones (``naive``). Create's
      config kwargs are recovered from the current collider via
      introspection (see :func:`jaxdem.colliders.refresh_collider`).
    * ``force_manager`` is rebuilt so that its per-particle buffers
      (``external_force``, ``external_force_com``, ``external_torque``)
      are sized for the reduced state. ``gravity``, ``force_functions``,
      ``energy_functions``, and ``is_com_force`` are preserved.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        Current system; all fields are carried over into the rebuilt
        system except the state-size-dependent ones listed above.
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
    ``unique_id``. :func:`remove_rattlers` re-indexes ``unique_id`` in
    the state but does **not** remap the bonded-model topology, because
    the correct behavior is ambiguous (an element that partially
    straddles removed vertices could be dropped, re-triangulated, or
    flagged). A warning is emitted when a bonded model is present;
    users with DP systems should handle the topology remap manually.

    **Custom collider settings.** Any collider Create-kwarg whose name
    is not a field on the current collider (e.g. ``number_density`` and
    ``safety_factor`` on :class:`NeighborList`) gets Create's default
    value, not the value originally used. If you need to preserve such
    settings, rebuild the system yourself.
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

    # ``bond_id`` is an adjacency list of neighbor *unique_ids* padded with
    # -1, not dense labels, so it must be remapped through an old-uid ->
    # new-uid table (padding and removed neighbors stay -1).
    new_unique_id = jnp.arange(N_new, dtype=int)
    uid_remap = jnp.full((state.N,), -1, dtype=int)
    uid_remap = uid_remap.at[new_state.unique_id].set(new_unique_id)
    old_bond_id = new_state.bond_id
    keep_bond = old_bond_id >= 0
    new_state.bond_id = jnp.where(
        keep_bond, uid_remap[jnp.where(keep_bond, old_bond_id, 0)], -1
    )
    new_state.unique_id = new_unique_id

    # 2. Rebuild the collider (if stateful).
    new_collider = refresh_collider(new_state, system.collider)

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

    # force the rebuild of the neighborlist, if it is used
    # easy way to do this is to re-calculate the forces and torques
    # this is also convenient as the returned state has a valid force network with no forces attributed to removed particles
    new_state, new_system = new_system.collider.compute_force(new_state, new_system)
    new_state, new_system = new_system.force_manager.apply(new_state, new_system)

    return new_state, new_system


def compute_group_pair_friction(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    group_by: str = "clump_id",
) -> tuple[State, System, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Per-group-pair friction coefficient from decomposed total contact force.

    For every unique pair of groups :math:`(I, J)` — where "group" is
    defined by a shared value of an integer state attribute named by
    ``group_by`` — this function sums the per-sphere-pair forces returned
    by :func:`get_pair_forces_and_ids` between spheres of group :math:`I`
    and spheres of group :math:`J`, decomposes the resulting total force
    along the centroid-to-centroid axis, and reports

    .. math::
        \mu_{IJ} \;=\; \frac{|\mathbf{F}^{t}_{IJ}|}{|\mathbf{F}^{n}_{IJ}|}

    where :math:`\mathbf{F}^{n}_{IJ}` is the component of the total
    group-group force along the domain-aware minimum-image centroid
    axis and :math:`\mathbf{F}^{t}_{IJ}` is the remainder. The group
    centroid :math:`\mathbf{r}^{\rm c}_{I}` is the mean of
    ``state.pos_c`` over spheres sharing the group id.

    ``group_by`` is the attribute name on :class:`State` whose values
    define the grouping:

    * ``"clump_id"`` (default) groups by rigid clump, which is what you
      want for a rigid-clump system. For bodies where one clump owns a
      single vertex (bare spheres, DP nodes), the centroid collapses to
      the vertex position and the result is the per-sphere friction,
      which is trivially zero for radial pair forces.
    * ``"bond_id"`` groups by bonded body, which is what you want for
      a deformable-particle (DP) system: the centroid becomes the DP's
      vertex centroid, and the returned matrix is indexed by unique DP
      ids.
    * Any other integer state attribute is accepted if you want a
      custom grouping.

    Parameters
    ----------
    state, system, cutoff, max_neighbors
        Same as :func:`get_pair_forces_and_ids`.
    group_by : str, optional
        Name of the integer state attribute used to group spheres.
        Default ``"clump_id"``.

    Returns
    -------
    state : State
        Potentially updated state (after neighbor-list rebuild).
    system : System
        Potentially updated system.
    F_groups : jax.Array
        ``(n_groups, n_groups, dim)`` antisymmetric tensor;
        ``F_groups[I, J]`` is the total contact force on group :math:`I`
        from group :math:`J`. By Newton's third law
        ``F_groups[J, I] = -F_groups[I, J]``.
    mu : jax.Array
        ``(n_groups, n_groups)`` symmetric matrix of friction
        coefficients. Zero where the directed total group-pair force is
        zero or the group centroid axis is degenerate.
    contact_mask : jax.Array
        ``(n_groups, n_groups)`` symmetric bool matrix, ``True`` where
        the directed total group-pair force is nonzero and the group
        centroid axis is well-defined.
    sphere_counts : jax.Array
        ``(n_groups, n_groups, 2)`` integer tensor.
        ``sphere_counts[I, J, 0]`` is the number of distinct spheres in
        group :math:`I` that have at least one force-bearing contact
        with some sphere in group :math:`J`;
        ``sphere_counts[I, J, 1]`` is the symmetric count for spheres
        in :math:`J` contacting :math:`I`. Together they classify the
        contact "type" — e.g., a ``(1, 1)`` entry is a single
        sphere-sphere touch, a ``(2, 1)`` entry is two spheres of
        :math:`I` touching one sphere of :math:`J`, and so on.

    Notes
    -----
    - For rigid clumps, ``state.pos_c`` is constant within a clump and
      the group centroid equals the clump COM exactly. For DP nodes,
      ``pos_c`` equals the node position, so the centroid is the
      geometric mean of the DP's vertex positions.
    - Sphere pairs whose endpoints share the same group id are excluded.
    - The sphere-pair list from :func:`get_pair_forces_and_ids` contains
      both ``(i, j)`` and ``(j, i)`` directions; aggregation keeps the
      directed totals so ``F_groups[I, J]`` is the force on group
      :math:`I` from group :math:`J`.
    """
    state, system, pair_ids, forces = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )

    group_ids = getattr(state, group_by)
    n_groups = int(jnp.max(group_ids)) + 1

    group_pair_ids = group_ids[pair_ids]
    # Preserve padding sentinels. Indexing with -1 above temporarily maps
    # padded entries to the final group id; map them back to -1 before
    # building segment ids.
    group_pair_ids += -(group_pair_ids + 1) * (pair_ids == -1)

    i_sphere = pair_ids[:, 0]
    i_group = group_pair_ids[:, 0]
    j_group = group_pair_ids[:, 1]
    valid_entry = (i_group >= 0) & (j_group >= 0) & (i_group != j_group)

    # Accumulate directed per-group-pair forces. F_groups[I, J] is the
    # total force on group I from group J.
    pair_ids_group_flat = i_group[valid_entry] * n_groups + j_group[valid_entry]
    F_groups = jax.ops.segment_sum(
        forces[valid_entry],
        pair_ids_group_flat,
        num_segments=n_groups * n_groups,
    ).reshape(n_groups, n_groups, state.dim)

    # Group centroids. Segment-mean of ``pos_c`` works for all body
    # types: rigid clumps (all pos_c equal -> mean = pos_c), single
    # spheres (one value -> mean = pos_c), DPs (distinct pos_c per node
    # -> mean = geometric centroid).
    counts = jnp.bincount(group_ids, length=n_groups).astype(state.pos_c.dtype)
    sums = jax.ops.segment_sum(state.pos_c, group_ids, num_segments=n_groups)
    group_centroid = sums / jnp.maximum(counts[:, None], 1.0)

    diff = system.domain.displacement(
        group_centroid[:, None, :],
        group_centroid[None, :, :],
        system,
    )
    diff_mag = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    valid_pair = (diff_mag[..., 0] > 0) & (counts[:, None] > 0) & (counts[None, :] > 0)
    n_hat = diff / jnp.where(valid_pair[..., None], diff_mag, 1.0)

    # Decompose the directed F_groups along the centroid axis.
    Fn_scalar = jnp.sum(F_groups * n_hat, axis=-1)
    Fn_vec = Fn_scalar[..., None] * n_hat
    Ft_vec = F_groups - Fn_vec
    Ft_mag = jnp.linalg.norm(Ft_vec, axis=-1)

    F_mag = jnp.linalg.norm(F_groups, axis=-1)
    force_mask = F_mag > 0
    contact_mask = (force_mask | force_mask.T) & valid_pair
    mu = Ft_mag / (jnp.abs(Fn_scalar) + 1e-16)
    mu = jnp.where(force_mask & valid_pair, mu, 0.0)
    mu = jnp.where(contact_mask, jnp.maximum(mu, mu.T), 0.0)

    # Per-(group, group) sphere participation count. Mark each
    # force-bearing inter-group pair (a, b) with a in group I, b in
    # group J as "sphere a contacts group J", then count the distinct
    # spheres per group that contact each other group.
    fnonzero = norm(forces) > 0
    inter_group = valid_entry & fnonzero
    safe_j_group = jnp.maximum(j_group, 0)
    contact_membership = jnp.zeros((state.N, n_groups), dtype=int)
    contact_membership = contact_membership.at[i_sphere, safe_j_group].add(
        inter_group.astype(int)
    )
    contact_membership = (contact_membership > 0).astype(int)
    # n_spheres_in_I_contacting_J[I, J] = sum over s in I of in_contact[s, J].
    n_in_I_contact_J = jax.ops.segment_sum(
        contact_membership, group_ids, num_segments=n_groups
    )
    sphere_counts = jnp.stack([n_in_I_contact_J, n_in_I_contact_J.T], axis=-1)

    return state, system, F_groups, mu, contact_mask, sphere_counts


def compute_clump_pair_friction(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
) -> tuple[State, System, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Per-clump-pair friction coefficient from decomposed total contact force.

    Convenience alias for
    :func:`compute_group_pair_friction` with ``group_by="clump_id"`` —
    see that function for the full description and return shapes. The
    returned ``F_groups`` and ``mu`` are indexed by clump id.
    """
    return compute_group_pair_friction(
        state, system, cutoff, max_neighbors, group_by="clump_id"
    )


__all__ = [
    "compute_clump_pair_friction",
    "compute_contact_pressure",
    "compute_contact_stress_tensor",
    "compute_group_pair_friction",
    "count_clump_contacts",
    "count_vertex_contacts",
    "get_clump_rattler_ids",
    "get_pair_forces_and_ids",
    "get_sphere_rattler_ids",
    "remove_rattlers",
]
