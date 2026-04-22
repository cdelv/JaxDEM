# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions for analyzing particle contacts and identifying rattlers."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .linalg import norm

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
        ``(M, 2)`` array of ``(i, j)`` sphere index pairs.
    forces : jax.Array
        ``(M, dim)`` array of pairwise force vectors, one per pair.

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
    pos_p_global = state.q.rotate(state.q, state.pos_p)
    pos = state.pos_c + pos_p_global

    def per_pair_force(
        i: jnp.ndarray, pos_pi: jnp.ndarray, neighbors: jnp.ndarray
    ) -> jnp.ndarray:
        def per_neighbor_force(j_id: jnp.ndarray) -> jnp.ndarray:
            valid = j_id != -1
            safe_j = jnp.maximum(j_id, 0)
            f, _ = system.force_model.force(i, safe_j, pos, state, system)
            return f * valid

        return jax.vmap(per_neighbor_force)(neighbors)

    neigh_force = jax.vmap(per_pair_force)(sphere_ids, pos_p_global, nl)

    n_neighbors = nl.shape[1]
    i_ids = jnp.repeat(sphere_ids[:, None], n_neighbors, axis=1).ravel()
    j_ids = nl.ravel()
    neigh_force = neigh_force.reshape(-1, state.dim)

    return state, system, jnp.column_stack((i_ids, j_ids)), neigh_force


def get_clump_rattler_ids(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    zc: int | None = None,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Identify rattler clumps by iteratively removing under-coordinated clumps.

    A clump is a rattler if its total vertex-contact count is below the
    coordination threshold *zc*.

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
    pair_ids = pair_ids[force_norm > 0]

    N_clumps = int(jnp.max(state.clump_id)) + 1

    if zc is None:
        zc = state.ang_vel.shape[-1] + state.dim + 1

    all_clump_ids = jnp.arange(N_clumps)
    clumps_in_contacts = jnp.unique(state.clump_id[pair_ids.ravel()])
    rattler_ids = jnp.setdiff1d(all_clump_ids, clumps_in_contacts)

    while True:
        if pair_ids.shape[0] == 0:
            warnings.warn(
                "No valid particles remain after rattler removal.", stacklevel=2
            )
            break

        clump_i = state.clump_id[pair_ids[:, 0]]
        vertex_contacts = jnp.bincount(clump_i, length=N_clumps)
        active_clumps = jnp.unique(clump_i)
        new_rattlers = jnp.setdiff1d(
            active_clumps[vertex_contacts[active_clumps] < zc], rattler_ids
        )

        if len(new_rattlers) == 0:
            break

        rattler_ids = jnp.union1d(rattler_ids, new_rattlers)
        clump_j = state.clump_id[pair_ids[:, 1]]
        pair_ids = pair_ids[
            ~(jnp.isin(clump_i, rattler_ids) | jnp.isin(clump_j, rattler_ids))
        ]

    non_rattler_ids = jnp.setdiff1d(all_clump_ids, rattler_ids)
    return state, system, rattler_ids, non_rattler_ids


def get_sphere_rattler_ids(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    zc: int | None = None,
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
    pair_ids = pair_ids[force_norm > 0]

    N = state.N
    if zc is None:
        zc = state.dim + 1

    all_ids = jnp.arange(N)
    rattler_ids = jnp.setdiff1d(all_ids, jnp.unique(pair_ids.ravel()))

    while True:
        if pair_ids.shape[0] == 0:
            warnings.warn(
                "No valid particles remain after rattler removal.", stacklevel=2
            )
            break

        contacts = jnp.bincount(pair_ids[:, 0], length=N)
        active = jnp.unique(pair_ids[:, 0])
        new_rattlers = jnp.setdiff1d(active[contacts[active] < zc], rattler_ids)

        if len(new_rattlers) == 0:
            break

        rattler_ids = jnp.union1d(rattler_ids, new_rattlers)
        pair_ids = pair_ids[
            ~(
                jnp.isin(pair_ids[:, 0], rattler_ids)
                | jnp.isin(pair_ids[:, 1], rattler_ids)
            )
        ]

    non_rattler_ids = jnp.setdiff1d(all_ids, rattler_ids)
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
    dim = state.dim
    # Mask out padding (j == -1) and intra-clump pairs; safe-index into
    # clump_id for padding so the scatter target is always valid.
    valid_pad = pair_ids[:, 1] != -1
    safe_j = jnp.maximum(pair_ids[:, 1], 0)
    clump_i = state.clump_id[pair_ids[:, 0]]
    clump_j = state.clump_id[safe_j]
    valid = valid_pad & (clump_i != clump_j)
    forces_masked = forces * valid[:, None]
    F_clumps = jnp.zeros((N_clumps, N_clumps, dim), dtype=forces.dtype)
    F_clumps = F_clumps.at[clump_i, clump_j].add(forces_masked)
    F_norm_IJ = jnp.sqrt(jnp.sum(F_clumps ** 2, axis=-1))
    has_contact = F_norm_IJ > 0
    return state, system, jnp.sum(has_contact, axis=1).astype(int)


def _refresh_collider(collider: Any, new_state: State) -> Any:
    """Rebuild a stateful collider for a new state size, preserving its Create-kwargs.

    Stateless colliders (``naive``) have no state-size-dependent buffers and
    are returned unchanged. Stateful colliders are rebuilt by introspecting
    their ``Create`` signature and forwarding any parameter whose name is
    also a dataclass field on the current collider instance (plus the new
    ``state``). Parameters not stored on the collider fall back to Create's
    own defaults.
    """
    from inspect import signature

    stateful = {
        "neighborlist", "celllist", "staticcelllist", "dynamiccelllist", "sap",
    }
    if collider.type_name.lower() not in stateful:
        return collider

    create_fn = getattr(type(collider), "Create", None)
    if create_fn is None:
        return collider

    kwargs: dict[str, Any] = {}
    for pname in signature(create_fn).parameters:
        if pname in ("cls", "self"):
            continue
        if pname == "state":
            kwargs[pname] = new_state
        elif hasattr(collider, pname):
            kwargs[pname] = getattr(collider, pname)
    return type(collider).Create(**kwargs)


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
      colliders (``NeighborList``, cell lists, sweep-and-prune) and
      passed through unchanged for stateless ones (``naive``). Create's
      config kwargs are recovered from the current collider via
      introspection (see :func:`_refresh_collider`).
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
    from ..forces.force_manager import ForceManager

    # 1. State update.
    keep = ~jnp.isin(state.clump_id, rattler_clump_ids)
    idx = jnp.where(keep)[0]
    new_state = jax.tree.map(lambda x: x[idx], state)
    N_new = idx.shape[0]
    _, new_clump_id = jnp.unique(new_state.clump_id, return_inverse=True, size=N_new)
    _, new_bond_id = jnp.unique(new_state.bond_id, return_inverse=True, size=N_new)
    new_state = replace(
        new_state,
        clump_id=new_clump_id,
        bond_id=new_bond_id,
        unique_id=jnp.arange(N_new, dtype=int),
    )

    # 2. Rebuild the collider (if stateful).
    new_collider = _refresh_collider(system.collider, new_state)

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

    return new_state, new_system


# TODO: add colinearity check for 2D and coplanarity check for 3D


def compute_clump_pair_friction(
    state: State,
    system: System,
    cutoff: float | None = None,
    max_neighbors: int | None = None,
) -> tuple[State, System, jax.Array, jax.Array, jax.Array]:
    r"""Per-clump-pair friction coefficient from decomposed total contact force.

    For every unique clump pair :math:`(I, J)` with at least one
    sphere-sphere contact, this function sums the per-pair forces returned
    by :func:`get_pair_forces_and_ids` between spheres in clump :math:`I`
    and spheres in clump :math:`J`, decomposes the resulting total force
    along the center-of-mass to center-of-mass axis, and reports

    .. math::
        \mu_{IJ} \;=\; \frac{|\mathbf{F}^{t}_{IJ}|}{|\mathbf{F}^{n}_{IJ}|}

    where :math:`\mathbf{F}^{n}_{IJ}` is the component of the total
    clump-clump force along :math:`\hat{\mathbf{n}}_{IJ} =
    (\mathbf{r}^{COM}_{J} - \mathbf{r}^{COM}_{I}) / |...|` and
    :math:`\mathbf{F}^{t}_{IJ}` is the remainder.

    Parameters
    ----------
    state, system, cutoff, max_neighbors
        Same as :func:`get_pair_forces_and_ids`.

    Returns
    -------
    state : State
        Potentially updated state (after neighbor-list rebuild).
    system : System
        Potentially updated system.
    F_clumps : jax.Array
        ``(n_clumps, n_clumps, dim)`` antisymmetric tensor;
        ``F_clumps[I, J]`` is the total contact force on clump :math:`I`
        from clump :math:`J`. By Newton's third law
        ``F_clumps[J, I] = -F_clumps[I, J]``.
    mu : jax.Array
        ``(n_clumps, n_clumps)`` symmetric matrix of friction coefficients.
        Zero where the pair has no contact. ``jnp.inf`` in the edge case
        where the total force is purely tangential (``|F_n| = 0`` with
        ``|F_t| > 0``) -- physically, infinite friction is the correct
        answer there.
    contact_mask : jax.Array
        ``(n_clumps, n_clumps)`` symmetric bool matrix, ``True`` where
        the clump pair has at least one sphere-sphere contact with
        nonzero force.

    Notes
    -----
    - The clump "center of mass" used for the decomposition is the mean
      of ``state.pos_c`` over spheres sharing a clump id. For rigid
      clumps all spheres share ``pos_c``, so this is exactly the COM.
      For single-sphere clumps it's the sphere position.
    - Sphere pairs inside the same clump are excluded.
    - The sphere-pair list from :func:`get_pair_forces_and_ids` contains
      both ``(i, j)`` and ``(j, i)`` directions; aggregation is
      canonicalized to the ``clump_id[i] < clump_id[j]`` direction so
      each unordered clump pair is summed exactly once.
    """
    state, system, pair_ids, forces = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )

    i_sphere = pair_ids[:, 0]
    j_sphere = pair_ids[:, 1]
    i_clump = state.clump_id[i_sphere]
    j_clump = state.clump_id[j_sphere]

    # Keep one direction of each unordered clump pair; drop padding and
    # drop intra-clump pairs (forces between spheres of the same clump).
    valid_entry = (j_sphere != -1) & (i_clump < j_clump)
    forces_masked = forces * valid_entry[:, None]

    # Accumulate per-clump-pair forces. Upper triangle (I < J) of F_IJ
    # holds "force on clump I from clump J"; lower triangle is zero at
    # this point.
    n_clumps = int(jnp.max(state.clump_id)) + 1
    dim = state.dim
    F_IJ = jnp.zeros((n_clumps, n_clumps, dim), dtype=forces.dtype)
    F_IJ = F_IJ.at[i_clump, j_clump].add(forces_masked)

    # Antisymmetrize: F_clumps[J, I] = -F_clumps[I, J] (Newton's third).
    F_clumps = F_IJ - jnp.transpose(F_IJ, (1, 0, 2))

    # Clump COMs. Segment-mean of ``pos_c`` works for all three body
    # types: rigid clumps (all pos_c equal -> mean = pos_c), single
    # spheres (one value -> mean = pos_c), DPs (distinct pos_c per node
    # -> mean = centroid).
    counts = jnp.bincount(state.clump_id, length=n_clumps).astype(
        state.pos_c.dtype
    )
    sums = jax.ops.segment_sum(state.pos_c, state.clump_id, num_segments=n_clumps)
    clump_com = sums / jnp.maximum(counts[:, None], 1.0)

    # n_hat[I, J] points from clump I to clump J.
    diff = clump_com[None, :, :] - clump_com[:, None, :]  # (n, n, dim)
    diff_mag = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / jnp.where(diff_mag == 0, 1.0, diff_mag)

    # Decompose the antisymmetric F_clumps along n_hat.
    Fn_scalar = jnp.sum(F_clumps * n_hat, axis=-1)  # (n, n)
    Ft_vec = F_clumps - Fn_scalar[..., None] * n_hat  # (n, n, dim)
    Fn_mag = jnp.abs(Fn_scalar)
    Ft_mag = jnp.linalg.norm(Ft_vec, axis=-1)

    # mu with the double-where trick to avoid nan under autograd. F_n=0
    # with nonzero F_t -> inf (infinite friction for a purely tangential
    # contact force). Both zero -> inf too (degenerate, doesn't occur in
    # practice for real contacts).
    mu = jnp.where(
        Fn_mag > 0,
        Ft_mag / jnp.where(Fn_mag > 0, Fn_mag, 1.0),
        jnp.inf,
    )

    # Zero mu out for non-contacts; symmetrize by reflecting the upper
    # triangle. Diagonal is zero in both.
    upper = jnp.triu(jnp.ones_like(Fn_mag, dtype=bool), k=1)
    F_mag = jnp.linalg.norm(F_clumps, axis=-1)
    contact_upper = upper & (F_mag > 0)
    mu = jnp.where(contact_upper, mu, 0.0)
    mu = mu + mu.T

    # Symmetric contact mask.
    contact_mask = contact_upper | contact_upper.T

    return state, system, F_clumps, mu, contact_mask


__all__ = [
    "compute_clump_pair_friction",
    "count_clump_contacts",
    "count_vertex_contacts",
    "get_clump_rattler_ids",
    "get_pair_forces_and_ids",
    "get_sphere_rattler_ids",
    "remove_rattlers",
]
