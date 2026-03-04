# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Utility functions for analyzing particle contacts and identifying rattlers.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


def get_pair_forces_and_ids(
    state: State,
    system: System,
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
) -> Tuple[State, System, jax.Array, jax.Array]:
    """
    Compute pairwise contact forces and their associated particle IDs.

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
        cutoff = jnp.max(state.rad) * 3.0
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

    def per_pair_force(i, pos_pi, neighbors):
        def per_neighbor_force(j_id):
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


def get_pair_potential_derivatives(
    state: State,
    system: System,
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
) -> Tuple[State, System, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""
    Compute the first and second derivatives of the pair interaction potential
    with respect to the scalar pairwise distance for all neighbour pairs.

    For each pair :math:`(i, j)` this returns

    .. math::

        \hat{\mathbf{n}}_{ij}, \quad
        \frac{\partial \phi(r_{ij})}{\partial r_{ij}}
        \quad\text{and}\quad
        \frac{\partial^2 \phi(r_{ij})}{\partial r_{ij}^2}

    where :math:`r_{ij} = \lVert \mathbf{r}_i - \mathbf{r}_j \rVert`
    is the scalar centre-to-centre distance and
    :math:`\hat{\mathbf{n}}_{ij} = (\mathbf{r}_i - \mathbf{r}_j) / r_{ij}`
    is the unit vector from *j* to *i*.

    The first derivative is obtained by projecting the pairwise force onto the
    pair direction:
    :math:`\partial\phi/\partial r = -\mathbf{F}_{ij} \cdot \hat{\mathbf{n}}_{ij}`,
    and the second derivative is obtained from
    ``system.force_model.stiffness``.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition containing the collider and force model.
    cutoff : float, optional
        Neighbour-search cutoff distance.  Defaults to ``3 * max(rad)``.
    max_neighbors : int, optional
        Maximum number of neighbours per particle (default 100).

    Returns
    -------
    state : State
        Potentially updated state (after neighbour-list rebuild).
    system : System
        Potentially updated system.
    pair_ids : jax.Array
        ``(M, 2)`` array of ``(i, j)`` sphere index pairs.
    nhat : jax.Array
        ``(M, dim)`` unit vectors
        :math:`\hat{\mathbf{n}}_{ij} = \mathbf{r}_{ij} / r_{ij}`.
    distances : jax.Array
        ``(M,)`` scalar pairwise distances :math:`r_{ij}`.
    dphi_dr : jax.Array
        ``(M,)`` first derivatives
        :math:`\partial\phi / \partial r_{ij}`.
    d2phi_dr2 : jax.Array
        ``(M,)`` second derivatives
        :math:`\partial^2\phi / \partial r_{ij}^2`.
    """
    if cutoff is None:
        cutoff = jnp.max(state.rad) * 3.0
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

    def per_pair(i, neighbors):
        def per_neighbor(j_id):
            valid = j_id != -1
            safe_j = jnp.maximum(j_id, 0)

            rij = system.domain.displacement(pos[i], pos[safe_j], system)
            r2 = jnp.sum(rij**2, axis=-1)
            r = jnp.sqrt(jnp.where(r2 == 0, 1.0, r2))
            nhat = rij / r

            F, _ = system.force_model.force(i, safe_j, pos, state, system)
            dp = -jnp.sum(F * nhat, axis=-1)
            d2p = system.force_model.stiffness(i, safe_j, pos, state, system)

            return nhat * valid, r * valid, dp * valid, d2p * valid

        return jax.vmap(per_neighbor)(neighbors)

    nhats, distances, dphi, d2phi = jax.vmap(per_pair)(sphere_ids, nl)

    n_neighbors = nl.shape[1]
    i_ids = jnp.repeat(sphere_ids[:, None], n_neighbors, axis=1).ravel()
    j_ids = nl.ravel()

    return (
        state,
        system,
        jnp.column_stack((i_ids, j_ids)),
        nhats.reshape(-1, state.dim),
        distances.ravel(),
        dphi.ravel(),
        d2phi.ravel(),
    )

def compute_hessian_spheres(
    state: State,
    system: System,
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    reshape: bool = True,
) -> Tuple[State, System, jax.Array]:
    """
    Compute the Hessian matrix of the pairwise interaction potential for a
    system of spheres.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition containing the collider and force model.
    cutoff : float, optional
        Neighbour-search cutoff distance.  Defaults to ``3 * max(rad)``.
    max_neighbors : int, optional
        Maximum number of neighbours per particle (default 100).
    reshape : bool, optional
        If ``True`` (default), return the Hessian as a 2-D
        ``(N*dim, N*dim)`` matrix.  Otherwise return the 4-D
        ``(N, N, dim, dim)`` block tensor.

    Returns
    -------
    state : State
        Potentially updated state (after neighbour-list rebuild).
    system : System
        Potentially updated system.
    hessian : jax.Array
        If *reshape* is ``True``, shape ``(N*dim, N*dim)``.
        Otherwise shape ``(N, N, dim, dim)``.
    """
    state, system, pair_ids, r_ij_unit, pair_dists, pair_t, pair_c = get_pair_potential_derivatives(
        state,
        system,
        cutoff,
        max_neighbors
    )

    valid = (pair_ids[:, 1] != -1) & (pair_ids[:, 0] != pair_ids[:, 1])
    ids = pair_ids[valid]
    nhat = r_ij_unit[valid]
    r = pair_dists[valid]
    t = pair_t[valid]
    c = pair_c[valid]

    hessian = jnp.zeros((state.N, state.N, state.dim, state.dim))
    nn = nhat[:, :, None] * nhat[:, None, :]
    eye = jnp.eye(state.dim)
    tr = (t / r)[:, None, None]
    block = -c[:, None, None] * nn + tr * (nn - eye[None])
    hessian = hessian.at[ids[:, 0], ids[:, 1], :state.dim, :state.dim].add(block) # off-diagonal
    hessian = hessian.at[ids[:, 0], ids[:, 0], :state.dim, :state.dim].add(-block) # diagonal

    if reshape:
        hessian = hessian.transpose(0, 2, 1, 3).reshape(state.N * state.dim, state.N * state.dim)
    return state, system, hessian

def compute_hessian_clumps_2d(
    state: State,
    system: System,
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    reshape: bool = True,
) -> Tuple[State, System, jax.Array]:
    """
    Compute the Hessian matrix of the pairwise interaction potential for
    a 2-D system of rigid clumps, including rotational degrees of freedom.

    Parameters
    ----------
    state : State
        Current simulation state (must be 2-D).
    system : System
        System definition containing the collider and force model.
    cutoff : float, optional
        Neighbour-search cutoff distance.  Defaults to ``3 * max(rad)``.
    max_neighbors : int, optional
        Maximum number of neighbours per particle (default 100).
    reshape : bool, optional
        If ``True`` (default), return the Hessian as a 2-D
        ``(N_c*df, N_c*df)`` matrix where ``df = dim + angular_dof``.
        Otherwise return the 4-D ``(N_c, N_c, df, df)`` block tensor.

    Returns
    -------
    state : State
        Potentially updated state (after neighbour-list rebuild).
    system : System
        Potentially updated system.
    hessian : jax.Array
        If *reshape* is ``True``, shape ``(N_c*df, N_c*df)``.
        Otherwise shape ``(N_c, N_c, df, df)``.

    Raises
    ------
    ValueError
        If ``state.dim != 2``.
    """
    if state.dim != 2:
        raise ValueError('compute_hessian_clumps_2d only works in 2D!')

    state, system, pair_ids, r_ij_unit, pair_dists, pair_t, pair_c = get_pair_potential_derivatives(
        state,
        system,
        cutoff,
        max_neighbors
    )

    N_c = int(jnp.max(state.clump_id)) + 1
    df = state.dim + state.ang_vel.shape[1]

    valid = (pair_ids[:, 1] != -1) & (pair_ids[:, 0] != pair_ids[:, 1])
    ids = pair_ids[valid]
    nhat = r_ij_unit[valid]
    r = pair_dists[valid]
    t = pair_t[valid]
    c = pair_c[valid]

    clump_i = state.clump_id[ids[:, 0]]
    clump_j = state.clump_id[ids[:, 1]]
    inter_body = clump_i != clump_j
    ids, nhat, r, t, c = ids[inter_body], nhat[inter_body], r[inter_body], t[inter_body], c[inter_body]
    clump_i, clump_j = clump_i[inter_body], clump_j[inter_body]

    lev_mu = state.pos[ids[:, 0]] - state.pos_c[ids[:, 0]]
    lev_nu = state.pos[ids[:, 1]] - state.pos_c[ids[:, 1]]

    n = len(ids)
    E_mu = jnp.zeros((n, 3, 2))
    E_mu = E_mu.at[:, 0, 0].set(1.0)
    E_mu = E_mu.at[:, 1, 1].set(1.0)
    E_mu = E_mu.at[:, 2, 0].set(-lev_mu[:, 1])
    E_mu = E_mu.at[:, 2, 1].set(lev_mu[:, 0])

    E_nu = jnp.zeros((n, 3, 2))
    E_nu = E_nu.at[:, 0, 0].set(1.0)
    E_nu = E_nu.at[:, 1, 1].set(1.0)
    E_nu = E_nu.at[:, 2, 0].set(-lev_nu[:, 1])
    E_nu = E_nu.at[:, 2, 1].set(lev_nu[:, 0])

    p_mu = jnp.einsum('kac,kc->ka', E_mu, nhat)
    p_nu = jnp.einsum('kac,kc->ka', E_nu, nhat)
    tr = t / r

    EE_cross = jnp.einsum('kac,kbc->kab', E_mu, E_nu)
    pp_cross = p_mu[:, :, None] * p_nu[:, None, :]
    off_block = -tr[:, None, None] * EE_cross - (c - tr)[:, None, None] * pp_cross

    EE_same = jnp.einsum('kac,kbc->kab', E_mu, E_mu)
    pp_same = p_mu[:, :, None] * p_mu[:, None, :]
    diag_block = tr[:, None, None] * EE_same + (c - tr)[:, None, None] * pp_same
    diag_block = diag_block.at[:, 2, 2].add(-t * jnp.einsum('kc,kc->k', nhat, lev_mu))

    hessian = jnp.zeros((N_c, N_c, 3, 3))
    hessian = hessian.at[clump_i, clump_j].add(off_block)
    hessian = hessian.at[clump_i, clump_i].add(diag_block)

    if reshape:
        hessian = hessian.transpose(0, 2, 1, 3).reshape(N_c * df, N_c * df)
    return state, system, hessian

def get_clump_rattler_ids(
    state: State,
    system: System,
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    zc: Optional[int] = None,
) -> Tuple[State, System, jax.Array, jax.Array]:
    """
    Identify rattler clumps by iteratively removing under-coordinated clumps.

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
        Minimum contact count. Defaults to ``dim + angular_dof + 1``.

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

    force_norm = jnp.linalg.norm(neigh_force, axis=-1)
    pair_ids = pair_ids[force_norm > 0]

    N_clumps = int(jnp.max(state.clump_id)) + 1

    if zc is None:
        dof = state.ang_vel.shape[-1] + state.dim
        zc = dof + 1

    all_clump_ids = jnp.arange(N_clumps)
    clumps_in_contacts = jnp.unique(state.clump_id[pair_ids.ravel()])
    rattler_ids = jnp.setdiff1d(all_clump_ids, clumps_in_contacts)

    while True:
        if pair_ids.shape[0] == 0:
            warnings.warn("No valid particles remain after rattler removal.")
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
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    zc: Optional[int] = None,
) -> Tuple[State, System, jax.Array, jax.Array]:
    """
    Identify rattler spheres by iteratively removing under-coordinated particles.

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
        Minimum contact count. Defaults to ``dim + 1``.

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

    force_norm = jnp.linalg.norm(neigh_force, axis=-1)
    pair_ids = pair_ids[force_norm > 0]

    N = state.N
    if zc is None:
        zc = state.dim + 1

    all_ids = jnp.arange(N)
    rattler_ids = jnp.setdiff1d(all_ids, jnp.unique(pair_ids.ravel()))

    while True:
        if pair_ids.shape[0] == 0:
            warnings.warn("No valid particles remain after rattler removal.")
            break

        contacts = jnp.bincount(pair_ids[:, 0], length=N)
        active = jnp.unique(pair_ids[:, 0])
        new_rattlers = jnp.setdiff1d(
            active[contacts[active] < zc], rattler_ids
        )

        if len(new_rattlers) == 0:
            break

        rattler_ids = jnp.union1d(rattler_ids, new_rattlers)
        pair_ids = pair_ids[
            ~(jnp.isin(pair_ids[:, 0], rattler_ids) | jnp.isin(pair_ids[:, 1], rattler_ids))
        ]

    non_rattler_ids = jnp.setdiff1d(all_ids, rattler_ids)
    return state, system, rattler_ids, non_rattler_ids


def _count_vertex_contacts_per_clump(
    pair_ids: jax.Array, clump_id: jax.Array, N_clumps: int
) -> jax.Array:
    """Count vertex-level contacts per clump (sphere pairs, not unique clump pairs)."""
    return jnp.bincount(clump_id[pair_ids[:, 0]], length=N_clumps)


def count_vertex_contacts(
    state: State,
    system: System,
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
) -> Tuple[State, System, jax.Array]:
    """
    Count vertex-level contacts per clump.

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
        ``(N_clumps,)`` array of vertex contact counts per clump.
    """
    state, system, pair_ids, _ = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )
    N_clumps = int(jnp.max(state.clump_id)) + 1
    return state, system, _count_vertex_contacts_per_clump(
        pair_ids, state.clump_id, N_clumps
    )


def _count_clump_contacts_per_clump(
    pair_ids: jax.Array, clump_id: jax.Array, N_clumps: int
) -> jax.Array:
    """Count unique clump-level contacts per clump via an adjacency matrix."""
    clump_i = clump_id[pair_ids[:, 0]]
    clump_j = clump_id[pair_ids[:, 1]]
    adj = jnp.zeros((N_clumps, N_clumps), dtype=bool)
    adj = adj.at[clump_i, clump_j].set(True)
    adj = adj & ~jnp.eye(N_clumps, dtype=bool)
    return jnp.sum(adj, axis=1)


def count_clump_contacts(
    state: State,
    system: System,
    cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
) -> Tuple[State, System, jax.Array]:
    """
    Count unique clump-level contacts per clump.

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
        ``(N_clumps,)`` array of unique clump contact counts per clump.
    """
    state, system, pair_ids, _ = get_pair_forces_and_ids(
        state, system, cutoff, max_neighbors
    )
    N_clumps = int(jnp.max(state.clump_id)) + 1
    return state, system, _count_clump_contacts_per_clump(
        pair_ids, state.clump_id, N_clumps
    )


def remove_rattlers_from_state(
    state: State,
    system: System,
    rattler_clump_ids: jax.Array,
) -> Tuple[State, System]:
    """
    Remove all spheres belonging to rattler clumps and rebuild the state.

    If the system uses a :class:`~jaxdem.colliders.NeighborList` collider, its
    cached buffers are resized to the new particle count and marked for rebuild.

    Parameters
    ----------
    state : State
        Current simulation state.
    system : System
        System definition (collider may be updated).
    rattler_clump_ids : jax.Array
        1-D array of clump IDs to remove.

    Returns
    -------
    state : State
        A new state with rattler spheres removed and IDs re-indexed.
    system : System
        System with collider buffers resized (if applicable).
    """
    from ..colliders import NeighborList

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

    if isinstance(system.collider, NeighborList):
        collider = system.collider
        new_state, nl, old_pos, n_build, overflow = NeighborList._rebuild(
            collider, new_state, system
        )
        system = replace(
            system,
            collider=replace(
                collider,
                neighbor_list=nl,
                old_pos=old_pos,
                n_build_times=n_build,
                overflow=overflow,
            ),
        )

    return new_state, system


# TODO: add force colinearity check for 2D and coplanarity check for 3D


__all__ = [
    "get_pair_forces_and_ids",
    "get_pair_potential_derivatives",
    "compute_hessian_spheres",
    "compute_hessian_clumps_2d",
    "get_clump_rattler_ids",
    "get_sphere_rattler_ids",
    "count_vertex_contacts",
    "count_clump_contacts",
    "remove_rattlers_from_state",
]
