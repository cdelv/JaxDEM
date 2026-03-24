# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, Union
from collections.abc import Callable
from functools import partial

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from . import Collider, valid_interaction_mask
from ..utils.linalg import cross, norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
@partial(jax.named_call, name="cell_list._get_spatial_partition")
def _get_spatial_partition(
    pos: jax.Array,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    iota: jax.Array,
) -> tuple[jax.Array, ...]:
    """Computes spatial hashing and partitioning for the cell list.

    Parameters
    ----------
    pos : jax.Array
        Particle positions, shape (N, dim).
    system : System
        The system configuration, used for domain information.
    cell_size : jax.Array
        Linear size of a grid cell.
    neighbor_mask : jax.Array
        Integer offsets defining the neighbor stencil, shape (M, dim).
    iota : jax.Array
        Indices [0, 1, ..., N-1].

    Returns
    -------
    Tuple[jax.Array, ...]
        A tuple containing:
        - perm: Sorting permutation to order particles by cell hash.
        - p_cell_coords: Cell coordinates for each particle, shape (N, dim).
        - p_cell_hash: Flattened cell hashes for each particle (sorted), shape (N,).
        - neighbor_cell_coords: Coordinates of neighbor cells for each particle, shape (N, M, dim).
        - neighbor_cell_hashes: Flattened hashes of neighbor cells for each particle, shape (N, M).

    """
    # 1. Determine Grid Dimensions
    # shape: (dim,)
    if system.domain.periodic:
        grid_dims = jnp.floor(system.domain.box_size / cell_size).astype(int)
    else:
        grid_dims = jnp.ceil(system.domain.box_size / cell_size).astype(int)

    # Compute strides (weights) for flattening 2D/3D indices to 1D hash
    # [1, nx, nx*ny, ...]
    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
    )

    # 2. Calculate Particle Cell Coords
    if system.domain.periodic:
        p_cell_coords = jnp.floor(
            (((pos - system.domain.anchor) / system.domain.box_size) % 1) * grid_dims
        ).astype(int)
    else:
        p_cell_coords = jnp.floor((pos - system.domain.anchor) / cell_size).astype(int)

    # 3. Spatial Hashing
    # shape (N,)
    p_cell_hash = jnp.dot(p_cell_coords, grid_strides)

    # 4. Sort hashes
    p_cell_hash, perm = jax.lax.sort([p_cell_hash, iota], num_keys=1)
    p_cell_coords = p_cell_coords[perm]

    # 5. Identify Neighboring Cells to search
    # For every particle, calculate the coords of all adjacent cells (including its own)
    # shape: (N, M, dim) where M is the number of cells in the stencil (e.g., 9 or 27)
    # (N, M, dim) = (N, 1, dim) + (1, M, dim)
    neighbor_cell_coords = p_cell_coords[:, None, :] + neighbor_mask

    if system.domain.periodic:
        neighbor_cell_coords -= grid_dims * jnp.floor(
            neighbor_cell_coords / grid_dims
        ).astype(int)

    # Flatten neighbor cell coords to hashes for quick lookup
    # shape (N, M)
    neighbor_cell_hashes = jnp.dot(neighbor_cell_coords, grid_strides)

    return (
        perm,
        p_cell_coords,
        p_cell_hash,
        neighbor_cell_coords,
        neighbor_cell_hashes,
    )


@jax.jit
@partial(jax.named_call, name="cell_list._dedup_stencil_hashes")
def _dedup_stencil_hashes(stencil_hashes: jax.Array) -> jax.Array:
    """Deduplicate one particle's stencil hashes, padding duplicates with -1.

    Parameters
    ----------
    stencil_hashes : jax.Array
        Array of cell hashes for a particle's neighbor stencil.

    Returns
    -------
    jax.Array
        Deduplicated hashes, with duplicates replaced by -1.

    """
    sorted_hashes = jnp.sort(stencil_hashes)
    is_unique = jnp.ones_like(sorted_hashes, dtype=bool)
    is_unique = is_unique.at[1:].set(sorted_hashes[1:] != sorted_hashes[:-1])
    return jnp.where(is_unique, sorted_hashes, -1)


@partial(jax.named_call, name="cell_list._compute_interaction")
def _compute_interaction(
    state: State,
    system: System,
    traverse_fn: Callable[..., Any],
    interaction_fn: Callable[..., Any],
    reduce_fn: Callable[..., Any] | None = None,
) -> tuple[State, Any]:
    """Common logic for computing interactions (Force or Energy) using a Cell List.

    Parameters
    ----------
    state : State
        The current state.
    system : System
        The system configuration.
    traverse_fn : Callable
        A function that traverses a single neighbor cell.
        Signature: (target_hash, state, system, pos, p_cell_hash, idx, interaction_fn) -> result
    interaction_fn : Callable
        A callback invoked by traverse_fn to compute the physics.
    reduce_fn : Optional[Callable]
        A function to post-process the summed results for a particle (e.g., adding torque cross product).
        Signature: (summed_result, pos_pi) -> final_result

    Returns
    -------
    Tuple[State, Any]
        (Sorted State, Result Array)

    """
    # We cast to Union to access cell_size and neighbor_mask which are present in both
    collider = cast(Union["StaticCellList", "DynamicCellList"], system.collider)
    iota = jax.lax.iota(dtype=int, size=state.N)
    pos_p = state.q.rotate(state.q, state.pos_p)
    pos = state.pos_c + pos_p

    # 1. Spatial Partitioning
    (
        perm,
        _,  # p_cell_coords
        p_cell_hash,
        _,  # neighbor_cell_coords
        p_neighbor_cell_hashes,
    ) = _get_spatial_partition(
        pos, system, collider.cell_size, collider.neighbor_mask, iota
    )

    # Permute state to sorted order
    state = jax.tree.map(lambda x: x[perm], state)
    pos = pos[perm]
    pos_p = pos_p[perm]

    def per_particle(
        idx: jax.Array, pos_pi: jax.Array, neighbor_hashes: jax.Array
    ) -> Any:
        if system.domain.periodic:
            neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes)

        def per_cell(target_hash: jax.Array) -> Any:
            return traverse_fn(
                target_hash, state, system, pos, p_cell_hash, idx, interaction_fn
            )

        # Map over stencil
        cell_results = jax.vmap(per_cell)(neighbor_hashes)

        # Sum contributions from all neighbor cells
        # jax.tree.map(sum) works for both scalar (Energy) and Tuple[Array, Array] (Force)
        summed = jax.tree.map(lambda x: x.sum(axis=0), cell_results)

        if reduce_fn is not None:
            summed = reduce_fn(summed, pos_pi)

        return summed

    # 2. Compute interactions for all particles
    results = jax.vmap(per_particle)(iota, pos_p, p_neighbor_cell_hashes)

    return state, results


@partial(jax.named_call, name="cell_list._compute_neighbor_list_common")
def _compute_neighbor_list_common(
    state: State,
    system: System,
    traverse_fn: Callable[..., Any],
    max_neighbors: int,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Common logic for creating neighbor lists using a Cell List.

    Parameters
    ----------
    state : State
        The current state.
    system : System
        The system configuration.
    traverse_fn : Callable
        A function that finds neighbors for a single particle.
        Signature: (idx, pos_i, stencil, state, system, pos, p_cell_hash) -> (neighbor_list, overflow)

    Returns
    -------
    Tuple[State, System, jax.Array, jax.Array]
        (Sorted State, System, Neighbor List, Overflow Flag)

    """
    if max_neighbors == 0:
        empty = jnp.empty((state.N, 0), dtype=int)
        return state, system, empty, jnp.asarray(False)

    collider = cast(Union["StaticCellList", "DynamicCellList"], system.collider)
    iota = jax.lax.iota(int, state.N)
    pos = state.pos

    # 1. Spatial Partitioning
    (
        perm,
        _,  # p_cell_coords
        p_cell_hash,
        _,  # neighbor_cell_coords
        p_neighbor_hashes,
    ) = _get_spatial_partition(
        pos, system, collider.cell_size, collider.neighbor_mask, iota
    )

    # Permute state to sorted order
    state = jax.tree.map(lambda x: x[perm], state)
    pos = pos[perm]

    def per_particle(
        idx: jax.Array, pos_i: jax.Array, stencil: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if system.domain.periodic:
            stencil = _dedup_stencil_hashes(stencil)

        return traverse_fn(idx, pos_i, stencil, state, system, pos, p_cell_hash)

    neighbor_list, overflows = jax.vmap(per_particle)(iota, pos, p_neighbor_hashes)
    return state, system, neighbor_list, jnp.any(overflows)


@partial(jax.named_call, name="cell_list._compute_cross_neighbor_list_common")
def _compute_cross_neighbor_list_common(
    pos_a: jax.Array,
    pos_b: jax.Array,
    system: System,
    traverse_fn: Callable[..., Any],
    max_neighbors: int,
) -> tuple[jax.Array, jax.Array]:
    """Common logic for creating cross-neighbor lists using a Cell List.

    Hashes ``pos_b`` into cells and, for each point in ``pos_a``, searches
    the neighboring cells in the sorted ``pos_b`` array.

    Parameters
    ----------
    pos_a : jax.Array
        Query positions, shape ``(N_A, dim)``.
    pos_b : jax.Array
        Database positions, shape ``(N_B, dim)``.
    system : System
        The system configuration.
    traverse_fn : Callable
        A function that finds neighbors for a single query point.
        Signature: ``(pos_ai, stencil, n_b, system, pos_b_sorted, p_cell_hash_b)
        -> (neighbor_list, overflow)``
    max_neighbors : int
        Maximum number of neighbors to store per query point.

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        ``(neighbor_list, overflow_flag)``

    """
    n_a = pos_a.shape[0]
    if max_neighbors == 0:
        empty = jnp.empty((n_a, 0), dtype=int)
        return empty, jnp.asarray(False)

    collider = cast(Union["StaticCellList", "DynamicCellList"], system.collider)
    n_b = pos_b.shape[0]
    iota_a = jax.lax.iota(int, n_a)
    iota_b = jax.lax.iota(int, n_b)

    # 1. Hash & sort B (the "database")
    (
        perm_b,
        _,
        p_cell_hash_b,
        _,
        _,
    ) = _get_spatial_partition(
        pos_b, system, collider.cell_size, collider.neighbor_mask, iota_b
    )
    pos_b_sorted = pos_b[perm_b]

    # 2. Hash A (the "queries") to obtain stencil hashes
    (
        perm_a,
        _,
        _,
        _,
        p_neighbor_hashes_a,
    ) = _get_spatial_partition(
        pos_a, system, collider.cell_size, collider.neighbor_mask, iota_a
    )
    pos_a_sorted = pos_a[perm_a]

    # 3. For each sorted-A point, find neighbors in sorted B
    def per_query(pos_ai: jax.Array, stencil: jax.Array) -> tuple[jax.Array, jax.Array]:
        stencil = _dedup_stencil_hashes(stencil)
        return traverse_fn(pos_ai, stencil, n_b, system, pos_b_sorted, p_cell_hash_b)

    neighbor_list, overflows = jax.vmap(per_query)(pos_a_sorted, p_neighbor_hashes_a)

    # 4. Map sorted-B indices back to original B indices
    valid_mask = neighbor_list != -1
    safe_indices = jnp.where(valid_mask, neighbor_list, 0)
    neighbor_list = jnp.where(valid_mask, perm_b[safe_indices], -1)

    # 5. Unsort from sorted-A order back to original A order
    inv_perm_a = jnp.empty_like(perm_a)
    inv_perm_a = inv_perm_a.at[perm_a].set(iota_a)
    neighbor_list = neighbor_list[inv_perm_a]

    return neighbor_list, jnp.any(overflows)


def _force_kernel(
    idx: jax.Array,
    k: jax.Array,
    valid: jax.Array,
    pos: jax.Array,
    state: State,
    system: System,
) -> tuple[jax.Array, jax.Array]:
    """Common interaction kernel for force computation.

    Parameters
    ----------
    idx : jax.Array
        Index of the target particle.
    k : jax.Array
        Indices of the neighbor particles.
    valid : jax.Array
        Boolean mask indicating valid interactions.
    pos : jax.Array
        Array of particle positions.
    state : State
        The current state.
    system : System
        The system configuration.

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        Accumulated force and torque vectors.

    """
    f, t = system.force_model.force(idx, k, pos, state, system)
    # Handle broadcasting for valid mask (k can be scalar or vector)
    mask = valid if valid.ndim == 0 else valid[:, None]
    return f * mask, t * mask


def _energy_kernel(
    idx: jax.Array,
    k: jax.Array,
    valid: jax.Array,
    pos: jax.Array,
    state: State,
    system: System,
) -> jax.Array:
    """Common interaction kernel for potential energy computation.

    Parameters
    ----------
    idx : jax.Array
        Index of the target particle.
    k : jax.Array
        Indices of the neighbor particles.
    valid : jax.Array
        Boolean mask indicating valid interactions.
    pos : jax.Array
        Array of particle positions.
    state : State
        The current state.
    system : System
        The system configuration.

    Returns
    -------
    jax.Array
        Potential energy contribution.

    """
    e = system.force_model.energy(idx, k, pos, state, system)
    return 0.5 * e * valid


def _force_reduce(
    res: tuple[jax.Array, jax.Array], pos_pi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Common reduction for force/torque accumulation.

    Parameters
    ----------
    res : Tuple[jax.Array, jax.Array]
        The summed (force, torque) from pairwise interactions.
    pos_pi : jax.Array
        The vector from the center of mass to the particle surface point.

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        The final reduced (force, torque) including cross-product contributions.

    """
    sum_f, sum_t = res
    sum_t += cross(pos_pi, sum_f)
    return sum_f, sum_t


def _static_traverse_cell(
    target_hash: jax.Array,
    state: State,
    system: System,
    pos: jax.Array,
    p_cell_hash: jax.Array,
    idx: jax.Array,
    interaction_fn: Callable[..., Any],
    max_occupancy: int,
) -> Any:
    """Traversal strategy for StaticCellList (unrolled fixed-size loop).

    Parameters
    ----------
    target_hash : jax.Array
        The hash of the cell to traverse.
    state : State
        The current state.
    system : System
        The system configuration.
    pos : jax.Array
        Array of particle positions.
    p_cell_hash : jax.Array
        Sorted array of all particle cell hashes.
    idx : jax.Array
        Index of the target particle.
    interaction_fn : Callable
        The function to compute the interaction (force or energy).
    max_occupancy : int
        Maximum number of particles per cell to check.

    Returns
    -------
    Any
        The accumulated result from the interaction function.

    """
    start_idx = jnp.searchsorted(
        p_cell_hash,
        target_hash,
        side="left",
        method="scan_unrolled",
    )

    k_indices = start_idx + jax.lax.iota(dtype=int, size=max_occupancy)
    safe_k = jnp.minimum(k_indices, state.N - 1)

    # Validity mask: index bounds, correct cell hash, and logic mask
    valid = (
        (k_indices < state.N)
        * (p_cell_hash[safe_k] == target_hash)
        * valid_interaction_mask(
            state.clump_id[safe_k],
            state.clump_id[idx],
            state.bond_id[safe_k],
            state.bond_id[idx],
            system.interact_same_bond_id,
        )
    )

    res = interaction_fn(idx, safe_k, valid, pos, state, system)
    # Perform summation here to align with Dynamic output (which is pre-summed)
    return jax.tree.map(lambda x: jnp.sum(x, axis=0), res)


def _dynamic_traverse_cell(
    target_hash: jax.Array,
    state: State,
    system: System,
    pos: jax.Array,
    p_cell_hash: jax.Array,
    idx: jax.Array,
    interaction_fn: Callable[..., Any],
    init_val: Any,
) -> Any:
    """Traversal strategy for DynamicCellList (while_loop).

    Parameters
    ----------
    target_hash : jax.Array
        The hash of the cell to traverse.
    state : State
        The current state.
    system : System
        The system configuration.
    pos : jax.Array
        Array of particle positions.
    p_cell_hash : jax.Array
        Sorted array of all particle cell hashes.
    idx : jax.Array
        Index of the target particle.
    interaction_fn : Callable
        The function to compute the interaction (force or energy).
    init_val : Any
        Initial value for the accumulation.

    Returns
    -------
    Any
        The accumulated result from the interaction function.

    """
    start_idx = jnp.searchsorted(
        p_cell_hash, target_hash, side="left", method="scan_unrolled"
    )

    def cond_fun(val: tuple[jax.Array, Any]) -> bool:
        k, _ = val
        return cast(bool, (k < state.N) * (p_cell_hash[k] == target_hash))

    def body_fun(val: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
        k, acc = val
        valid = valid_interaction_mask(
            state.clump_id[k],
            state.clump_id[idx],
            state.bond_id[k],
            state.bond_id[idx],
            system.interact_same_bond_id,
        )
        res = interaction_fn(idx, k, valid, pos, state, system)

        # Accumulate results (works for scalar or tree)
        new_acc = jax.tree.map(lambda a, b: a + b, acc, res)
        return k + 1, new_acc

    _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (start_idx, init_val))
    return final_acc


@Collider.register("StaticCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class StaticCellList(Collider):
    r"""Implicit cell-list (spatial hashing) collider.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid of cubic/square cells of side length ``cell_size``.
    Each particle is assigned to a cell, particles are sorted by cell hash, and
    interactions are evaluated only against particles in the same or neighboring
    cells given by ``neighbor_mask``. The cell list is *implicit* because we never
    store per-cell particle lists explicitly; instead, we exploit the sorted hashes
    and fixed ``max_occupancy`` to probe neighbors in-place.

    The operation of this collider can be understood as the following nested loop:

    .. code-block:: python

        for particle in particles:
            for hash in stencil(particle):
                for i in range(max_occupancy):
                    ...

    The loop is flattened and everything happens in parallel. We can do this because
    the stencil size and ``max_occupancy`` are static values. This means that the cost
    of the calculations done with this cell list is:

    .. math::
        O(N \cdot \text{neighbor\_mask\_size} \cdot \text{max\_occupancy})

    Plus the cost of hashing :math:`O(N)` and sorting :math:`O(N \log N)`. As we sort
    every time step, the system remains mainly sorted, reducing the practical sorting
    complexity to something closer to :math:`O(N)`. This fixed ``max_occupancy`` makes
    the cell list ideal for certain types of systems but very bad for others. To
    understand the difference, let's analyze the cost components:

    * Stencil size:
        The stencil size depends on the ratio between the cell size (:math:`L`) and
        the radius of the largest particle (:math:`r_{max}`).

        .. math::
            \text{neighbor\_mask\_size} = \left( 2\left\lceil \frac{2r_{max}}{L} \right\rceil + 1 \right)^{dim}

    * Max occupancy:
        The maximum number of particles that can occupy a cell depends on the cell volume,
        the density (represented as local volume fraction :math:`\phi`), and the volume of
        the smallest particle (:math:`V_{min}`). To prevent missing contacts during local
        density fluctuations, we estimate the expected occupancy (:math:`\lambda`) and add
        a safety margin of 3 standard deviations:

        .. math::
            \text{max\_occupancy} = \left\lceil \lambda + 3\sqrt{\lambda} \right\rceil \quad \text{where} \quad \lambda = \phi \frac{L^{dim}}{V_{min}}

        Here, :math:`\phi` is the local volume fraction, representing the ratio of the
        volume actually occupied by particles to the total volume of the cell. For dense
        packings of spheres, :math:`\phi \approx 0.74`, but in systems with highly
        overlapping internal spheres (like rigid clumps), :math:`\phi` can be much larger.

    Putting this together and expressing the cell size in units of the maximum radius
    :math:`L^\prime = L/r_{max}`, we find the total theoretical cost to be:

    .. math::
        \text{cost} \approx N \left( 2\left\lceil \frac{2}{L^\prime} \right\rceil + 1 \right)^{dim} \left\lceil \phi \frac{L^{dim}}{V_{min}} + 3\sqrt{\phi \frac{L^{dim}}{V_{min}}} \right\rceil

    Then, if we define polydispersity as the ratio between the largest and smallest particle
    :math:`\alpha = r_{max}/r_{min}`, we can express :math:`\lambda` solely in terms of these
    dimensionless parameters. Knowing that the volume of the smallest particle is
    :math:`V_{min} = k_v r_{min}^{dim}` (where :math:`k_v` is the geometric volume factor,
    such as :math:`4\pi/3` in 3D or :math:`\pi` in 2D), we can write :math:`\lambda` as:

    .. math::
        \lambda = \frac{\phi}{k_v} (\alpha L^\prime)^{dim}

    Substituting this back, we find the final cost function:

    .. math::
        \text{cost} \approx N \left( 2\left\lceil \frac{2}{L^\prime} \right\rceil + 1 \right)^{dim} \left\lceil \frac{\phi}{k_v} (\alpha L^\prime)^{dim} + 3\sqrt{\frac{\phi}{k_v}} (\alpha L^\prime)^{dim/2} \right\rceil

    * The Polydispersity Penalty:
        Because we must safely bound the maximum possible number of particles in a cell,
        we base :math:`\lambda` on the *smallest* particle. As shown in the equation above,
        the required array padding grows dramatically as :math:`O(\alpha^{dim})`,
        causing performance to degrade in highly polydisperse systems.

    This collider is ideal for systems of spheres with minimum polydispersity and no
    dramatic overlaps. In those cases, it might be even faster than the dynamic cell list.
    However, it's not recommended for systems with clumps (where internal overlaps cause
    extreme local :math:`\phi`) or high polydispersity, as both drastically inflate the
    required ``max_occupancy`` padding.

    Complexity
    ----------
    - Time: :math:`O(N)` - :math:`O(N \log N)` from sorting, plus :math:`O(N \cdot M \cdot K)`
      for neighbor probing (M = ``neighbor_mask_size``, K = ``max_occupancy``).
      The state is close to sorted every frame.
    - Memory: :math:`O(N)`.

    Attributes
    ----------
    cell_size : jax.Array
        Linear size of a grid cell (scalar).
    max_occupancy : int
        Maximum number of particles assumed to occupy a single cell. The algorithm
        probes exactly ``max_occupancy`` entries starting from the first particle
        in a neighbor cell. This should be set high enough that real cells rarely
        exceed it; otherwise, contacts and energy will be undercounted.

    Notes
    -----
    - ``max_occupancy`` is an upper bound on particles per cell.
      If a cell contains more than this many particles, some interactions
      might be missed (you should carefully choose ``cell_size`` and
      ensure your local density does not exceed your expected ``max_occupancy``).
    """

    neighbor_mask: jax.Array
    """
    Integer offsets defining the neighbor stencil.

    Shape is ``(M, dim)``, where each row is a displacement in cell coordinates.
    For ``search_range=1`` in 2D this is the 3×3 Moore neighborhood (M=9);
    in 3D this is the 3×3×3 neighborhood (M=27).
    """

    cell_size: jax.Array
    """
    Linear size of a grid cell (scalar).
    """

    max_occupancy: int = jax.tree.static()  # type: ignore[attr-defined]
    """
    Maximum number of particles assumed to occupy a single cell.

    The algorithm probes exactly ``max_occupancy`` entries starting from the
    first particle in a neighbor cell. This should be set high enough that
    real cells rarely exceed it; otherwise contacts/energy will be undercounted.
    """

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        search_range: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
        max_occupancy: int | None = None,
    ) -> Self:
        r"""Creates a StaticCellList collider with robust defaults.

        Defaults are chosen to prevent missing contacts while keeping the
        neighbor stencil and assumed cell occupancy as small as possible
        given the available information from ``state``.

        The optimal cell size parameter, denoted here as the dimensionless
        ratio :math:`L^\prime = L/r_{max}`, depends heavily on the system's
        volume fraction (:math:`\phi`) and polydispersity
        (:math:`\alpha = r_{max}/r_{min}`):

        * **Standard density (** :math:`\phi \le 1` **):** The optimal cell size is
          typically :math:`L^\prime = 2` (the diameter of the largest particle).
          This minimizes the search stencil to just the immediate neighboring cells.
        * **Extreme local density (** :math:`\phi \gg 1` **):** For systems with
          heavy internal overlaps (like rigid clumps), the optimal cell size shrinks.
          Values like :math:`L^\prime = 0.5` or :math:`L^\prime = 0.25` often yield
          better performance by reducing the massive array padding penalty, even
          at the cost of a larger search stencil.
        * **High polydispersity (** :math:`\alpha \gg 1` **):** High polydispersity
          severely degrades performance regardless of density, because the fixed
          occupancy arrays must always be padded to accommodate the volume of the
          smallest particles.

        By default, if ``cell_size`` or ``max_occupancy`` are not provided, this
        method infers optimal safe values based on the radius distribution in the
        reference state, assuming a maximum local volume fraction of :math:`\phi = 1`.
        If your system contains clumps with internal overlaps where local :math:`\phi > 1`,
        you **must** override these defaults.

        Parameters
        ----------
        state : State
            Reference state used to determine spatial dimension and default parameters.
        cell_size : float, optional
            Cell edge length. If None, defaults to :math:`2 r_{max}` for systems with
            low polydispersity (:math:`\alpha < 2.5`), or :math:`0.5 r_{max}` for
            highly polydisperse systems to mitigate exponential array padding costs.
        search_range : int, optional
            Neighbor range in cell units. If None, the smallest safe value is
            computed such that :math:`\text{search\_range} \cdot L \geq \text{cutoff}`.
        box_size : jax.Array, optional
            Size of the periodic box used to ensure there are at least
            ``2 * search_range + 1`` cells per axis. If None, these checks are
            skipped.
        max_occupancy : int, optional
            Assumed maximum particles per cell. If None, estimated using the
            statistical model: :math:`\lambda + 3\sqrt{\lambda}`, assuming a worst-case
            standard granular density of :math:`\phi = 1` and the volume of the
            smallest particle.

        Returns
        -------
        CellList
            Configured collider instance.
        """
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
        alpha = max_rad / min_rad

        if cell_size is None:
            # Default to L' = 2 for standard packing.
            # If polydispersity is high, shift to L' = 0.5 to mitigate the O(alpha^dim) padding cost.
            cell_size = 2.0 * max_rad if alpha < 2.5 else 0.5 * max_rad

        # make sure that the stencil fits in the box, if provided
        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            for _ in range(2):  # decreasing cell_size changes search_range, so iterate
                if search_range is None:
                    sr = jnp.ceil(2 * max_rad / cell_size).astype(int)
                    sr = jnp.maximum(1, sr)
                else:
                    sr = jnp.asarray(search_range, dtype=int)
                min_grids_per_axis = 2 * sr + 1
                grid_dims = jnp.floor(box_size / cell_size).astype(int)
                grid_dims = jnp.maximum(grid_dims, min_grids_per_axis)
                cell_size = jnp.min(box_size / grid_dims)

        if search_range is None:
            search_range = jnp.ceil(2 * max_rad / cell_size).astype(int)
            search_range = jnp.maximum(1, search_range)
        search_range = jnp.array(search_range, dtype=int)

        if max_occupancy is None:
            cell_vol = cell_size**state.dim
            smallest_sphere_vol = jnp.array(0.0, dtype=float)

            if state.dim == 3:
                smallest_sphere_vol = (4.0 / 3.0) * jnp.pi * min_rad**3
            elif state.dim == 2:
                smallest_sphere_vol = jnp.pi * min_rad**2

            # Assume geometric maximum filling (phi = 1.0) without dramatic overlaps
            phi_assumed = 1.0
            expected_occ = phi_assumed * (cell_vol / smallest_sphere_vol)

            # Add 3 standard deviations to safely cover local density fluctuations
            statistical_max = expected_occ + 3.0 * jnp.sqrt(expected_occ)
            max_occupancy = int(jnp.maximum(1, jnp.ceil(statistical_max)))

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
            max_occupancy=int(max_occupancy),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="StaticCellList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
        r"""Computes the total force acting on each particle using an implicit cell list :math:`O(N log N)`.
        This method sums the force contributions from all particle pairs (i, j)
        as computed by the ``system.force_model`` and updates the particle forces.

        Parameters
        ----------
        state : State
            The current state of the simulation.

        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated ``State`` object with computed forces and the unmodified ``System`` object.

        """
        collider = cast(StaticCellList, system.collider)

        traverse = partial(_static_traverse_cell, max_occupancy=collider.max_occupancy)

        state, (state.force, state.torque) = _compute_interaction(
            state, system, traverse, _force_kernel, _force_reduce
        )
        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="StaticCellList.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        r"""Computes the potential energy acting on each particle using an implicit cell list :math:`O(N log N)`.
        This method sums the energy contributions from all particle pairs (i, j)
        as computed by the ``system.force_model``.

        Parameters
        ----------
        state : State
            The current state of the simulation.

        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            Scalar containing the total potential energy of the system.

        """
        collider = cast(StaticCellList, system.collider)

        traverse = partial(_static_traverse_cell, max_occupancy=collider.max_occupancy)

        _, energy = _compute_interaction(state, system, traverse, _energy_kernel)
        return jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="StaticCellList.create_neighbor_list")
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        r"""Computes the list of neighbors for each particle. The shape of the list is (N, max_neighbors).
        If a particle has less neighbors than max_neighbors, the list is padded with -1. The indices of the list
        correspond to the indices of the returned sorted state.

        Note that no neighbors further than cell_size * (1 + search_range) (how many neighbors to check in the cells)
        can be found due to the nature of the cell list. If cutoff is greater than this value, the list might not
        return the expected list. Note that if a cell contains more spheres than those specified in max_occupancy, there might be missing neighbors.

        Parameters
        ----------
        state : State
            The current state of the simulation.

        system : System
            The configuration of the simulation.

        cutoff : float
            Search radius

        max_neighbors : int
            Maximum number of neighbors to store per particle.

        Returns
        -------
        tuple[State, System, jax.Array, jax.Array]
            The sorted state, the system, the neighbor list, and a boolean flag for overflow.

        """
        collider = cast(StaticCellList, system.collider)
        MAX_OCCUPANCY = collider.max_occupancy
        cutoff_sq = cutoff**2

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            stencil: jax.Array,
            state: State,
            system: System,
            pos: jax.Array,
            p_cell_hash: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            starts = jnp.searchsorted(
                p_cell_hash,
                stencil,
                side="left",
                method="scan_unrolled",
            )

            k_indices = (starts[:, None] + jax.lax.iota(int, MAX_OCCUPANCY)).reshape(-1)
            safe_k = jnp.minimum(k_indices, state.N - 1)

            dr = system.domain.displacement(pos_i, pos[safe_k], system)
            dist_sq = norm2(dr)

            valid = (
                (k_indices < state.N)
                * (p_cell_hash[safe_k] == jnp.repeat(stencil, MAX_OCCUPANCY))
                * (jnp.repeat(stencil, MAX_OCCUPANCY) != -1)
                * valid_interaction_mask(
                    state.clump_id[safe_k],
                    state.clump_id[idx],
                    state.bond_id[safe_k],
                    state.bond_id[idx],
                    system.interact_same_bond_id,
                )
                * (dist_sq <= cutoff_sq)
            )
            num_neighbors = jnp.sum(valid)
            overflow_flag = num_neighbors > max_neighbors

            candidates = jnp.where(valid, safe_k, -1)
            k_eff = min(max_neighbors, candidates.shape[0])
            topk = jax.lax.top_k(candidates, k_eff)[0]
            if k_eff < max_neighbors:
                topk = jnp.concatenate(
                    [topk, jnp.full((max_neighbors - k_eff,), -1, dtype=topk.dtype)]
                )
            return topk, overflow_flag

        return _compute_neighbor_list_common(state, system, traverse, max_neighbors)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="StaticCellList.create_cross_neighbor_list")
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Build a cross-neighbor list between two sets of positions using an implicit cell list.

        For each point in ``pos_a``, finds all neighbors from ``pos_b`` within the
        given ``cutoff`` distance. The ``pos_b`` array is hashed and sorted into cells,
        and the neighbor stencil of each query point in ``pos_a`` is used to probe the
        sorted ``pos_b`` hashes with a fixed-size unrolled loop.

        No neighbors further than ``cell_size * (1 + search_range)`` can be found due to
        the nature of the cell list. If a cell contains more particles than
        ``max_occupancy``, some neighbors may be missed.

        Parameters
        ----------
        pos_a : jax.Array
            Query positions, shape ``(N_A, dim)``.
        pos_b : jax.Array
            Database positions, shape ``(N_B, dim)``.
        system : System
            The configuration of the simulation.
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
        collider = cast(StaticCellList, system.collider)
        MAX_OCCUPANCY = collider.max_occupancy
        cutoff_sq = cutoff**2

        def traverse(
            pos_ai: jax.Array,
            stencil: jax.Array,
            n_b: int,
            system: System,
            pos_b_sorted: jax.Array,
            p_cell_hash_b: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            starts = jnp.searchsorted(
                p_cell_hash_b,
                stencil,
                side="left",
                method="scan_unrolled",
            )

            k_indices = (starts[:, None] + jax.lax.iota(int, MAX_OCCUPANCY)).reshape(-1)
            safe_k = jnp.minimum(k_indices, n_b - 1)

            dr = system.domain.displacement(pos_ai, pos_b_sorted[safe_k], system)
            dist_sq = norm2(dr)

            valid = (
                (k_indices < n_b)
                * (p_cell_hash_b[safe_k] == jnp.repeat(stencil, MAX_OCCUPANCY))
                * (jnp.repeat(stencil, MAX_OCCUPANCY) != -1)
                * (dist_sq <= cutoff_sq)
            )
            num_neighbors = jnp.sum(valid)
            overflow_flag = num_neighbors > max_neighbors

            candidates = jnp.where(valid, safe_k, -1)
            k_eff = min(max_neighbors, candidates.shape[0])
            topk = jax.lax.top_k(candidates, k_eff)[0]
            if k_eff < max_neighbors:
                topk = jnp.concatenate(
                    [topk, jnp.full((max_neighbors - k_eff,), -1, dtype=topk.dtype)]
                )
            return topk, overflow_flag

        return _compute_cross_neighbor_list_common(
            pos_a, pos_b, system, traverse, max_neighbors
        )


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicCellList(Collider):
    r"""Implicit cell-list (spatial hashing) collider using dynamic while-loops.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid of cubic/square cells of side length ``cell_size``.
    Each particle is assigned to a cell, particles are sorted by cell hash, and
    interactions are evaluated only against particles in the same or neighboring
    cells given by ``neighbor_mask``.

    Unlike the static cell list, this implementation does not use a fixed
    ``max_occupancy`` array padding. Instead, it uses a dynamic ``jax.lax.while_loop``
    to iterate over the exact number of particles present in each neighboring cell.

    The operation of this collider can be understood as the following nested loop:

    .. code-block:: python

        for particle in particles: # parallel
            for hash in stencil(particle): # parallel
                while next_neighbor in cell(hash): # sequential
                    ...

    Because the innermost loop is evaluated sequentially, the computational cost
    is driven by the *average* cell occupancy rather than the maximum possible
    occupancy. This makes the total theoretical cost:

    .. math::
        O(N \cdot \text{neighbor\_mask\_size} \cdot \langle K \rangle)

    where :math:`\langle K \rangle` is the average cell occupancy. To understand
    how this scales, let's analyze the cost components:

    * Stencil size:
        The stencil size depends on the ratio between the cell size (:math:`L`) and
        the radius of the largest particle (:math:`r_{max}`).

        .. math::
            \text{neighbor\_mask\_size} = \left( 2\left\lceil \frac{2r_{max}}{L} \right\rceil + 1 \right)^{dim}

    * Average occupancy:
        The average number of particles that occupy a cell depends only on the cell
        volume and the macroscopic number density (:math:`\rho`), completely
        independent of the smallest particle size.

        .. math::
            \langle K \rangle = \rho L^{dim}

    To express this in terms of the local volume fraction :math:`\phi` (the ratio of
    volume actually occupied by particles to the total cell volume) and our
    normalized cell size :math:`L^\prime = L/r_{max}`, we use the average particle
    volume :math:`\langle V \rangle`:

    .. math::
        \langle K \rangle = \phi \frac{L^{dim}}{\langle V \rangle} = \phi \frac{(L^\prime r_{max})^{dim}}{\langle V \rangle}

    Knowing that the volume of the largest particle is :math:`V_{max} = k_v r_{max}^{dim}`
    (where :math:`k_v` is the geometric volume factor, such as :math:`4\pi/3` in 3D or
    :math:`\pi` in 2D), we find the final theoretical cost:

    .. math::
        \text{cost} \approx N \left( 2\left\lceil \frac{2}{L^\prime} \right\rceil + 1 \right)^{dim} \left( \frac{\phi}{k_v} \frac{V_{max}}{\langle V \rangle} (L^\prime)^{dim} \right)

    * The Polydispersity Advantage:
        In the static cell list, cost scales with the ratio of the largest to smallest
        particle volume (:math:`V_{max}/V_{min} \propto \alpha^{dim}`, where
        :math:`\alpha = r_{max}/r_{min}`). In this dynamic list, the cost scales with
        the ratio of the largest to the *average* particle volume
        (:math:`V_{max}/\langle V \rangle`). Because adding tiny particles
        barely changes the average volume, the severe :math:`O(\alpha^{dim})` padding
        penalty is significantly reduced or offset. The dynamic loop only iterates over
        particles that actually exist.

    This collider is ideal for highly polydisperse systems, sparse systems
    (low packing fractions), or systems with rigid clumps that create massive
    local density spikes, as it completely avoids the memory bloat and wasted
    gather operations caused by array padding.

    Complexity
    ----------
    - Time: :math:`O(N)` - :math:`O(N \log N)` from sorting, plus :math:`O(N \cdot M \cdot \langle K \rangle)`
      for neighbor probing (M = ``neighbor_mask_size``, :math:`\langle K \rangle` = average occupancy).
      The state is close to sorted every frame.
    - Memory: :math:`O(N)`.

    Attributes
    ----------
    neighbor_mask : jax.Array
        Integer offsets defining the neighbor stencil (M, dim).
    cell_size : jax.Array
        Linear size of a grid cell (scalar).

    Notes
    -----
    - **Batching with ``vmap``**: If you use ``jax.vmap`` to evaluate multiple
      simulation environments simultaneously, be aware of JAX's SIMD execution model.
      Because the innermost ``while`` loop executes sequentially, the loop must continue
      running for *all* environments in the batch until the environment with the highest
      local cell occupancy finishes its iterations. Consequently, the computational cost
      of a batched execution is bottlenecked by the single worst-case occupancy across
      the entire batch.
    """

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a grid cell (scalar)."""

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        search_range: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
    ) -> Self:
        r"""Creates a CellList collider with robust defaults.

        Defaults are chosen to prevent missing contacts while keeping the
        neighbor stencil and assumed cell occupancy as small as possible
        given the available information from ``state``.

        Because this collider uses a dynamic while-loop, its optimal parameters
        differ slightly from the static implementation. The optimal cell size
        parameter, denoted here as the dimensionless ratio :math:`L^\prime = L/r_{max}`,
        is primarily driven by balancing the stencil size overhead against the
        sequential loop cost.

        * **Standard density:** The optimal cell size is typically :math:`L^\prime = 2`
          (the diameter of the largest particle). This minimizes the search stencil
          to just the immediate 27 neighboring cells (in 3D), which is usually the
          most efficient balance for JAX compilations.
        * **High polydispersity (** :math:`\alpha \gg 1` **):** Unlike the static cell
          list, the dynamic cell list handles high polydispersity gracefully.
          We generally maintain :math:`L^\prime = 2` unless the size ratio is extreme,
          as shrinking the cell size drastically inflates the neighbor stencil, which
          harms the parallelized outer loops.

        By default, if ``cell_size`` is not provided, this method will infer an
        optimal value based on the radius distribution in the reference state.

        Parameters
        ----------
        state : State
            Reference state used to determine spatial dimension and default parameters.
        cell_size : float, optional
            Cell edge length. If None, defaults to :math:`2 r_{max}` for systems with
            low polydispersity (:math:`\alpha < 2.5`), or :math:`0.5 r_{max}` for
            highly polydisperse systems to balance stencil overhead.
        search_range : int, optional
            Neighbor range in cell units. If None, the smallest safe value is
            computed such that :math:`\text{search\_range} \cdot L \geq \text{cutoff}`.
        box_size : jax.Array, optional
            Size of the periodic box used to ensure there are at least
            ``2 * search_range + 1`` cells per axis. If None, these checks are
            skipped.

        Returns
        -------
        CellList
            Configured collider instance.
        """
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
        alpha = max_rad / min_rad

        if cell_size is None:
            cell_size = 2.0 * max_rad if alpha < 2.5 else 0.5 * max_rad

        # make sure that the stencil fits in the box, if provided
        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            for _ in range(2):  # decreasing cell_size changes search_range, so iterate
                if search_range is None:
                    sr = jnp.ceil(2 * max_rad / cell_size).astype(int)
                    sr = jnp.maximum(1, sr)
                else:
                    sr = jnp.asarray(search_range, dtype=int)
                min_grids_per_axis = 2 * sr + 1
                grid_dims = jnp.floor(box_size / cell_size).astype(int)
                grid_dims = jnp.maximum(grid_dims, min_grids_per_axis)
                cell_size = jnp.min(box_size / grid_dims)

        if search_range is None:
            search_range = jnp.ceil(2 * max_rad / cell_size).astype(int)
            search_range = jnp.maximum(1, search_range)
        search_range = jnp.array(search_range, dtype=int)

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicCellList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
        r"""Computes the total force acting on each particle using an implicit cell list :math:`O(N log N)`.
        This method sums the force contributions from all particle pairs (i, j)
        as computed by the ``system.force_model`` and updates the particle forces.

        Parameters
        ----------
        state : State
            The current state of the simulation.

        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated ``State`` object with computed forces and the unmodified ``System`` object.

        """
        zero_f = (
            jnp.zeros(state.dim, dtype=float),
            jnp.zeros(1 if state.dim == 2 else 3, dtype=float),
        )
        traverse = partial(_dynamic_traverse_cell, init_val=zero_f)

        state, (state.force, state.torque) = _compute_interaction(
            state, system, traverse, _force_kernel, _force_reduce
        )
        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicCellList.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        r"""Computes the potential energy acting on each particle using an implicit cell list :math:`O(N log N)`.
        This method sums the energy contributions from all particle pairs (i, j)
        as computed by the ``system.force_model``.

        Parameters
        ----------
        state : State
            The current state of the simulation.

        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            Scalar containing the total potential energy of the system.

        """
        cast(DynamicCellList, system.collider)

        traverse = partial(_dynamic_traverse_cell, init_val=jnp.array(0.0, dtype=float))

        _, energy = _compute_interaction(state, system, traverse, _energy_kernel)
        return jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        r"""Computes the list of neighbors for each particle. The shape of the list is (N, max_neighbors).
        If a particle has less neighbors than max_neighbors, the list is padded with -1. The indices of the list
        correspond to the indices of the returned sorted state.

        Note that no neighbors further than cell_size * (1 + search_range) (how many neighbors to check in the cells)
        can be found due to the nature of the cell list. If cutoff is greater than this value, the list might not
        return the expected list.

        Parameters
        ----------
        state : State
            The current state of the simulation.

        system : System
            The configuration of the simulation.

        cutoff : float
            Search radius

        max_neighbors : int
            Maximum number of neighbors to store per particle.

        Returns
        -------
        tuple[State, System, jax.Array, jax.Array]
            The sorted state, the system, the neighbor list, and a boolean flag for overflow.

        """
        cutoff_sq = cutoff**2

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            stencil: jax.Array,
            state: State,
            system: System,
            pos: jax.Array,
            p_cell_hash: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            cell_starts = jnp.searchsorted(
                p_cell_hash, stencil, side="left", method="scan_unrolled"
            )

            def stencil_body(
                target_cell_hash: jax.Array, start_idx: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                local_capacity = max_neighbors // 2 + 1
                init_carry = (
                    start_idx,
                    jnp.array(0, dtype=int),
                    jnp.full((local_capacity,), -1, dtype=int),
                    jnp.array(False),  # overflow flag
                )

                def cond_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> bool:
                    k, c, _, _ = val
                    in_cell = (k < state.N) * (p_cell_hash[k] == target_cell_hash)
                    has_space = c < local_capacity + 1  # so we can catch the overflow
                    return cast(bool, in_cell * has_space)

                def body_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    k, c, nl, overflow = val
                    dr = system.domain.displacement(pos_i, pos[k], system)
                    d_sq = norm2(dr)
                    valid = valid_interaction_mask(
                        state.clump_id[k],
                        state.clump_id[idx],
                        state.bond_id[k],
                        state.bond_id[idx],
                        system.interact_same_bond_id,
                    ) * (d_sq <= cutoff_sq)
                    nl = jax.lax.cond(
                        valid,
                        lambda nl_: nl_.at[c].set(k, mode="drop"),
                        lambda nl_: nl_,
                        nl,
                    )
                    c_new = c + valid.astype(c.dtype)
                    return k + 1, c_new, nl, overflow + (c_new > local_capacity)

                _, local_c, local_nl, local_overflow = jax.lax.while_loop(
                    cond_fun, body_fun, init_carry
                )
                return local_nl, local_c, local_overflow

            final_n_list, stencil_counts, stencil_overflows = jax.vmap(stencil_body)(
                stencil, cell_starts
            )
            row_offsets = jnp.cumsum(stencil_counts) - stencil_counts
            local_iota = jnp.arange(final_n_list.shape[1])
            target_indices = row_offsets[:, None] + local_iota[None, :]
            # 1. Simplified Masking
            #    We only need to identify "Local Padding" (invalid entries within the stencil rows).
            valid_mask = local_iota[None, :] < stencil_counts[:, None]
            # 2. Direct Index Selection
            #    If valid: use the calculated target index.
            #    If invalid: redirect to 'max_neighbors' (which is Out-Of-Bounds).
            safe_indices = jnp.where(
                valid_mask.flatten(), target_indices.flatten(), max_neighbors
            )
            # 3. Scatter with Implicit Bounds Check
            #    mode='drop' automatically ignores any writes where index >= max_neighbors.
            result = jnp.full((max_neighbors,), -1, dtype=final_n_list.dtype)
            final_n_list = result.at[safe_indices].set(
                final_n_list.flatten(), mode="drop"
            )
            # Propagate overflow if any stencil cell overflowed or total count exceeds max_neighbors
            overflow_flag = (
                jnp.sum(stencil_overflows)
                > 0 + (jnp.sum(stencil_counts) > max_neighbors)
            ).astype(bool)
            return final_n_list, overflow_flag

        return _compute_neighbor_list_common(state, system, traverse, max_neighbors)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="DynamicCellList.create_cross_neighbor_list")
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Build a cross-neighbor list between two sets of positions using a dynamic cell list.

        For each point in ``pos_a``, finds all neighbors from ``pos_b`` within the
        given ``cutoff`` distance. The ``pos_b`` array is hashed and sorted into cells,
        and the neighbor stencil of each query point in ``pos_a`` is used to probe the
        sorted ``pos_b`` hashes with a dynamic ``jax.lax.while_loop``.

        No neighbors further than ``cell_size * (1 + search_range)`` can be found due to
        the nature of the cell list.

        Parameters
        ----------
        pos_a : jax.Array
            Query positions, shape ``(N_A, dim)``.
        pos_b : jax.Array
            Database positions, shape ``(N_B, dim)``.
        system : System
            The configuration of the simulation.
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
        cutoff_sq = cutoff**2

        def traverse(
            pos_ai: jax.Array,
            stencil: jax.Array,
            n_b: int,
            system: System,
            pos_b_sorted: jax.Array,
            p_cell_hash_b: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            cell_starts = jnp.searchsorted(
                p_cell_hash_b, stencil, side="left", method="scan_unrolled"
            )

            def stencil_body(
                target_cell_hash: jax.Array, start_idx: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                local_capacity = max_neighbors // 2 + 1
                init_carry = (
                    start_idx,
                    jnp.array(0, dtype=int),
                    jnp.full((local_capacity,), -1, dtype=int),
                    jnp.array(False),
                )

                def cond_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> bool:
                    k, c, _, _ = val
                    in_cell = (k < n_b) * (p_cell_hash_b[k] == target_cell_hash)
                    has_space = c < local_capacity + 1
                    return cast(bool, in_cell * has_space)

                def body_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    k, c, nl, overflow = val
                    dr = system.domain.displacement(pos_ai, pos_b_sorted[k], system)
                    d_sq = norm2(dr)
                    valid = d_sq <= cutoff_sq
                    nl = jax.lax.cond(
                        valid,
                        lambda nl_: nl_.at[c].set(k, mode="drop"),
                        lambda nl_: nl_,
                        nl,
                    )
                    c_new = c + valid.astype(c.dtype)
                    return k + 1, c_new, nl, overflow + (c_new > local_capacity)

                _, local_c, local_nl, local_overflow = jax.lax.while_loop(
                    cond_fun, body_fun, init_carry
                )
                return local_nl, local_c, local_overflow

            final_n_list, stencil_counts, stencil_overflows = jax.vmap(stencil_body)(
                stencil, cell_starts
            )
            row_offsets = jnp.cumsum(stencil_counts) - stencil_counts
            local_iota = jnp.arange(final_n_list.shape[1])
            target_indices = row_offsets[:, None] + local_iota[None, :]
            valid_mask = local_iota[None, :] < stencil_counts[:, None]
            safe_indices = jnp.where(
                valid_mask.flatten(), target_indices.flatten(), max_neighbors
            )
            result = jnp.full((max_neighbors,), -1, dtype=final_n_list.dtype)
            final_n_list = result.at[safe_indices].set(
                final_n_list.flatten(), mode="drop"
            )
            overflow_flag = (
                jnp.sum(stencil_overflows)
                > 0 + (jnp.sum(stencil_counts) > max_neighbors)
            ).astype(bool)
            return final_n_list, overflow_flag

        return _compute_cross_neighbor_list_common(
            pos_a, pos_b, system, traverse, max_neighbors
        )


__all__ = ["DynamicCellList", "StaticCellList"]
