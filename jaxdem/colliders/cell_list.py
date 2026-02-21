# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, cast, Union
from functools import partial

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from . import Collider, valid_interaction_mask
from ..utils.linalg import cross

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
) -> Tuple[jax.Array, ...]:
    """
    Computes spatial hashing and partitioning for the cell list.

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
    """
    Deduplicate one particle's stencil hashes, padding duplicates with -1.

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
    reduce_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[State, Any]:
    """
    Common logic for computing interactions (Force or Energy) using a Cell List.

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
) -> Tuple[State, System, jax.Array, jax.Array]:
    """
    Common logic for creating neighbor lists using a Cell List.

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
    ) -> Tuple[jax.Array, jax.Array]:
        if system.domain.periodic:
            stencil = _dedup_stencil_hashes(stencil)

        return traverse_fn(idx, pos_i, stencil, state, system, pos, p_cell_hash)

    neighbor_list, overflows = jax.vmap(per_particle)(iota, pos, p_neighbor_hashes)
    return state, system, neighbor_list, jnp.any(overflows)


def _force_kernel(
    idx: jax.Array,
    k: jax.Array,
    valid: jax.Array,
    pos: jax.Array,
    state: State,
    system: System,
) -> Tuple[jax.Array, jax.Array]:
    """
    Common interaction kernel for force computation.

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
    """
    Common interaction kernel for potential energy computation.

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
    res: Tuple[jax.Array, jax.Array], pos_pi: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Common reduction for force/torque accumulation.

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
    """
    Traversal strategy for StaticCellList (unrolled fixed-size loop).

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
    """
    Traversal strategy for DynamicCellList (while_loop).

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

    def cond_fun(val: Tuple[jax.Array, Any]) -> bool:
        k, _ = val
        return cast(bool, (k < state.N) * (p_cell_hash[k] == target_hash))

    def body_fun(val: Tuple[jax.Array, Any]) -> Tuple[jax.Array, Any]:
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
    r"""
    Implicit cell-list (spatial hashing) collider.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid of cubic/square cells of side length ``cell_size``.
    Each particle is assigned to a cell, particles are sorted by cell hash, and
    interactions are evaluated only against particles in the same or neighboring
    cells given by ``neighbor_mask``. The cell list is *implicit* because we never
    store per-cell particle lists explicitly; instead, we exploit the sorted hashes
    and fixed ``max_occupancy`` to probe neighbors in-place.

    This collider is ideal for systems of spheres with minimum polydispersity and no dramatic overlaps.
    In this case, it might be even faster than the default cell list. However, it's not recommended for systems
    with clumps, dramatic overlaps, as it might skip some contacts, or polydispersity, as it hinders the performance of this collider.

    Complexity
    ----------
    - Time: :math:`O(N)` - :math:`O(N \log N)` from sorting, plus :math:`O(N M K)` for neighbor
      probing (M = number of neighbor cells, K = ``max_occupancy``). The state is close to sorted every frame.
    - Memory: :math:`O(N)`.

    Notes
    -----
    - ``max_occupancy`` is an upper bound on particles per cell.
      If a cell contains more than this many particles, some interactions
      might be missed (you should choose ``cell_size`` and ``max_occupancy`` so this does not happen).
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

    max_occupancy: int = field(metadata={"static": True})
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
        cell_size: Optional[ArrayLike] = None,
        search_range: Optional[ArrayLike] = None,
        box_size: Optional[ArrayLike] = None,
        max_occupancy: Optional[int] = None,
    ) -> Self:
        r"""
        Creates a StaticCellList collider with robust defaults.

        Defaults are chosen to avoid missing any contacts while keeping the
        neighbor stencil and assumed cell occupancy as small as possible given
        available information from ``state``. For this we assume no overlap between spheres.

        The cost of computing forces for one particle is determined by the number
        of neighboring cells to check and the occupancy of each cell. This cost
        can be estimated as:

        .. math::
            \text{cost} = (2R + 1)^{dim} \cdot \text{max_occupancy} \\
            \text{cost} = (2R + 1)^{dim} \cdot \left(\left\lceil \frac{L^{dim}}{V_{min}} \right\rceil +2 \right)

        where :math:`R` is the search radius, :math:`L` is the cell size, and
        :math:`V_{min}` is the volume of the smallest element. We assume
        :math:`V_{min}` to be the volume of the smallest sphere, without
        accounting for the packing fraction, to provide a conservative upper bound.
        The search radius :math:`R` is computed as:

        .. math::
            R = \left\lceil \frac{2 r_{max}}{L} \right\rceil

        By default, we choose the options that yield the lowest computational
        cost: :math:`L = 2 \cdot r_{max}` if :math:`\alpha < 2.5`, else :math:`L = r_{max}/2`.

        The complexity of searching neighbors is :math:`O(N)`, where the choice
        of cell size and :math:`R` attempts to minimize the constant factor. The constant factor
        grows with polydispersity (:math:`\alpha`) as :math:`O(\alpha^{dim})` with :math:`\alpha = r_{max}/r_{min}`.
        The cost for sorting and binary search remains :math:`O(N \log N)`.

        Parameters
        ----------
        state : State
            Reference state used to determine spatial dimension and default parameters.
        cell_size : float, optional
            Cell edge length. If None, defaults to a value optimized for the
            radius distribution.
        search_range : int, optional
            Neighbor range in cell units. If None, the smallest safe value is
            computed such that :math:`\text{search\_range} \cdot L \geq \text{cutoff}`.
        box_size : jax.Array, optional
            Size of the periodic box used to ensure there are at least
            ``2 * search_range + 1`` cells per axis. If None, these checks are
            skipped.
        max_occupancy : int, optional
            Assumed maximum particles per cell. If None, estimated from a
            conservative packing upper bound using the smallest radius.

        Returns
        -------
        CellList
            Configured collider instance.
        """
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
        alpha = max_rad / min_rad

        if cell_size is None:
            cell_size = 2.0 * max_rad
            if alpha < 2.5:
                cell_size = 2 * max_rad
            else:
                cell_size = max_rad / 2

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
            box_vol = cell_size**state.dim
            smallest_sphere_vol = jnp.array(0.0, dtype=float)
            if state.dim == 3:
                smallest_sphere_vol = (4.0 / 3.0) * jnp.pi * min_rad**3 / 0.9
            elif state.dim == 2:
                smallest_sphere_vol = jnp.pi * min_rad**2

            max_occupancy = int(jnp.ceil(box_vol / smallest_sphere_vol) + 2)

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
            max_occupancy=int(max_occupancy),
        )

    @staticmethod
    @jax.jit(donate_argnames=("state", "system"))
    @partial(jax.named_call, name="StaticCellList.compute_force")
    def compute_force(state: State, system: System) -> Tuple[State, System]:
        r"""
        Computes the total force acting on each particle using an implicit cell list :math:`O(N log N)`.
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

        Note
        ----
        - This method donates ``state`` and ``system``.
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
        r"""
        Computes the potential energy acting on each particle using an implicit cell list :math:`O(N log N)`.
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
            An array containing the potential energy for each particle.
        """
        collider = cast(StaticCellList, system.collider)

        traverse = partial(_static_traverse_cell, max_occupancy=collider.max_occupancy)

        _, energy = _compute_interaction(state, system, traverse, _energy_kernel)
        return energy

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="StaticCellList.create_neighbor_list")
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        r"""
        Computes the list of neighbors for each particle. The shape of the list is (N, max_neighbors).
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
        ) -> Tuple[jax.Array, jax.Array]:
            starts = jnp.searchsorted(
                p_cell_hash,
                stencil,
                side="left",
                method="scan_unrolled",
            )

            k_indices = (starts[:, None] + jax.lax.iota(int, MAX_OCCUPANCY)).reshape(-1)
            safe_k = jnp.minimum(k_indices, state.N - 1)

            dr = system.domain.displacement(pos_i, pos[safe_k], system)
            dist_sq = jnp.sum(dr**2, axis=-1)

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


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicCellList(Collider):
    r"""
    Implicit cell-list (spatial hashing) collider using dynamic while-loops.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid. Unlike the static cell list, this implementation
    uses a dynamic ``jax.lax.while_loop`` to probe neighbor cells, which can be
    more efficient with polydisperse systems or low packing fractions. It's also useful for systems that
    have a high occupancy per cell, for example, systems with clumps.

    Complexity
    ----------
    - Time: :math:`O(N)` - :math:`O(N \log N)` from sorting, plus :math:`O(N M \langle K \rangle)`
      for neighbor probing, where :math:`\langle K \rangle` is the average cell occupancy. The state is close to sorted every frame.
    - Memory: :math:`O(N)`.
    """

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a grid cell (scalar)."""

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: Optional[ArrayLike] = None,
        search_range: Optional[ArrayLike] = None,
        box_size: Optional[ArrayLike] = None,
    ) -> Self:
        r"""
        Creates a CellList collider with robust defaults.

        Defaults are chosen to avoid missing any contacts while keeping the
        neighbor stencil and assumed cell occupancy as small as possible given
        available information from ``state``.

        The cost of computing forces for one particle is determined by the number
        of neighboring cells to check and the occupancy of each cell. This cost
        can be estimated as:

        .. math::
            \text{cost} = (2R + 1)^{dim} \cdot \text{max_occupancy} \\
            \text{cost} = (2R + 1)^{dim} \cdot \left(\left\lceil \frac{L^{dim}}{V_{min}} \right\rceil +2 \right)

        where :math:`R` is the search radius, :math:`L` is the cell size, and
        :math:`V_{min}` is the volume of the smallest element. We assume
        :math:`V_{min}` to be the volume of the smallest sphere, without
        accounting for the packing fraction, to provide a conservative upper bound.
        The search radius :math:`R` is computed as:

        .. math::
            R = \left\lceil \frac{2 r_{max}}{L} \right\rceil

        By default, we choose the options that yield the lowest computational cost: :math:`L = 2 \cdot r_{max}` if :math:`\alpha < 2.5`, else :math:`L = r_{max}/2`.

        The complexity of searching neighbors is :math:`O(N)`, where the choice
        of cell size and :math:`R` attempts to minimize the constant factor. The constant factor
        grows with polydispersity; however, the dynamic nature of the collider greatly minimizes polydispersity's impact.

        Parameters
        ----------
        state : State
            Reference state used to determine spatial dimension and default parameters.
        cell_size : float, optional
            Cell edge length. If None, defaults to a value optimized for the
            radius distribution.
        box_size : jax.Array, optional
            Size of the periodic box used to ensure there are at least 3 cells per axis.
            If None, these checks are ignored and will lead to errors if violated.
        search_range : int, optional
            Neighbor range in cell units. If None, the smallest safe value is
            computed such that :math:`\text{search\_range} \cdot L \geq \text{cutoff}`.

        Returns
        -------
        CellList
            Configured collider instance.
        """
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
        alpha = max_rad / min_rad

        if cell_size is None:
            cell_size = 2.0 * max_rad
            if alpha < 2.5:
                cell_size = 2 * max_rad
            else:
                cell_size = max_rad / 2

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
    @jax.jit(donate_argnames=("state", "system"))
    @partial(jax.named_call, name="DynamicCellList.compute_force")
    def compute_force(state: State, system: System) -> Tuple[State, System]:
        r"""
        Computes the total force acting on each particle using an implicit cell list :math:`O(N log N)`.
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

        Note
        ----
        - This method donates ``state`` and ``system``.
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
        r"""
        Computes the potential energy acting on each particle using an implicit cell list :math:`O(N log N)`.
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
            An array containing the potential energy for each particle.
        """
        collider = cast(DynamicCellList, system.collider)

        traverse = partial(_dynamic_traverse_cell, init_val=jnp.array(0.0, dtype=float))

        _, energy = _compute_interaction(state, system, traverse, _energy_kernel)
        return energy

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> Tuple[State, System, jax.Array, jax.Array]:
        r"""
        Computes the list of neighbors for each particle. The shape of the list is (N, max_neighbors).
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
        ) -> Tuple[jax.Array, jax.Array]:
            cell_starts = jnp.searchsorted(
                p_cell_hash, stencil, side="left", method="scan_unrolled"
            )

            def stencil_body(
                target_cell_hash: jax.Array, start_idx: jax.Array
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
                local_capacity = max_neighbors // 2 + 1
                init_carry = (
                    start_idx,
                    jnp.array(0, dtype=int),
                    jnp.full((local_capacity,), -1, dtype=int),
                    jnp.array(False),  # overflow flag
                )

                def cond_fun(
                    val: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> bool:
                    k, c, _, _ = val
                    in_cell = (k < state.N) * (p_cell_hash[k] == target_cell_hash)
                    has_space = c < local_capacity + 1  # so we can catch the overflow
                    return cast(bool, in_cell * has_space)

                def body_fun(
                    val: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    k, c, nl, overflow = val
                    dr = system.domain.displacement(pos_i, pos[k], system)
                    d_sq = jnp.sum(dr**2, axis=-1)
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


__all__ = ["StaticCellList", "DynamicCellList"]
