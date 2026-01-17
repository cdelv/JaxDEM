# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from typing import Tuple, Optional, TYPE_CHECKING, cast
from functools import partial

try:  # Python 3.11+
    from typing import Self  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from . import Collider
from ..utils.linalg import cross

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
@partial(jax.named_call, name="cell_list._get_spatial_partition")
def _get_spatial_partition(
    pos: jax.Array,
    system: "System",
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    iota: jax.Array,
) -> Tuple[jax.Array, ...]:
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
    p_cell_coords = jnp.floor((pos - system.domain.anchor) / cell_size).astype(int)

    # Wrap indices for hashing purposes if periodic
    # system.domain.periodic is a static variable. This is a compile time if
    if system.domain.periodic:
        p_cell_coords -= grid_dims * jnp.floor(p_cell_coords / grid_dims).astype(int)

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
      might be missed (you should choose ``cell_size`` and ``max_occupancy`` so this
      does not happen).
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
        state: "State",
        cell_size: Optional[ArrayLike] = None,
        search_range: Optional[ArrayLike] = None,
        max_occupancy: Optional[ArrayLike] = None,
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

        By default, we choose the options that yield the lowest computational cost: :math:`L = 2 \cdot r_{max}` if :math:`\alpha < 2.5`, else :math:`L = r_{max}/2`.

        The complexity of searching neighbors is :math:`O(N)`, where the choice
        of cell size and :math:`R` attempts to minimize the constant factor. The constant factor
        grows with polydispersity (:math:`\alpha`) as :math:`O(\alpha^{dim})` with :math:`\alpha = r_{max}/r_{min}`. The cost for sorting and binary search remains :math:`O(N \log N)`.

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

            max_occupancy = jnp.ceil(box_vol / smallest_sphere_vol) + 2

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
            max_occupancy=int(max_occupancy),  # type: ignore[arg-type]
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="StaticCellList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
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
        """
        collider = cast(StaticCellList, system.collider)
        iota = jax.lax.iota(dtype=int, size=state.N)
        MAX_OCCUPANCY = collider.max_occupancy

        # 1. Get spatial partitioning
        (
            perm,
            _,  # p_cell_coords
            p_cell_hash,
            _,  # neighbor_cell_coords
            p_neighbor_cell_hashes,
        ) = _get_spatial_partition(
            state.pos, system, collider.cell_size, collider.neighbor_mask, iota
        )
        state = jax.tree.map(lambda x: x[perm], state)
        pos_p = state.q.rotate(state.q, state.pos_p)  # to lab

        def per_particle(
            idx: jax.Array, pos_pi: jax.Array, neighbor_cell_hashes: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            def per_neighbor_cell(
                target_cell_hash: jax.Array,
            ) -> Tuple[jax.Array, jax.Array]:
                # Find Start Indices
                # Find where each neighbor hash starts in the sorted particle list.
                # 'searchsorted' returns the insertion index.
                # We do this inside the vmap to save memory (N, M) -> (N,)
                start_idx = jnp.searchsorted(
                    p_cell_hash,
                    target_cell_hash,
                    side="left",
                    method="scan_unrolled",
                )

                k_indices = start_idx + jax.lax.iota(dtype=int, size=MAX_OCCUPANCY)
                safe_k = jnp.minimum(k_indices, state.N - 1)

                # Validity mask: index bounds, correct cell hash, and not self-interaction
                valid = (
                    (k_indices < state.N)
                    * (p_cell_hash[safe_k] == target_cell_hash)
                    * (state.ID[safe_k] != state.ID[idx])
                )

                res_f, res_t = system.force_model.force(idx, safe_k, state, system)
                sum_f = jnp.sum(res_f * valid[:, None], axis=0)
                sum_t = jnp.sum(res_t * valid[:, None], axis=0)
                sum_t += cross(pos_pi, sum_f)
                return sum_f, sum_t

            # VMAP over all over the stencil of neighbor cells
            result = jax.vmap(per_neighbor_cell)(neighbor_cell_hashes)
            return jax.tree.map(lambda x: x.sum(axis=0), result)

        # 2. Compute forces for all particles
        total_force, total_torque = jax.vmap(per_particle)(
            iota, pos_p, p_neighbor_cell_hashes
        )

        # 3. Aggregate back to original particle IDs
        total_torque = jax.ops.segment_sum(total_torque, state.ID, num_segments=state.N)
        total_force = jax.ops.segment_sum(total_force, state.ID, num_segments=state.N)
        state.force += total_force[state.ID]
        state.torque += total_torque[state.ID]

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="StaticCellList.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
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
        iota = jax.lax.iota(dtype=int, size=state.N)
        MAX_OCCUPANCY = collider.max_occupancy

        # 1. Get spatial partitioning
        (
            perm,
            _,  # p_cell_coords
            p_cell_hash,
            _,  # neighbor_cell_coords
            p_neighbor_cell_hashes,
        ) = _get_spatial_partition(
            state.pos, system, collider.cell_size, collider.neighbor_mask, iota
        )
        state = jax.tree.map(lambda x: x[perm], state)

        def per_particle(idx: jax.Array, neighbor_cell_hashes: jax.Array) -> jax.Array:
            def per_neighbor_cell(target_cell_hash: jax.Array) -> jax.Array:
                # Find Start Indices
                # Find where each neighbor hash starts in the sorted particle list.
                # 'searchsorted' returns the insertion index.
                # We do this inside the vmap to save memory (N, M) -> (N,)
                start_idx = jnp.searchsorted(
                    p_cell_hash,
                    target_cell_hash,
                    side="left",
                    method="scan_unrolled",
                )

                k_indices = start_idx + jax.lax.iota(dtype=int, size=MAX_OCCUPANCY)
                safe_k = jnp.minimum(k_indices, state.N - 1)

                # Validity mask: index bounds, correct cell hash, and not self-interaction
                valid = (
                    (k_indices < state.N)
                    * (p_cell_hash[safe_k] == target_cell_hash)
                    * (state.ID[safe_k] != state.ID[idx])
                )

                e_ij = system.force_model.energy(idx, safe_k, state, system)
                return jnp.sum(e_ij * valid)

            # 2. VMAP over all over the stencil of neighbor cells
            return jax.vmap(per_neighbor_cell)(neighbor_cell_hashes).sum()

        # 3. VMAP over all particles
        return 0.5 * jax.vmap(per_particle)(iota, p_neighbor_cell_hashes)

    @staticmethod
    @partial(jax.jit, static_argnames=("max_neighbors"))
    @partial(jax.named_call, name="StaticCellList.create_neighbor_list")
    def create_neighbor_list(
        state: "State", system: "System", cutoff: float, max_neighbors: int
    ) -> tuple["State", "System", jax.Array, jax.Array]:
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
        tuple["State", "System", jax.Array, jax.Array]
            The sorted state, the system, the neighbor list, and a boolean flag for overflow.
        """
        collider = cast(StaticCellList, system.collider)
        iota = jax.lax.iota(int, state.N)
        MAX_OCCUPANCY = collider.max_occupancy
        cutoff_sq = cutoff**2

        (perm, _, p_cell_hash, _, p_neighbor_hashes) = _get_spatial_partition(
            state.pos, system, collider.cell_size, collider.neighbor_mask, iota
        )
        state = jax.tree.map(lambda x: x[perm], state)
        pos = state.pos

        def per_particle(
            idx: jax.Array, pos_i: jax.Array, stencil: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            # Find the starting memory position of each neighbor cell
            starts = jnp.searchsorted(
                p_cell_hash,
                stencil,
                side="left",
                method="scan_unrolled",
            )

            # Vectorized index generation (StencilSize, MAX_OCC)
            k_indices = (starts[:, None] + jax.lax.iota(int, MAX_OCCUPANCY)).reshape(-1)
            safe_k = jnp.minimum(k_indices, state.N - 1)

            # Compute distance
            dr = system.domain.displacement(pos_i, pos[safe_k], system)
            dist_sq = jnp.sum(dr**2, axis=-1)

            # Minimal Validity Mask
            # Instead of repeating hashes, we check index bounds and the actual hash at safe_k
            valid = (
                (k_indices < state.N)
                * (p_cell_hash[safe_k] == jnp.repeat(stencil, MAX_OCCUPANCY))
                * (state.ID[safe_k] != state.ID[idx])
                * (dist_sq <= cutoff_sq)
            )
            num_neighbors = jnp.sum(valid)
            overflow_flag = num_neighbors > max_neighbors

            candidates = jnp.where(valid, safe_k, -1)
            return jax.lax.top_k(candidates, max_neighbors)[0], overflow_flag

        neighbor_list, overflows = jax.vmap(per_particle)(iota, pos, p_neighbor_hashes)
        return state, system, neighbor_list, jnp.any(overflows)


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
        state: "State",
        cell_size: Optional[ArrayLike] = None,
        search_range: Optional[ArrayLike] = None,
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
        search_range : int, optional
            Neighbor range in cell units. If None, the smallest safe value is
            computed such that :math:`\text{search\_range} \cdot L \geq \text{cutoff}`.
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
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="DynamicCellList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
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
        """
        collider = cast(DynamicCellList, system.collider)
        iota = jax.lax.iota(dtype=int, size=state.N)

        # 1. Get spatial partitioning
        (
            perm,
            _,  # p_cell_coords
            p_cell_hash,
            _,  # neighbor_cell_coords
            p_neighbor_cell_hashes,
        ) = _get_spatial_partition(
            state.pos, system, collider.cell_size, collider.neighbor_mask, iota
        )
        state = jax.tree.map(lambda x: x[perm], state)
        pos_p = state.q.rotate(state.q, state.pos_p)  # to lab frame

        def per_particle(
            idx: jax.Array, pos_pi: jax.Array, stencil: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            def per_neighbor_cell(
                target_cell_hash: jax.Array,
            ) -> Tuple[jax.Array, jax.Array]:
                start_idx = jnp.searchsorted(
                    p_cell_hash, target_cell_hash, side="left", method="scan_unrolled"
                )

                def cond_fun(val: Tuple[jax.Array, jax.Array, jax.Array]) -> bool:
                    k, _, _ = val
                    # Continue if k is in bounds AND the particle at k is still in the target cell
                    return (k < state.N) * (p_cell_hash[k] == target_cell_hash)

                def body_fun(
                    val: Tuple[jax.Array, jax.Array, jax.Array],
                ) -> Tuple[jax.Array, jax.Array, jax.Array]:
                    k, acc_f, acc_t = val
                    valid = state.ID[k] != state.ID[idx]
                    f_kj, t_kj = system.force_model.force(idx, k, state, system)
                    f_kj *= valid
                    t_kj *= valid
                    return k + 1, acc_f + f_kj, acc_t + t_kj

                # Initial loop state
                init_val = (
                    start_idx,
                    jnp.zeros_like(state.force[idx]),
                    jnp.zeros_like(state.torque[idx]),
                )

                _, final_f, final_t = jax.lax.while_loop(cond_fun, body_fun, init_val)
                return final_f, final_t

            # Vmap over the cells in the stencil
            cell_forces, cell_torques = jax.vmap(per_neighbor_cell)(stencil)

            # Sum contributions from all neighbor cells
            sum_f = cell_forces.sum(axis=0)
            # Add the cross product for the particle's contact point once for the total force
            sum_t = cell_torques.sum(axis=0) + cross(pos_pi, sum_f)

            return sum_f, sum_t

        # 2. Compute forces for all particles in parallel
        total_force, total_torque = jax.vmap(per_particle)(
            iota, pos_p, p_neighbor_cell_hashes
        )

        # 3. Aggregate back to original particle slots/IDs
        state.force += total_force
        state.torque += total_torque
        state.torque = jax.ops.segment_sum(state.torque, state.ID, num_segments=state.N)
        state.force = jax.ops.segment_sum(state.force, state.ID, num_segments=state.N)
        state.force = state.force[state.ID]
        state.torque = state.torque[state.ID]

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicCellList.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
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
        iota = jax.lax.iota(dtype=int, size=state.N)

        # 1. Get spatial partitioning
        (perm, _, p_cell_hash, _, p_neighbor_cell_hashes) = _get_spatial_partition(
            state.pos, system, collider.cell_size, collider.neighbor_mask, iota
        )
        state = jax.tree.map(lambda x: x[perm], state)

        def per_particle(idx: jax.Array, stencil: jax.Array) -> jax.Array:
            def per_neighbor_cell(target_hash: jax.Array) -> jax.Array:
                start_idx = jnp.searchsorted(p_cell_hash, target_hash, side="left")

                def cond_fun(val: Tuple[jax.Array, jax.Array]) -> bool:
                    k, _ = val
                    return (k < state.N) * (p_cell_hash[k] == target_hash)

                def body_fun(
                    val: Tuple[jax.Array, jax.Array],
                ) -> Tuple[jax.Array, jax.Array]:
                    k, acc_e = val
                    valid = state.ID[k] != state.ID[idx]
                    e_ij = system.force_model.energy(idx, k, state, system)
                    return k + 1, acc_e + (0.5 * e_ij * valid)

                init_val = (start_idx, jnp.array(0.0, dtype=float))
                _, final_e = jax.lax.while_loop(cond_fun, body_fun, init_val)
                return final_e

            # 2. VMAP over the stencil
            cell_energies = jax.vmap(per_neighbor_cell)(stencil)
            return jnp.sum(cell_energies)

        # 3. VMAP over all particles
        # This returns an array of energies [N]
        return jax.vmap(per_particle)(iota, p_neighbor_cell_hashes)

    @staticmethod
    @partial(jax.jit, static_argnames=("max_neighbors"))
    def create_neighbor_list(
        state: "State", system: "System", cutoff: float, max_neighbors: int
    ) -> Tuple["State", "System", jax.Array, jax.Array]:
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
        tuple["State", "System", jax.Array, jax.Array]
            The sorted state, the system, the neighbor list, and a boolean flag for overflow.
        """
        collider = cast(DynamicCellList, system.collider)
        iota = jax.lax.iota(int, state.N)
        cutoff_sq = cutoff**2

        # 1. Spatial Partitioning
        (perm, _, p_cell_hash, _, p_neighbor_hashes) = _get_spatial_partition(
            state.pos, system, collider.cell_size, collider.neighbor_mask, iota
        )
        state = jax.tree.map(lambda x: x[perm], state)
        pos = state.pos

        def per_particle(
            idx: jax.Array, pos_i: jax.Array, stencil: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            def stencil_body(
                i: int, carry: Tuple[jax.Array, jax.Array, jax.Array]
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
                global_c, n_list, overflow = carry
                target_cell_hash = stencil[i]
                start_idx = jnp.searchsorted(
                    p_cell_hash, target_cell_hash, side="left", method="scan_unrolled"
                )

                def cond_fun(
                    val: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> bool:
                    k, _, _, _ = val
                    return (k < state.N) * (p_cell_hash[k] == target_cell_hash)

                def body_fun(
                    val: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    k, c, nl, ovr = val
                    dr = system.domain.displacement(pos_i, pos[k], system)
                    d_sq = jnp.sum(dr**2, axis=-1)
                    valid = (state.ID[k] != state.ID[idx]) * (d_sq <= cutoff_sq)
                    nl = nl.at[c].set(k * valid + (valid - 1))
                    return k + 1, c + valid, nl, ovr + c > max_neighbors

                _, global_c, n_list, overflow = jax.lax.while_loop(
                    cond_fun, body_fun, (start_idx, global_c, n_list, overflow)
                )
                return global_c, n_list, overflow > 0

            init_carry = (0, jnp.full((max_neighbors,), -1, dtype=int), False)
            final_c, final_n_list, final_ovr = jax.lax.fori_loop(
                0, stencil.shape[0], stencil_body, init_carry
            )
            return final_n_list, final_ovr

        neighbor_list, overflows = jax.vmap(per_particle)(iota, pos, p_neighbor_hashes)
        return state, system, neighbor_list, jnp.any(overflows)


__all__ = ["StaticCellList", "DynamicCellList"]
