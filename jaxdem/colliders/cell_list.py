# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field, replace
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


# occupancy = ops.segment_sum(jnp.ones_like(hashes), hashes, cell_count)
# max_occupancy = jnp.max(occupancy)
# overflow = overflow | (max_occupancy > cell_capacity)


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CellList(Collider):
    r"""
    Implicit cell-list (spatial hashing) collider.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid of cubic/square cells of side length ``cell_size``.
    Each particle is assigned to a cell, particles are sorted by cell hash, and
    interactions are evaluated only against particles in the same or neighboring
    cells given by ``neighbor_mask``. The cell list is *implicit* because we never
    store per-cell particle lists explicitly; instead, we exploit the sorted hashes
    and fixed ``max_occupancy`` to probe neighbors in-place.

    Complexity
    ----------
    - Time: :math:`O(N \log N)` from sorting, plus :math:`O(N M K)` for neighbor
      probing (M = number of neighbor cells, K = ``max_occupancy``).
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
        Creates a CellList collider with robust defaults.

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
            cell_size=1.02 * jnp.asarray(cell_size, dtype=float),
            max_occupancy=int(max_occupancy),  # type: ignore[arg-type]
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="CellList.compute_force")
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
        collider = cast(CellList, system.collider)
        iota = jax.lax.iota(dtype=int, size=state.N)
        MAX_OCCUPANCY = collider.max_occupancy
        pos = state.pos
        pos_p = state.q.rotate(state.q, state.pos_p)  # to lab

        # 1. Determine Grid Dimensions
        # shape: (dim,)
        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
        else:
            grid_dims = jnp.ceil(system.domain.box_size / collider.cell_size).astype(
                int
            )

        # Compute strides (weights) for flattening 2D/3D indices to 1D hash
        # [1, nx, nx*ny, ...]
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        # 2. Calculate Particle Cell Indices
        cell_ids = jnp.floor((pos - system.domain.anchor) / collider.cell_size).astype(
            int
        )

        # Wrap indices for hashing purposes if periodic
        # system.domain.periodic is a static variable. This is a compile time if
        if system.domain.periodic:
            cell_ids -= grid_dims * jnp.floor(cell_ids / grid_dims).astype(int)

        # 3. Spatial Hashing
        # shape (N,)
        particle_hash = jnp.dot(cell_ids, strides)

        # 4. Sort hashes and state
        particle_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)
        state = jax.tree.map(lambda x: x[perm], state)
        cell_ids = cell_ids[perm]

        # 5. Precompute Neighbor Cell Hashes for every particle
        # (N, M, dim) = (N, 1, dim) + (1, M, dim)
        # M is number of neighbor cells (e.g., 27)
        current_cell = cell_ids[:, None, :] + collider.neighbor_mask

        if system.domain.periodic:
            current_cell -= grid_dims * jnp.floor(current_cell / grid_dims).astype(int)

        # shape (N,M)
        cell_hashes = jnp.dot(current_cell, strides)

        def per_particle(
            i: jax.Array, pos_pi: jax.Array, my_cell_id: jax.Array, cell_hash: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            def per_neighbor_cell(
                current_cell_hash: jax.Array,
            ) -> Tuple[jax.Array, jax.Array]:
                # 1. Find Start Indices
                # Find where each neighbor hash starts in the sorted particle list.
                # 'searchsorted' returns the insertion index.
                # We do this inside the vmap to save memory (N, M) -> (N,)
                start_idx = jnp.searchsorted(
                    particle_hash,
                    current_cell_hash,
                    side="left",
                    method="scan_unrolled",
                )

                def body_fun(offset: jax.Array) -> Tuple[jax.Array, jax.Array]:
                    k = start_idx + offset
                    safe_k = jnp.minimum(k, state.N - 1)
                    valid = (
                        (k < state.N)
                        * (particle_hash[safe_k] == current_cell_hash)
                        * (state.ID[safe_k] != state.ID[i])
                    )
                    result = system.force_model.force(i, safe_k, state, system)
                    forces, torques = jax.tree.map(lambda x: valid * x, result)
                    torques += cross(pos_pi, forces)

                    return forces, torques

                # VMAP over the fixed number of contacts
                result = jax.vmap(body_fun)(jax.lax.iota(size=MAX_OCCUPANCY, dtype=int))
                return jax.tree.map(lambda x: x.sum(axis=0), result)

            # VMAP over neighbor cells
            result = jax.vmap(per_neighbor_cell)(cell_hash)
            return jax.tree.map(lambda x: x.sum(axis=0), result)

        # VMAP over all particles
        total_force, total_torque = jax.vmap(per_particle)(
            iota, pos_p, cell_ids, cell_hashes
        )

        total_torque = jax.ops.segment_sum(total_torque, state.ID, num_segments=state.N)
        total_force = jax.ops.segment_sum(total_force, state.ID, num_segments=state.N)

        state.force += total_force[state.ID]
        state.torque += total_torque[state.ID]

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="CellList.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
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
        collider = cast(CellList, system.collider)
        iota = jax.lax.iota(dtype=int, size=state.N)
        MAX_OCCUPANCY = collider.max_occupancy
        pos = state.pos

        # 1. Determine Grid Dimensions
        # shape: (dim,)
        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
        else:
            grid_dims = jnp.ceil(system.domain.box_size / collider.cell_size).astype(
                int
            )

        # Compute strides (weights) for flattening 2D/3D indices to 1D hash
        # [1, nx, nx*ny, ...]
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        # 2. Calculate Particle Cell Indices
        cell_ids = jnp.floor((pos - system.domain.anchor) / collider.cell_size).astype(
            int
        )

        # Wrap indices for hashing purposes if periodic
        # system.domain.periodic is a static variable. This is a compile time if
        if system.domain.periodic:
            cell_ids -= grid_dims * jnp.floor(cell_ids / grid_dims).astype(int)

        # 3. Spatial Hashing
        # shape (N,)
        particle_hash = jnp.dot(cell_ids, strides)

        # 4. Sort hashes and state
        particle_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)
        state = jax.tree.map(lambda x: x[perm], state)
        cell_ids = cell_ids[perm]

        # 5. Precompute Neighbor Cell Hashes for every particle
        # (N, M, dim) = (N, 1, dim) + (1, M, dim)
        # M is number of neighbor cells (e.g., 27)
        current_cell = cell_ids[:, None, :] + collider.neighbor_mask

        if system.domain.periodic:
            current_cell -= grid_dims * jnp.floor(current_cell / grid_dims).astype(int)

        # shape (N,M)
        cell_hashes = jnp.dot(current_cell, strides)

        def per_particle(
            i: jax.Array, my_cell_id: jax.Array, cell_hash: jax.Array
        ) -> jax.Array:
            def per_neighbor_cell(
                current_cell_hash: jax.Array,
            ) -> jax.Array:
                # 1. Find Start Indices
                # Find where each neighbor hash starts in the sorted particle list.
                # 'searchsorted' returns the insertion index.
                # We do this inside the vmap to save memory (N, M) -> (N,)
                start_idx = jnp.searchsorted(
                    particle_hash,
                    current_cell_hash,
                    side="left",
                    method="scan_unrolled",
                )

                def body_fun(offset: jax.Array) -> jax.Array:
                    k = start_idx + offset
                    safe_k = jnp.minimum(k, state.N - 1)
                    e_ij = system.force_model.energy(i, safe_k, state, system)
                    valid = (
                        (k < state.N)
                        * (particle_hash[safe_k] == current_cell_hash)
                        * (state.ID[safe_k] != state.ID[i])
                    )
                    e_ij *= valid
                    return 0.5 * e_ij

                # VMAP over the fixed number of contacts
                return jax.vmap(body_fun)(
                    jax.lax.iota(size=MAX_OCCUPANCY, dtype=int)
                ).sum()

            # VMAP over neighbor cells
            return jax.vmap(per_neighbor_cell)(cell_hash).sum()

        # VMAP over all particles
        return jax.vmap(per_particle)(iota, cell_ids, cell_hashes)

    # @staticmethod
    # @partial(jax.jit, inline=True)
    # def find_neighbors(state: "State", system: "System") -> jax.Array:
    #     """
    #     Finds neighbors for ALL particles using a single global vectorized search.

    #     Optimized Strategy:
    #     -------------------
    #     1. Calculate all (N, M) neighbor cell hashes at once.
    #     2. Perform ONE global `searchsorted` on the (N*M) query points.
    #     3. Broadcast the results to generate the (N, M*K) neighbor matrix.

    #     Returns
    #     -------
    #     jnp.ndarray
    #         Matrix of neighbor indices with shape ``(N, M * MAX_OCCUPANCY)``.
    #     """
    #     collider = cast(CellList, system.collider)
    #     N = state.N
    #     MAX_OCCUPANCY = collider.max_occupancy

    #     # --- 1. Grid & Hash Setup ---
    #     # Shape: (dim,)
    #     grid_dims = jnp.ceil(system.domain.box_size / collider.cell_size).astype(int)
    #     strides = jnp.concatenate(
    #         [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
    #     )

    #     # Calculate cell IDs for all particles: (N, dim)
    #     cell_ids = jnp.floor(
    #         (state.pos - system.domain.anchor) / collider.cell_size
    #     ).astype(int)

    #     if system.domain.periodic:
    #         cell_ids -= grid_dims * jnp.floor(cell_ids / grid_dims).astype(int)

    #     # Particle hashes (sorted by caller assumption): (N,)
    #     particle_hash = jnp.dot(cell_ids, strides)

    #     # --- 2. Generate All Neighbor Queries Globally ---

    #     # Broadcast add to get all neighbor cell coords for all particles
    #     # (N, 1, dim) + (M, dim) -> (N, M, dim)
    #     neighbor_cells = cell_ids[:, None, :] + collider.neighbor_mask

    #     if system.domain.periodic:
    #         neighbor_cells -= grid_dims * jnp.floor(neighbor_cells / grid_dims).astype(
    #             int
    #         )

    #     # Compute hashes for all N*M neighbor cells
    #     # (N, M, dim) dot (dim,) -> (N, M)
    #     neighbor_hashes = jnp.dot(neighbor_cells, strides)

    #     # --- 3. Global Binary Search ("Preallocation" step) ---

    #     # Instead of vmapping searchsorted, we pass the entire (N, M) array.
    #     # JAX will execute this as a single kernel.
    #     # Result `start_indices` has shape (N, M)
    #     start_indices = jnp.searchsorted(
    #         particle_hash, neighbor_hashes, side="left", method="scan_unrolled"
    #     )

    #     # --- 4. Expand to Max Occupancy ---

    #     # We need to generate indices [start, start+1, ..., start+K]
    #     # Reshape start_indices for broadcasting: (N, M, 1)
    #     # Add offset (K,): Result shape (N, M, K)
    #     offset = jax.lax.iota(int, MAX_OCCUPANCY)
    #     candidate_indices = start_indices[:, :, None] + offset

    #     # --- 5. Validate & Mask (The Matrix Construction) ---

    #     # A. Clip indices to avoid OOB reads during validation
    #     safe_indices = jnp.minimum(candidate_indices, N - 1)

    #     # B. Check 1: Is the candidate index within the particle list size?
    #     # C. Check 2: Does the candidate actually belong to the target neighbor cell?
    #     #    (This handles cells that are not full)
    #     # D. Check 3: Is the candidate NOT the particle itself?

    #     # Expand particle_hash to map against candidates: (N, M, K) matches (N, M) queries
    #     # Expand iota for self-check: (N, 1, 1)
    #     iota_N = jax.lax.iota(int, N)[:, None, None]

    #     is_valid = (
    #         (candidate_indices < N)
    #         * (particle_hash[safe_indices] == neighbor_hashes[:, :, None])
    #         * (candidate_indices != iota_N)
    #     )

    #     # Apply mask: Set invalid to -1
    #     # Shape: (N, M, K)
    #     neighbors = jnp.where(is_valid, candidate_indices, -1)

    #     # --- 6. Flatten ---
    #     # Combine M neighbors cells * K occupancy -> (N, M*K)
    #     return neighbors.reshape(N, -1)


@Collider.register("DynamicCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicCellList(Collider):
    r"""
    Implicit cell-list (spatial hashing) collider using dynamic while-loops.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid. Unlike the standard CellList, this implementation
    uses a dynamic ``jax.lax.while_loop`` to probe neighbor cells, which can be
    more efficient in systems with highly non-uniform particle distributions.

    Complexity
    ----------
    - Time: :math:`O(N \log N)` from sorting, plus :math:`O(N M \langle K \rangle)`
      for neighbor probing, where :math:`\langle K \rangle` is the average cell occupancy.
    - Memory: :math:`O(N)`.
    """

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a grid cell (scalar)."""

    max_occupancy: int = field(metadata={"static": True})
    """Maximum number of particles assumed to occupy a single cell (loop safety limit)."""

    @classmethod
    def Create(
        cls,
        state: "State",
        cell_size: Optional[ArrayLike] = None,
        search_range: Optional[ArrayLike] = None,
        max_occupancy: Optional[ArrayLike] = None,
    ) -> Self:
        r"""
        Creates a DynamicCellList collider with robust defaults.
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
            cell_size=1.02 * jnp.asarray(cell_size, dtype=float),
            max_occupancy=int(max_occupancy),  # type: ignore[arg-type]
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="DynamicCellList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        collider = cast(DynamicCellList, system.collider)
        iota = jax.lax.iota(dtype=int, size=state.N)
        MAX_OCCUPANCY = collider.max_occupancy
        pos = state.pos
        pos_p = state.q.rotate(state.q, state.pos_p)  # to lab

        # 1. Determine Grid Dimensions
        # shape: (dim,)
        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
        else:
            grid_dims = jnp.ceil(system.domain.box_size / collider.cell_size).astype(
                int
            )

        # Compute strides (weights) for flattening 2D/3D indices to 1D hash
        # [1, nx, nx*ny, ...]
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        # 2. Calculate Particle Cell Indices
        cell_ids = jnp.floor((pos - system.domain.anchor) / collider.cell_size).astype(
            int
        )

        # Wrap indices for hashing purposes if periodic
        # system.domain.periodic is a static variable. This is a compile time if
        if system.domain.periodic:
            cell_ids -= grid_dims * jnp.floor(cell_ids / grid_dims).astype(int)

        # 3. Spatial Hashing
        # shape (N,)
        particle_hash = jnp.dot(cell_ids, strides)

        # 4. Sort hashes and state
        particle_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)
        state = jax.tree.map(lambda x: x[perm], state)
        cell_ids = cell_ids[perm]

        # 5. Precompute Neighbor Cell Hashes for every particle
        # (N, M, dim) = (N, 1, dim) + (1, M, dim)
        # M is number of neighbor cells (e.g., 27)
        current_cell = cell_ids[:, None, :] + collider.neighbor_mask

        if system.domain.periodic:
            current_cell -= grid_dims * jnp.floor(current_cell / grid_dims).astype(int)

        # shape (N,M)
        cell_hashes = jnp.dot(current_cell, strides)

        def per_particle(
            i: jax.Array, pos_pi: jax.Array, my_cell_id: jax.Array, cell_hash: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            def per_neighbor_cell(
                current_cell_hash: jax.Array,
            ) -> Tuple[jax.Array, jax.Array]:
                start_idx = jnp.searchsorted(
                    particle_hash,
                    current_cell_hash,
                    side="left",
                    method="scan_unrolled",
                )

                def cond_fun(val: Tuple[int, jax.Array, jax.Array]) -> jax.Array:
                    k, _, _ = val
                    return (k < state.N) * (current_cell_hash == particle_hash[k])

                def body_fun(
                    val: Tuple[int, jax.Array, jax.Array]
                ) -> Tuple[int, jax.Array, jax.Array]:
                    k, acc_f, acc_t = val
                    valid = state.ID[k] != state.ID[i]
                    result = system.force_model.force(i, k, state, system)
                    forces, torques = jax.tree.map(lambda x: valid * x, result)
                    torques += cross(pos_pi, forces)
                    return k + 1, acc_f + forces, acc_t + torques

                init_val = (
                    cast(int, start_idx),
                    jnp.zeros_like(state.force[i]),
                    jnp.zeros_like(state.torque[i]),
                )
                _, final_f, final_t = jax.lax.while_loop(cond_fun, body_fun, init_val)
                return final_f, final_t

            # VMAP over neighbor cells
            result = jax.vmap(per_neighbor_cell)(cell_hash)
            return jax.tree.map(lambda x: x.sum(axis=0), result)

        # VMAP over all particles
        total_force, total_torque = jax.vmap(per_particle)(
            iota, pos_p, cell_ids, cell_hashes
        )

        total_torque = jax.ops.segment_sum(total_torque, state.ID, num_segments=state.N)
        total_force = jax.ops.segment_sum(total_force, state.ID, num_segments=state.N)

        state.force += total_force[state.ID]
        state.torque += total_torque[state.ID]

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicCellList.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        collider = cast(DynamicCellList, system.collider)
        iota = jax.lax.iota(dtype=int, size=state.N)
        MAX_OCCUPANCY = collider.max_occupancy
        pos = state.pos

        # 1. Determine Grid Dimensions
        # shape: (dim,)
        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
        else:
            grid_dims = jnp.ceil(system.domain.box_size / collider.cell_size).astype(
                int
            )

        # Compute strides (weights) for flattening 2D/3D indices to 1D hash
        # [1, nx, nx*ny, ...]
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        # 2. Calculate Particle Cell Indices
        cell_ids = jnp.floor((pos - system.domain.anchor) / collider.cell_size).astype(
            int
        )

        # Wrap indices for hashing purposes if periodic
        # system.domain.periodic is a static variable. This is a compile time if
        if system.domain.periodic:
            cell_ids -= grid_dims * jnp.floor(cell_ids / grid_dims).astype(int)

        # 3. Spatial Hashing
        # shape (N,)
        particle_hash = jnp.dot(cell_ids, strides)

        # 4. Sort hashes and state
        particle_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)
        state = jax.tree.map(lambda x: x[perm], state)
        cell_ids = cell_ids[perm]

        # 5. Precompute Neighbor Cell Hashes for every particle
        # (N, M, dim) = (N, 1, dim) + (1, M, dim)
        # M is number of neighbor cells (e.g., 27)
        current_cell = cell_ids[:, None, :] + collider.neighbor_mask

        if system.domain.periodic:
            current_cell -= grid_dims * jnp.floor(current_cell / grid_dims).astype(int)

        # shape (N,M)
        cell_hashes = jnp.dot(current_cell, strides)

        def per_particle(
            i: jax.Array, my_cell_id: jax.Array, cell_hash: jax.Array
        ) -> jax.Array:
            def per_neighbor_cell(
                current_cell_hash: jax.Array,
            ) -> jax.Array:
                # 1. Find Start Indices
                # Find where each neighbor hash starts in the sorted particle list.
                # 'searchsorted' returns the insertion index.
                # We do this inside the vmap to save memory (N, M) -> (N,)
                start_idx = jnp.searchsorted(
                    particle_hash,
                    current_cell_hash,
                    side="left",
                    method="scan_unrolled",
                )

                def cond_fun(val: Tuple[int, jax.Array]) -> jax.Array:
                    k, _ = val
                    return (k < state.N) * (current_cell_hash == particle_hash[k])

                def body_fun(val: Tuple[int, jax.Array]) -> Tuple[int, jax.Array]:
                    k, acc_e = val
                    valid = state.ID[k] != state.ID[i]
                    e_ij = system.force_model.energy(i, k, state, system)
                    return k + 1, acc_e + (0.5 * e_ij * valid)

                _, final_e = jax.lax.while_loop(
                    cond_fun,
                    body_fun,
                    (cast(int, start_idx), jnp.array(0.0, dtype=float)),
                )
                return final_e

            # VMAP over neighbor cells
            return jax.vmap(per_neighbor_cell)(cell_hash).sum()

        # VMAP over all particles
        return jax.vmap(per_particle)(iota, cell_ids, cell_hashes)


@Collider.register("MaterializedCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MaterializedCellList(Collider):
    r"""
    Ultra-Optimized Explicitly materialized cell list collider.

    Uses "dummy particle" padding and Wide Vectorization to process entire neighbor
    blocks in single operations, maximizing GPU parallel throughput.
    """

    neighbor_mask: jax.Array
    cell_size: jax.Array
    max_occupancy: int = field(metadata={"static": True})
    grid_dims: Tuple[int, ...] = field(metadata={"static": True})
    strides: jax.Array = field(metadata={"static": True})
    num_cells: int = field(metadata={"static": True})

    @classmethod
    def Create(
        cls,
        state: "State",
        box_size: ArrayLike,
        cell_size: Optional[ArrayLike] = None,
        search_range: Optional[ArrayLike] = None,
        max_occupancy: Optional[ArrayLike] = None,
        periodic: bool = True,
    ) -> Self:
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
        alpha = max_rad / min_rad
        if cell_size is None:
            cell_size = 2.0 * max_rad if alpha < 2.5 else max_rad / 2.0
        if search_range is None:
            search_range = jnp.ceil(2 * max_rad / cell_size).astype(int)
            search_range = jnp.maximum(1, search_range)
        search_range = jnp.array(search_range, dtype=int)
        if max_occupancy is None:
            box_vol = cell_size**state.dim
            smallest_sphere_vol = (
                (4.0 / 3.0) * jnp.pi * min_rad**3 / 0.9
                if state.dim == 3
                else jnp.pi * min_rad**2
            )
            max_occupancy = jnp.ceil(box_vol / smallest_sphere_vol) + 2

        if periodic:
            grid_dims_val = jnp.floor(jnp.asarray(box_size) / cell_size).astype(int)
        else:
            grid_dims_val = jnp.ceil(jnp.asarray(box_size) / cell_size).astype(int)

        grid_dims = tuple(map(int, grid_dims_val))
        num_cells = 1
        for d in grid_dims:
            num_cells *= d
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(jnp.array(grid_dims[:-1]))]
        )

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=1.02 * jnp.asarray(cell_size, dtype=float),
            max_occupancy=int(max_occupancy),
            grid_dims=grid_dims,
            strides=strides,
            num_cells=int(num_cells),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="MaterializedCellList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        collider = cast(MaterializedCellList, system.collider)
        N, MAX_OCC = state.N, collider.max_occupancy
        grid_dims = jnp.array(collider.grid_dims)
        strides = collider.strides

        iota = jax.lax.iota(dtype=int, size=N)
        pos_p = state.q.rotate(state.q, state.pos_p)

        # 1. Spatial Hash
        cell_indices = jnp.floor(
            (state.pos - system.domain.anchor) / collider.cell_size
        ).astype(int)
        if system.domain.periodic:
            cell_indices %= grid_dims
        else:
            cell_indices = jnp.clip(cell_indices, 0, grid_dims - 1)
        particle_hash = jnp.dot(cell_indices, strides)

        # 2. Sort
        particle_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)

        # 3. Cumsum for slots
        is_first = jnp.concatenate(
            [jnp.array([True]), (particle_hash[1:] != particle_hash[:-1])]
        )
        slots = (
            jax.lax.associative_scan(
                lambda a, b: (jnp.where(b[1], b[0], a[0] + b[0]), a[1] | b[1]),
                (jnp.ones(N, int), is_first),
            )[0]
            - 1
        )

        # Build reordered state + Dummy Particle at index N
        def add_dummy(x):
            return jnp.concatenate([x[perm], jnp.expand_dims(x[perm[0]], 0)], axis=0)

        padded_state = jax.tree.map(add_dummy, state)
        padded_state = replace(
            padded_state,
            ID=padded_state.ID.at[N].set(-1),
            pos_c=padded_state.pos_c.at[N].set(jnp.array([1e10] * state.dim)),
            rad=padded_state.rad.at[N].set(0.0),
        )
        padded_pos_p = jnp.concatenate([pos_p[perm], pos_p[perm[:1]]], axis=0)

        # 4. Building Cell Matrix with Dummy Padding (N = dummy index)
        valid_slot = slots < MAX_OCC
        flat_idx = particle_hash * MAX_OCC + jnp.where(valid_slot, slots, 0)
        cell_matrix_flat = jnp.full(collider.num_cells * MAX_OCC, N, dtype=int)
        cell_matrix = (
            cell_matrix_flat.at[flat_idx]
            .set(jnp.where(valid_slot, jax.lax.iota(int, N), N))
            .reshape(-1, MAX_OCC)
        )

        # 5. Stencil Deduplication
        neighbor_offsets = collider.neighbor_mask
        if system.domain.periodic:
            wrapped_offsets = neighbor_offsets % grid_dims
            off_hashes = jnp.dot(wrapped_offsets, strides)
            first_occurrence = jnp.argmax(
                off_hashes[None, :] == off_hashes[:, None], axis=1
            )
            is_primary = jnp.arange(len(off_hashes)) == first_occurrence
        else:
            is_primary = jnp.ones(len(neighbor_offsets), dtype=bool)

        # 6. Precompute Neighbor Cell Hashes
        sorted_cell_ids = cell_indices[perm]
        neighbor_cells = sorted_cell_ids[:, None, :] + neighbor_offsets[None, :, :]
        if system.domain.periodic:
            neighbor_cells %= grid_dims
        else:
            neighbor_cells = jnp.clip(neighbor_cells, 0, grid_dims - 1)
        neighbor_hashes = jnp.dot(neighbor_cells, strides)

        # Build flat neighbor list and mask for wide vectorization
        flat_neighbors = cell_matrix[neighbor_hashes].reshape(N, -1)  # (N, M*MAX_OCC)
        flat_active = jnp.broadcast_to(
            is_primary[:, None], (len(is_primary), MAX_OCC)
        ).reshape(-1)

        def per_particle(i, pos_pi, neighbors):
            def body_fun(idx, k):
                valid = flat_active[idx] & (padded_state.ID[k] != padded_state.ID[i])
                res_f, res_t = system.force_model.force(i, k, padded_state, system)
                return valid * res_f, valid * (res_t + cross(pos_pi, res_f))

            fs, ts = jax.vmap(body_fun)(jnp.arange(len(flat_active)), neighbors)
            return fs.sum(axis=0), ts.sum(axis=0)

        total_f, total_t = jax.vmap(per_particle)(
            iota, padded_pos_p[:N], flat_neighbors
        )
        state.force = state.force.at[perm].add(total_f)
        state.torque = state.torque.at[perm].add(total_t)
        return state, system

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        collider = cast(MaterializedCellList, system.collider)
        N, MAX_OCC = state.N, collider.max_occupancy
        strides = collider.strides
        iota = jax.lax.iota(int, N)
        cell_indices = jnp.floor(
            (state.pos - system.domain.anchor) / collider.cell_size
        ).astype(int)
        if system.domain.periodic:
            cell_indices %= collider.grid_dims
        particle_hash, perm = jax.lax.sort(
            [jnp.dot(cell_indices, strides), iota], num_keys=1
        )
        is_first = jnp.concatenate(
            [jnp.array([True]), (particle_hash[1:] != particle_hash[:-1])]
        )
        slots = (
            jax.lax.associative_scan(
                lambda a, b: (jnp.where(b[1], b[0], a[0] + b[0]), a[1] | b[1]),
                (jnp.ones(N, int), is_first),
            )[0]
            - 1
        )
        cell_matrix = (
            jnp.full(collider.num_cells * MAX_OCC, N, dtype=int)
            .at[particle_hash * MAX_OCC + jnp.where(slots < MAX_OCC, slots, 0)]
            .set(jnp.where(slots < MAX_OCC, iota, N))
            .reshape(-1, MAX_OCC)
        )

        def add_dummy(x):
            return jnp.concatenate([x[perm], jnp.expand_dims(x[perm[0]], 0)], axis=0)

        padded_state = jax.tree.map(add_dummy, state)
        padded_state = replace(
            padded_state,
            ID=padded_state.ID.at[N].set(-1),
            pos_c=padded_state.pos_c.at[N].set(jnp.array([1e10] * state.dim)),
            rad=padded_state.rad.at[N].set(0.0),
        )

        neighbor_hashes = jnp.dot(
            (cell_indices[perm][:, None, :] + collider.neighbor_mask[None, :, :])
            % collider.grid_dims
            if system.domain.periodic
            else jnp.clip(
                cell_indices[perm][:, None, :] + collider.neighbor_mask[None, :, :],
                0,
                jnp.array(collider.grid_dims) - 1,
            ),
            strides,
        )

        def per_particle(i, hash_list):
            def per_cell(h):
                occupants = cell_matrix[h]

                def body(k):
                    valid = padded_state.ID[k] != padded_state.ID[i]
                    return (
                        0.5
                        * valid
                        * system.force_model.energy(i, k, padded_state, system)
                    )

                return jax.vmap(body)(occupants).sum()

            return jax.vmap(per_cell)(hash_list).sum()

        energies = jax.vmap(per_particle)(iota, neighbor_hashes)
        return jnp.zeros(N).at[perm].set(energies)


@Collider.register("NeighborList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NeighborList(Collider):
    r"""
    Neighbor List (Verlet List) collider following jax-md architectural patterns.

    This implementation uses a persistent neighbor list state, handles buffer
    overflows, and uses periodic-aware displacement for rebuild triggering.
    """

    idx: jax.Array
    """Neighbor indices of shape (N, max_neighbors). -1 or N indicates padding."""

    prev_pos: jax.Array
    """Positions at the time of the last neighbor list rebuild."""

    did_buffer_overflow: jax.Array
    """Boolean scalar indicating if the neighbor list was too small to hold all pairs."""

    update_threshold: float
    """Verlet skin distance. Rebuilds occur when max displacement > update_threshold / 2."""

    max_neighbors: int = field(metadata={"static": True})
    """Maximum number of neighbors stored per particle."""

    # Implicit Cell List properties for efficient rebuilds
    cell_size: jax.Array
    neighbor_mask: jax.Array
    max_occupancy: int = field(metadata={"static": True})
    grid_dims: Tuple[int, ...] = field(metadata={"static": True})
    strides: jax.Array = field(metadata={"static": True})

    @classmethod
    def Create(
        cls,
        state: "State",
        box_size: ArrayLike,
        max_neighbors: int = 64,
        update_threshold: float = 0.05,
        cell_size: Optional[ArrayLike] = None,
        max_occupancy: Optional[ArrayLike] = None,
        periodic: bool = True,
    ) -> Self:
        max_rad = jnp.max(state.rad)

        # Rebuild search radius must account for skin
        rebuild_radius = 2.0 * max_rad + update_threshold

        # Determine cell size for rebuild
        if cell_size is None:
            cell_size = rebuild_radius
        cell_size = jnp.asarray(cell_size, dtype=float)

        search_range = 1  # Standard for 1st-neighbor cell list rebuild
        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        if max_occupancy is None:
            min_rad = jnp.min(state.rad)
            box_vol = cell_size**state.dim
            smallest_sphere_vol = (
                (4.0 / 3.0) * jnp.pi * min_rad**3 / 0.9
                if state.dim == 3
                else jnp.pi * min_rad**2
            )
            max_occupancy = jnp.ceil(box_vol / smallest_sphere_vol) + 2

        if periodic:
            grid_dims_val = jnp.floor(jnp.asarray(box_size) / cell_size).astype(int)
        else:
            grid_dims_val = jnp.ceil(jnp.asarray(box_size) / cell_size).astype(int)

        # Ensure grid dims are at least 1
        grid_dims_val = jnp.maximum(1, grid_dims_val)
        grid_dims = tuple(map(int, grid_dims_val))

        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(jnp.array(grid_dims[:-1]))]
        )

        return cls(
            idx=jnp.full((state.N, max_neighbors), -1, dtype=int),
            prev_pos=state.pos,
            did_buffer_overflow=jnp.array(False),
            update_threshold=float(update_threshold),
            max_neighbors=int(max_neighbors),
            cell_size=cell_size,
            neighbor_mask=neighbor_mask.astype(int),
            max_occupancy=int(max_occupancy),
            grid_dims=grid_dims,
            strides=strides,
        )

    @staticmethod
    def _rebuild_list(
        state: "State", system: "System", collider: "NeighborList"
    ) -> Tuple[jax.Array, jax.Array]:
        """Rebuilds the neighbor list indices using an implicit cell list traversal."""
        N = state.N
        MAX_OCC = collider.max_occupancy
        MAX_NEIGH = collider.max_neighbors
        grid_dims = jnp.array(collider.grid_dims)
        strides = collider.strides
        iota = jax.lax.iota(int, N)

        # 1. Spatial Hash & Sort
        cell_indices = jnp.floor(
            (state.pos - system.domain.anchor) / collider.cell_size
        ).astype(int)
        if system.domain.periodic:
            cell_indices %= grid_dims
        particle_hash = jnp.dot(cell_indices, strides)
        p_hash_sorted, perm = jax.lax.sort([particle_hash, iota], num_keys=1)

        # 2. Neighborhood Probes
        neighbor_cell_indices = (
            cell_indices[:, None, :] + collider.neighbor_mask[None, :, :]
        )
        if system.domain.periodic:
            neighbor_cell_indices %= grid_dims
        neighbor_hashes = jnp.dot(neighbor_cell_indices, strides)

        # 3. Candidate Gathering
        start_indices = jnp.searchsorted(p_hash_sorted, neighbor_hashes, side="left")
        cand_offsets = jax.lax.iota(int, MAX_OCC)
        cand_ranks = start_indices[:, :, None] + cand_offsets[None, None, :]
        safe_ranks = jnp.minimum(cand_ranks, N - 1)
        valid_cand = (cand_ranks < N) & (
            p_hash_sorted[safe_ranks] == neighbor_hashes[:, :, None]
        )
        candidates = jnp.where(valid_cand, perm[safe_ranks], -1).reshape(N, -1)

        # 4. Filtering: pairwise distance check
        def filter_particle(i_idx, pos_i, rad_i, cand_list):
            pos_js = state.pos[jnp.where(cand_list >= 0, cand_list, 0)]
            rad_js = state.rad[jnp.where(cand_list >= 0, cand_list, 0)]
            rij = system.domain.displacement(pos_i, pos_js, system)
            dist_sq = jnp.sum(rij * rij, axis=-1)

            # Local cutoff for rebuild correctness including skin
            local_cutoff_sq = (rad_i + rad_js + collider.update_threshold) ** 2
            survivors = (
                (cand_list >= 0) & (cand_list != i_idx) & (dist_sq < local_cutoff_sq)
            )

            num_survivors = jnp.sum(survivors)
            overflow = num_survivors > MAX_NEIGH

            # Efficient packing: move survivors to front
            packed = jnp.sort(jnp.where(survivors, cand_list, -1), descending=True)[
                :MAX_NEIGH
            ]
            return packed, overflow

        idx, overflows = jax.vmap(filter_particle)(
            iota, state.pos, state.rad, candidates
        )
        return idx, jnp.any(overflows)

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="NeighborList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        collider = cast(NeighborList, system.collider)

        # 1. Decide Rebuild (periodic-aware displacement check)
        disp_since_last = system.domain.displacement(
            state.pos, collider.prev_pos, system
        )
        max_disp_sq = jnp.max(jnp.sum(disp_since_last**2, axis=-1))
        should_rebuild = max_disp_sq > (0.5 * collider.update_threshold) ** 2

        def rebuild_fun(_):
            idx, overflow = NeighborList._rebuild_list(state, system, collider)
            return replace(
                collider, idx=idx, prev_pos=state.pos, did_buffer_overflow=overflow
            )

        new_collider = jax.lax.cond(
            should_rebuild, rebuild_fun, lambda _: collider, None
        )
        system = replace(system, collider=new_collider)

        # 2. Lab-frame rotated contact point vector
        pos_p_rotated = state.q.rotate(state.q, state.pos_p)

        # 3. Branch-free interaction with Dummy Padding
        N = state.N

        def add_dummy(x):
            return jnp.concatenate([x, jnp.expand_dims(x[0], 0)], axis=0)

        padded_state = jax.tree.map(add_dummy, state)
        padded_state = replace(
            padded_state,
            ID=padded_state.ID.at[N].set(-1),
            pos_c=padded_state.pos_c.at[N].set(jnp.array([1e10] * state.dim)),
            rad=padded_state.rad.at[N].set(0.0),
        )

        # Map indices to dummy particle if -1
        active_list = jnp.where(new_collider.idx < 0, N, new_collider.idx)

        def per_particle(i, neighbors, pos_pi):
            def interact(j):
                valid = padded_state.ID[j] != state.ID[i]
                res_f, res_t = system.force_model.force(i, j, padded_state, system)
                # Correct torque calculation with rotated local vector
                return valid * res_f, valid * (res_t + cross(pos_pi, res_f))

            fs, ts = jax.vmap(interact)(neighbors)
            return fs.sum(axis=0), ts.sum(axis=0)

        total_f, total_t = jax.vmap(per_particle)(
            jax.lax.iota(int, N), active_list, pos_p_rotated
        )
        state.force += total_f
        state.torque += total_t
        return state, system

    @staticmethod
    @jax.jit
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        collider = cast(NeighborList, system.collider)
        N = state.N

        def add_dummy(x):
            return jnp.concatenate([x, jnp.expand_dims(x[0], 0)], axis=0)

        padded_state = jax.tree.map(add_dummy, state)
        padded_state = replace(
            padded_state,
            ID=padded_state.ID.at[N].set(-1),
            pos_c=padded_state.pos_c.at[N].set(jnp.array([1e10] * state.dim)),
            rad=padded_state.rad.at[N].set(0.0),
        )
        active_list = jnp.where(collider.idx < 0, N, collider.idx)

        def per_particle(i, neighbors):
            def interact(j):
                valid = padded_state.ID[j] != state.ID[i]
                return (
                    0.5 * valid * system.force_model.energy(i, j, padded_state, system)
                )

            return jax.vmap(interact)(neighbors).sum()

        return jax.vmap(per_particle)(jax.lax.iota(int, N), active_list)


__all__ = ["CellList", "DynamicCellList", "MaterializedCellList", "NeighborList"]
