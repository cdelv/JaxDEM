# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING, cast
from functools import partial

from . import Collider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


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

    @staticmethod
    def Create(state: "State", cell_size=None, search_range=None, max_occupancy=None):
        """
        Creates a CellList collider with robust defaults.

        Defaults are chosen to avoid missing any contacts while keeping the
        neighbor stencil and assumed cell occupancy as small as possible given
        available information from `state`.

        Parameters
        ----------
        state : State
            Reference state used to determine spatial dimension and default parameters.
        cell_size : float or array-like, optional
            Cell edge length. If None, defaults to the safe cutoff distance
            `2 * max(state.rad)` which allows search_range=1.
        search_range : int, optional
            Neighbor range in cell units. If None, the smallest safe value is
            computed such that `search_range * cell_size >= cutoff`.
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
        cutoff = 2.0 * max_rad

        if cell_size is None:
            cell_size = 1.02 * cutoff
        cell_size = float(cell_size)

        if search_range is None:
            search_range = int(jnp.ceil(cutoff / cell_size))
            search_range = max(1, search_range)
        search_range = int(search_range)

        if max_occupancy is None:
            box_vol = cell_size**state.dim
            smallest_sphere_vol = 0.0
            if state.dim == 3:
                smallest_sphere_vol = (4.0 / 3.0) * jnp.pi * min_rad**3 / 0.9
            elif state.dim == 2:
                smallest_sphere_vol = jnp.pi * min_rad**2

            max_occupancy = jnp.ceil(box_vol / smallest_sphere_vol)
            max_occupancy = jnp.maximum(2, max_occupancy)
        max_occupancy = int(max_occupancy)

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return CellList(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
            max_occupancy=int(max_occupancy),
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

        # 1. Determine Grid Dimensions
        # shape: (dim,)
        grid_dims = jnp.ceil(system.domain.box_size / collider.cell_size).astype(int)

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

        def per_particle(i, my_cell_id, cell_hash):
            def per_neighbor_cell(current_cell_hash):
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

                def body_fun(offset):
                    k = start_idx + offset
                    safe_k = jnp.minimum(k, state.N - 1)
                    valid = (
                        (k < state.N)
                        * (particle_hash[safe_k] == current_cell_hash)
                        * (state.ID[safe_k] != state.ID[i])
                    )
                    result = system.force_model.force(i, safe_k, state, system)
                    forces, torques = jax.tree.map(lambda x: valid * x, result)
                    induced_torque = jnp.cross(pos[i], forces)
                    if state.dim == 2:
                        induced_torque = induced_torque[..., None]
                    torques += induced_torque

                    return forces, torques

                # VMAP over the fixed number of contacts
                result = jax.vmap(body_fun)(jax.lax.iota(size=MAX_OCCUPANCY, dtype=int))
                return jax.tree.map(lambda x: x.sum(axis=0), result)

            # VMAP over neighbor cells
            result = jax.vmap(per_neighbor_cell)(cell_hash)
            return jax.tree.map(lambda x: x.sum(axis=0), result)

        # VMAP over all particles
        total_force, total_torque = jax.vmap(per_particle)(iota, cell_ids, cell_hashes)

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

        # 1. Determine Grid Dimensions
        # shape: (dim,)
        grid_dims = jnp.ceil(system.domain.box_size / collider.cell_size).astype(int)

        # Compute strides (weights) for flattening 2D/3D indices to 1D hash
        # [1, nx, nx*ny, ...]
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        # 2. Calculate Particle Cell Indices
        cell_ids = jnp.floor(
            (state.pos - system.domain.anchor) / collider.cell_size
        ).astype(int)

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

        def per_particle(i, my_cell_id, cell_hash):
            def per_neighbor_cell(current_cell_hash):
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

                def body_fun(offset):
                    k = start_idx + offset
                    safe_k = jnp.minimum(k, state.N - 1)
                    valid = (
                        (k < state.N)
                        * (particle_hash[safe_k] == current_cell_hash)
                        * (state.ID[safe_k] != state.ID[i])
                    )
                    return valid * system.force_model.energy(i, safe_k, state, system)

                # VMAP over the fixed number of contacts
                return jax.vmap(body_fun)(
                    jax.lax.iota(size=MAX_OCCUPANCY, dtype=int)
                ).sum()

            # VMAP over neighbor cells
            return jax.vmap(per_neighbor_cell)(cell_hash).sum()

        # VMAP over all particles
        return 0.5 * jax.vmap(per_particle)(iota, cell_ids, cell_hashes).sum()

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


__all__ = ["CellList"]
