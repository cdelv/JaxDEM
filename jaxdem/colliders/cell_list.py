# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING
from functools import partial

from . import Collider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from ..domains import Domain


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CellList(Collider):
    r"""
    Implementation that computes forces and potential energies using an efficient
    implicit Cell List (Spatial Hashing) algorithm with :math:`O(N log N)` complexity.
    The algorithm is implicit because we never materialize in memory the cell list,
    making it :math:`O(N)` in memory.

    Notes
    -----
    This collider is significantly faster than `NaiveSimulator` for systems
    with a large number of particles (:math:`N > 1000`). It divides the domain
    into a grid and only checks collisions between particles in adjacent cells.

    Parameters
    ----------
    neighbor_mask : jax.Array
        Static mask of relative cell coordinates to check for neighbors (e.g., 3x3 offsets).
    cell_size : jax.Array
        The edge length of a cell.
    """

    neighbor_mask: jax.Array
    cell_size: jax.Array

    @staticmethod
    def Create(state: "State", cell_size=None, search_range=1):
        """
        Creates a CellList collider.

        Parameters
        ----------
        state : State
            Reference state to determine dimensions and default cell size.
        cell_size : float, optional
            Size of the cells. If None, defaults to 2.0 * max(state.rad).
        search_range : int
            Number of cells to search in each direction. Defaults to 1 (3x3 or 3x3x3 neighborhood).
        """
        if cell_size is None:
            # Safe default: largest possible diameter
            cell_size = 2.0 * jnp.max(state.rad)
        cell_size = jnp.asarray(cell_size, dtype=float)

        # Create the mask of neighbor offsets (e.g. -1, 0, 1 for range=1)
        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        # Stack and flatten to shape (M, dim), where M = (2*range+1)^dim
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return CellList(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=cell_size,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="CellList._build_cell_list")
    def _build_cell_list(
        state: "State", domain: "Domain", collider: "CellList"
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Builds the spatial hash and neighbor lookups.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        particle_hash : jax.Array
            Particle hashes.
        nbr_hashes : jax.Array
            The hash ID of every neighbor cell for every particle i.
        valid_cell_mask : jax.Array
            Boolean mask indicating if a neighbor cell is valid (within domain bounds).
        """
        # 1. Determine Grid Dimensions
        # shape: (dim,)
        grid_dims = jnp.ceil(domain.box_size / collider.cell_size).astype(int)

        # Compute strides (weights) for flattening 2D/3D indices to 1D hash
        # [1, nx, nx*ny, ...]
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        # 2. Calculate Particle Cell Indices
        cell_idx = jnp.floor((state.pos - domain.anchor) / collider.cell_size).astype(
            int
        )

        # Wrap indices for hashing purposes if periodic
        # system.domain.periodic is a static variable. This is a compile time if
        if domain.periodic:
            cell_idx -= grid_dims * jnp.floor(cell_idx / grid_dims).astype(int)

        # Compute 1D hash for every particle
        particle_hash = jnp.dot(cell_idx, strides)  # (N,)

        # 3. Precompute Neighbor Cell Hashes for every particle
        # (N, M, dim) = (N, 1, dim) + (1, M, dim)
        # M is number of neighbor cells (e.g., 27)
        nbr_cell_coords = cell_idx[:, None, :] + collider.neighbor_mask[None, :, :]

        # 4. Validity Masking & Hashing
        # The valid cell mask is not necesary for correctness, but it
        # Improves performance by reducing the number of iterations.
        if domain.periodic:
            # Logic: If periodic, we wrap coordinates. All neighbors are valid.
            nbr_cell_coords -= grid_dims * jnp.floor(
                nbr_cell_coords / grid_dims
            ).astype(int)
            valid_cell_mask = jnp.ones(nbr_cell_coords.shape[:-1], dtype=bool)
        else:
            # Logic: Check bounds [0, grid_dims)
            in_bounds = (nbr_cell_coords >= 0) * (nbr_cell_coords < grid_dims)
            valid_cell_mask = jnp.all(in_bounds, axis=-1)

        # Compute hashes for all neighbor cells: (N, M)
        nbr_hashes = jnp.tensordot(nbr_cell_coords, strides, axes=([2], [0]))

        return particle_hash, nbr_hashes, valid_cell_mask

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="CellList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        r"""
        Computes the total force acting on each particle using an implicit cell list :math:`O(N log N)`.

        This method sums the force contributions from all particle pairs (i, j)
        as computed by the ``system.force_model`` and updates the particle accelerations.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated ``State`` object with computed accelerations
            and the unmodified ``System`` object.
        """
        iota = jax.lax.iota(dtype=int, size=state.N)

        # 1. Build the implicit cell list
        particle_hash, nbr_hashes, valid_cell_mask = CellList._build_cell_list(
            state, system.domain, system.collider
        )

        # 2. Sort hashes
        sorted_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)

        # i iterates 0..N (Original Indices)
        def per_particle(i, nbr_hash_row, valid_row):
            # 3. Find Start Indices
            # Find where each neighbor hash starts in the sorted particle list.
            # 'searchsorted' returns the insertion index.
            # We do this inside the vmap to save memory (N, M) -> (N,)
            start_idx_row = jnp.searchsorted(
                sorted_hash, nbr_hash_row, side="left", method="scan_unrolled"
            )

            # Inner VMAP over M neighbor cells
            def compute_cell_force(start_k, target_hash, is_valid_cell):
                # While loop iterates through particles `k` (sorted index) in the target cell.
                def cond(loop_carry):
                    k, _, _ = loop_carry
                    # Valid if: cell valid AND k in bounds AND hash matches
                    return (
                        is_valid_cell * (k < state.N) * (sorted_hash[k] == target_hash)
                    )

                def body(loop_carry):
                    k, f_curr, t_curr = loop_carry
                    j = perm[k]
                    f_ij, t_ij = system.force_model.force(i, j, state, system)
                    return k + 1, f_curr + f_ij, t_curr + t_ij

                _, f_cell, t_cell = jax.lax.while_loop(
                    cond,
                    body,
                    (
                        start_k,
                        jnp.zeros_like(state.pos[i]),
                        jnp.zeros_like(state.angVel[i]),
                    ),
                )
                return f_cell, t_cell

            # Vmap over neighbor cells (M,)
            forces_M, torques_M = jax.vmap(compute_cell_force)(
                start_idx_row, nbr_hash_row, valid_row
            )

            # Reduce forces to avoid large memory usage
            return jnp.sum(forces_M, axis=0), jnp.sum(torques_M, axis=0)

        # Map over all particles
        total_force, total_torque = jax.vmap(per_particle)(
            iota, nbr_hashes, valid_cell_mask
        )

        # Update accelerations
        state.accel += total_force / state.mass[..., None]
        state.angAccel += total_torque / state.inertia

        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="CellList.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        r"""
        Computes the potential energy associated with each particle using an implicit cell list :math:`O(N log N)`.

        This method iterates over all particle pairs (i, j) and sums the potential energy
        contributions computed by the ``system.force_model``.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            One-dimensional array containing the total potential energy contribution for each particle.
        """
        iota = jax.lax.iota(dtype=int, size=state.N)

        # 1. Build the implicit cell list
        particle_hash, nbr_hashes, valid_cell_mask = CellList._build_cell_list(
            state, system.domain, system.collider
        )

        # 2. Sort hashes
        sorted_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)

        def per_particle_energy(i, nbr_hash_row, valid_row):
            start_idx_row = jnp.searchsorted(
                sorted_hash, nbr_hash_row, side="left", method="scan_unrolled"
            )

            def compute_cell_energy(start_k, target_hash, is_valid_cell):
                def cond(loop_carry):
                    k, _ = loop_carry
                    # Valid if: cell valid AND k in bounds AND hash matches
                    return (
                        is_valid_cell * (k < state.N) * (sorted_hash[k] == target_hash)
                    )

                def body(loop_carry):
                    k, e_curr = loop_carry
                    j = perm[k]
                    # Energy is halved to account for double counting (i-j and j-i)
                    e_ij = system.force_model.energy(i, j, state, system) / 2.0
                    return k + 1, e_curr + e_ij

                _, e_cell = jax.lax.while_loop(cond, body, (start_k, 0.0))

                return e_cell

            energies_M = jax.vmap(compute_cell_energy)(
                start_idx_row, nbr_hash_row, valid_row
            )
            return jnp.sum(energies_M)

        return jax.vmap(per_particle_energy)(iota, nbr_hashes, valid_cell_mask)


__all__ = ["CellList"]
