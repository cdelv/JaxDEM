# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Naive :math:`O(N^2)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING
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
    Implementation that computes forces and potential energies using a naive :math:`O(N^2)` all-pairs interaction loop.

    Notes
    -----
    Due to its :math:`O(N^2)` complexity, `NaiveSimulator` is suitable for simulations
    with a relatively small number of particles. For larger systems, a more
    efficient spatial partitioning collider should be used. However, this collider should be the fastest
    option for small systems (:math:`<1k-5k` spheres depending on the GPU).

    Parameters
    ----------
    neighbor_mask : jax.Array
        Mask of the cells to check for contact..
    cell_size : State
        Size of each cell in the cell list. Defaults to the diameter of the smallest particle.
    max_n_contacts : System
        Maximun number of contacts a particle can have.
    """
    neighbor_mask: jax.Array
    cell_size: jax.Array

    @staticmethod
    def Create(state: "State", cell_size=None, n_cells=None, max_n_contacts: int = 8):
        if cell_size is None:
            cell_size = 2 * jnp.min(state.rad)
        cell_size = jnp.asarray(cell_size, dtype=float)

        if n_cells is None:
            max_rad = 2 * jnp.max(state.rad)
            n_cells = jnp.round(max_rad / cell_size).astype(int)
        n_cells = jnp.asarray(n_cells, dtype=int)

        ranges = [jnp.arange(-n_cells, n_cells + 1)] * state.dim
        mesh = jnp.meshgrid(*ranges, indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return CellList(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=cell_size,
            # max_n_contacts=int(max_n_contacts),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="NaiveSimulator.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        r"""
        Computes the potential energy associated with each particle using a naive :math:`O(N^2)` all-pairs loop.

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

        return jax.vmap(
            lambda i, j, st, sys: jax.vmap(
                sys.force_model.energy, in_axes=(None, 0, None, None)
            )(i, j, st, sys).sum(axis=0),
            in_axes=(0, None, None, None),
        )(iota, iota, state, system)

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="CellList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Cell-list force. Supports periodic, reflective, and free boundaries.

        Free boundaries: cells are indexed relative to the domain anchor (min corner),
        and the grid size uses ceil(box_size / cell_size) so the rightmost partially-filled cell
        is included. Periodic dims are wrapped with mod; non-periodic dims are range-checked.
        """
        N = state.N
        iota = jax.lax.iota(dtype=int, size=state.N)

        # 1) Compute the current grid
        n_cells = jnp.ceil(system.domain.box_size / system.collider.cell_size).astype(
            int
        )  # (dim,)
        weights = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(n_cells[:-1])]
        )  # Flattening weights: [1, n0, n0*n1, ...] (dim,)

        # 2) Calculate the particle hash (cell id)
        cell_idx = jnp.floor(
            (state.pos - system.domain.anchor) / system.collider.cell_size
        ).astype(int)
        cell_idx -= (
            system.domain.periodic * n_cells * jnp.floor(cell_idx / n_cells).astype(int)
        )  # (N, dim)
        particle_hash = jnp.dot(cell_idx, weights)  # (N,)

        # 3) Sort particles by cell hash
        particle_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)
        state = jax.tree_util.tree_map(lambda x: x[perm], state)
        cell_idx = cell_idx[perm]

        # 4) Precompute neighbor cell indices & hashes for ALL particles (N, M, dim)
        nbr_cell_all = (
            cell_idx[:, None, :] + system.collider.neighbor_mask[None, :, :]
        )  # (N, M, dim)
        nbr_cell_all -= (
            system.domain.periodic
            * n_cells
            * jnp.floor(nbr_cell_all / n_cells).astype(int)
        )  # (N, M, dim)

        # Validity for non-periodic dims: 0 <= idx < n_cells
        nonper_ok = (nbr_cell_all >= 0) * (nbr_cell_all < n_cells)  # (N, M, dim)
        valid_cell_all = jnp.all(
            jnp.where(system.domain.periodic, True, nonper_ok), axis=-1
        )  # (N, M)

        # Flatten neighbor cells -> ids: tensordot over dim
        nbr_hash_all = jnp.tensordot(nbr_cell_all, weights, axes=([2], [0]))  # (N, M)

        # Start indices of each cell run in the sorted particle hashes
        start_idx_all = valid_cell_all * jnp.searchsorted(
            particle_hash, nbr_hash_all, side="left", method="scan_unrolled"
        )

        # 5) Per-particle accumulation: outer vmap over N, inner vmap over M
        def per_particle(i, start_idx_i, nbr_hash_i, valid_i):
            def neighbor_fn(start_k, hash_k, valid_k):
                def cond(carry):
                    j, _, _ = carry
                    return valid_k * (j < N) * (particle_hash[j] == hash_k)

                def body(carry):
                    j, f_acc, t_acc = carry
                    f_ij, t_ij = system.force_model.force(i, j, state, system)
                    return j + 1, f_acc + f_ij, t_acc + t_ij

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

            f_cells, t_cells = jax.vmap(neighbor_fn)(start_idx_i, nbr_hash_i, valid_i)
            return f_cells.sum(axis=0), t_cells.sum(axis=0)

        total_force, total_torque = jax.vmap(
            per_particle,
            in_axes=(
                0,
                0,
                0,
                0,
            ),
        )(
            iota,
            start_idx_all,
            nbr_hash_all,
            valid_cell_all,
        )

        # 6) Convert to accelerations
        state.accel += total_force / state.mass[..., None]
        state.angAccel += total_torque / state.inertia
        return state, system


__all__ = ["CellList"]
