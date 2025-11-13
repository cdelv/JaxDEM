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
    max_n_contacts: int = field(metadata={"static": True})

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
            max_n_contacts=int(max_n_contacts),
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
        iota = (jax.lax.iota(dtype=int, size=state.N),)

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
        Cell-list force with max_n_contacts. Supports periodic, reflective, and free boundaries.

        Free boundaries: cells are indexed relative to the domain anchor (min corner), and
        the grid size uses ceil(box_size / cell_size) so the rightmost partially-filled cell
        is included. Periodic dims are wrapped with mod; non-periodic dims are range-checked.
        """
        N = state.N
        dim = state.dim

        # --- Cell grid / hashing parameters ---
        cell_size = system.collider.cell_size  # () or (dim,)
        box_size = system.domain.box_size  # (dim,)
        anchor = system.domain.anchor  # (dim,)  min corner of box
        periodic_b = system.domain.periodic  # (dim,)

        # Use ceil so the last partially-filled cell is included for free boundaries.
        n_cells = jnp.ceil(box_size / cell_size).astype(int)  # (dim,)

        # Flattening weights: [1, n0, n0*n1, ...]
        weights = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(n_cells[:-1])]
        )  # (dim,)

        # --- Particle -> cell indices (relative to anchor) ---
        # Integer cell index for each particle per dim
        cell_idx_raw = jnp.floor((state.pos - anchor) / cell_size).astype(
            int
        )  # (N, dim)

        # Wrap periodic dims using modulo; keep raw for non-periodic
        # jnp.mod handles negative indices properly (Python-style remainder).
        cell_idx = jnp.where(
            periodic_b,
            jnp.mod(cell_idx_raw, n_cells),
            cell_idx_raw,
        ).astype(
            int
        )  # (N, dim)

        # Flatten to 1D cell hash
        particle_hash = jnp.dot(cell_idx, weights)  # (N,)

        # --- Sort particles by cell hash ---
        iota = jax.lax.iota(dtype=int, size=N)
        particle_hash_sorted, perm = jax.lax.sort([particle_hash, iota], num_keys=1)

        # Reorder state & indices to sorted order
        state = jax.tree_util.tree_map(lambda x: x[perm], state)
        cell_idx = cell_idx[perm]

        # --- Neighbor mask and limits ---
        neighbor_mask = system.collider.neighbor_mask  # (M, dim) integer offsets
        M = neighbor_mask.shape[0]
        max_n_contacts = system.collider.max_n_contacts

        # --- Per-particle accumulation ---
        def per_particle(i: int, args):
            st, sys, p_hash = args

            base_cell = cell_idx[i]  # (dim,)

            # Vectorized neighbor-cell geometry for this particle
            nbr_raw = base_cell[None, :] + neighbor_mask  # (M, dim)

            # Wrap periodic dims with mod; keep raw for non-periodic
            nbr_cell = jnp.where(
                periodic_b,
                jnp.mod(nbr_raw, n_cells),
                nbr_raw,
            ).astype(
                int
            )  # (M, dim)

            # Validity for non-periodic dims: 0 <= idx < n_cells
            nonper_ok = jnp.logical_and(nbr_raw >= 0, nbr_raw < n_cells)  # (M, dim)
            valid_cell = jnp.all(
                jnp.where(periodic_b, True, nonper_ok), axis=-1
            )  # (M,)

            # Flatten neighbor-cell ids and locate their first index in the sorted hashes
            nbr_hash = jnp.dot(nbr_cell, weights)  # (M,)
            start_idx = jnp.where(
                valid_cell,
                jnp.searchsorted(p_hash, nbr_hash, side="left", method="scan_unrolled"),
                0,
            )  # (M,)

            # For each neighbor cell, walk its run [start, ...) while hash matches,
            # stopping at max_n_contacts for that cell.
            def neighbor_fn(start_k, hash_k, valid_k):
                def cond(carry):
                    j, f_acc, t_acc, cnt = carry
                    still_in = j < N
                    same = jnp.logical_and(still_in, p_hash[j] == hash_k)
                    room = cnt < max_n_contacts
                    return jnp.logical_and(valid_k, jnp.logical_and(same, room))

                def body(carry):
                    j, f_acc, t_acc, cnt = carry
                    f_ij, t_ij = sys.force_model.force(i, j, st, sys)
                    return j + 1, f_acc + f_ij, t_acc + t_ij, cnt + 1

                init = (
                    start_k,
                    jnp.zeros_like(st.pos[i]),
                    jnp.zeros_like(st.angVel[i]),
                    jnp.array(0, dtype=int),
                )
                _, f_cell, t_cell, _ = jax.lax.while_loop(cond, body, init)
                return f_cell, t_cell

            f_cells, t_cells = jax.vmap(neighbor_fn, in_axes=(0, 0, 0))(
                start_idx, nbr_hash, valid_cell
            )

            return f_cells.sum(axis=0), t_cells.sum(axis=0)

        total_force, total_torque = jax.vmap(per_particle, in_axes=(0, None))(
            jnp.arange(N, dtype=int), (state, system, particle_hash_sorted)
        )

        # Convert to accelerations
        state.accel += total_force / state.mass[..., None]
        state.angAccel += total_torque / state.inertia

        return state, system


__all__ = ["CellList"]
