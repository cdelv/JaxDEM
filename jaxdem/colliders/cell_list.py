# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N)` collider implementation."""

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


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CellList(Collider):
    neighbor_mask: jax.Array
    cell_size: jax.Array
    # Critical optimization: Fixed max neighbors per cell to allow vectorization
    max_occupancy: int = 4

    @staticmethod
    def Create(state: "State", cell_size=None, search_range=1, max_occupancy=4):
        if cell_size is None:
            cell_size = 2.0 * jnp.min(state.rad)

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return CellList(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
            max_occupancy=max_occupancy,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="CellList.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        iota = jax.lax.iota(dtype=int, size=state.N)
        MAX_OCCUPANCY = 4

        # 1. Grid geometry
        grid_dims = jnp.ceil(system.domain.box_size / system.collider.cell_size).astype(
            int
        )
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        # 2. Particle Mapping
        cell_ids = jnp.floor(
            (state.pos - system.domain.anchor) / system.collider.cell_size
        ).astype(int)

        if system.domain.periodic:
            cell_ids -= grid_dims * jnp.floor(cell_ids / grid_dims).astype(int)

        # 3. Spatial Hashing
        particle_hash = jnp.dot(cell_ids, strides)

        # 4. Sorting
        particle_hash, perm = jax.lax.sort([particle_hash, iota], num_keys=1)
        state = jax.tree.map(lambda x: x[perm], state)
        cell_ids = cell_ids[perm]

        def per_particle(i, my_cell_id):
            def per_neighbor_cell(cell_offset):
                current_cell = my_cell_id + cell_offset
                if system.domain.periodic:
                    current_cell -= grid_dims * jnp.floor(
                        current_cell / grid_dims
                    ).astype(int)
                current_cell_hash = jnp.dot(current_cell, strides)

                # Find Start Index
                start_idx = jnp.searchsorted(
                    particle_hash,
                    current_cell_hash,
                    side="left",
                    method="scan_unrolled",
                )

                def body_fun(offset):
                    k = start_idx + offset
                    safe_k = jnp.minimum(k, state.N - 1)
                    j = safe_k  # perm[safe_k]
                    valid = (k < state.N) * (particle_hash[safe_k] == current_cell_hash)
                    f, t = system.force_model.force(i, j, state, system)
                    return valid * f, valid * t

                # VMAP over the fixed capacity slots
                f_vec, t_vec = jax.vmap(body_fun)(
                    jax.lax.iota(size=MAX_OCCUPANCY, dtype=int)
                )
                return f_vec.sum(axis=0), t_vec.sum(axis=0)

                # def loop_body(offset, carry):
                #     f_acc, t_acc = carry

                #     k = start_idx + offset
                #     safe_k = jnp.minimum(k, state.N - 1)
                #     j = safe_k

                #     valid = (k < state.N) & (particle_hash[safe_k] == current_cell_hash)

                #     f_ij, t_ij = system.force_model.force(i, j, state, system)

                #     return f_acc + (valid * f_ij), t_acc + (valid * t_ij)

                # f_init = jnp.zeros_like(state.force[i])
                # t_init = jnp.zeros_like(state.torque[i])

                # return jax.lax.fori_loop(0, MAX_OCCUPANCY, loop_body, (f_init, t_init))

            # VMAP over neighbor cells
            f_cells, t_cells = jax.vmap(per_neighbor_cell)(
                system.collider.neighbor_mask
            )
            return jnp.sum(f_cells, axis=0), jnp.sum(t_cells, axis=0)

        # VMAP over all particles
        f_tot, t_tot = jax.vmap(per_particle)(iota, cell_ids)

        state.force += f_tot
        state.torque += t_tot
        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="CellList.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        # Same logic as compute_force but simplified for energy
        iota = jax.lax.iota(dtype=int, size=state.N)
        capacity = state.N
        domain = system.domain
        collider = system.collider
        max_occ = collider.max_occupancy

        grid_dims = jnp.ceil(domain.box_size / collider.cell_size).astype(int)
        strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )
        cell_ids = jnp.floor((state.pos - domain.anchor) / collider.cell_size).astype(
            int
        )

        if domain.periodic:
            cell_ids_wrapped = jnp.mod(cell_ids, grid_dims)
        else:
            cell_ids_wrapped = cell_ids

        p_raw_hash = jnp.dot(cell_ids_wrapped, strides)
        p_comp_hash = jnp.mod(p_raw_hash, capacity)
        p_comp_sorted, p_raw_sorted, perm = jax.lax.sort(
            [p_comp_hash, p_raw_hash, iota], num_keys=1
        )

        compare = p_comp_sorted[1:] != p_comp_sorted[:-1]
        first = jnp.concatenate([jnp.array([True]), compare])
        last = jnp.concatenate([compare, jnp.array([True])])
        BIG = capacity + 1
        starts = jnp.where(first, iota, BIG)
        starts = jax.ops.segment_min(starts, p_comp_sorted, num_segments=capacity)
        starts = jnp.where(starts == BIG, BIG, starts)
        ends = jnp.where(last, iota + 1, -1)
        ends = jax.ops.segment_max(ends, p_comp_sorted, num_segments=capacity)
        ends = jnp.where(ends < 0, starts, ends)

        def per_particle(i, my_cell_id):
            def per_neighbor_cell(cell_offset):
                target_cell = my_cell_id + cell_offset
                if domain.periodic:
                    target_cell = jnp.where(
                        target_cell < 0, target_cell + grid_dims, target_cell
                    )
                    target_cell = jnp.where(
                        target_cell >= grid_dims, target_cell - grid_dims, target_cell
                    )
                    is_valid_cell = True
                else:
                    is_valid_cell = jnp.all(
                        (target_cell >= 0) & (target_cell < grid_dims)
                    )

                target_raw = jnp.dot(target_cell, strides)
                target_comp = jnp.mod(target_raw, capacity)

                start_index = starts[target_comp]
                end_index = ends[target_comp]
                candidate_offsets = jax.lax.iota(int, max_occ)
                fetch_indices = jnp.minimum(
                    start_index + candidate_offsets, state.N - 1
                )
                candidate_particles = perm[fetch_indices]

                real_indices_mask = (start_index + candidate_offsets) < end_index
                hash_match_mask = p_raw_sorted[fetch_indices] == target_raw
                self_mask = candidate_particles != i
                active_mask = (
                    real_indices_mask & hash_match_mask & self_mask & is_valid_cell
                )

                safe_j = jnp.where(active_mask, candidate_particles, (i + 1) % state.N)
                e_vec = jax.vmap(
                    lambda j: system.force_model.energy(i, j, state, system)
                )(safe_j)

                # Energy is shared, divide by 2.0
                return jnp.sum((e_vec / 2.0) * active_mask)

            energies = jax.vmap(per_neighbor_cell)(collider.neighbor_mask)
            return jnp.sum(energies)

        return jnp.sum(jax.vmap(per_particle)(iota, cell_ids))


__all__ = ["CellList"]
