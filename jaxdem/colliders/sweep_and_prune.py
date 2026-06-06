# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from ..utils.linalg import cross
from . import Collider, valid_interaction_mask

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@Collider.register("SweepAndPrune")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SweepAndPrune(Collider):
    """
    A high-performance Sweep and Prune algorithm leveraging macroscopic spatial hashing and partial sorting.

    Algorithm Description:
    ----------------------
    This algorithm replaces the traditional Sweep and Prune (which relies on maintaining fully sorted arrays
    of bounding boxes along axes and testing AABB overlaps) with a hardware-accelerated "Tiled Sort and Gather"
    approach. It works by hybridizing a coarse cell-list (spatial hash) with a continuous 1D projection (sweep).

    1. The simulation domain is discretized into macroscopic blocks (cells) along D-1 perpendicular axes,
       with cell sizes determined by `2 * max_rad` (or `2 * search_limit`).
    2. Particles are hashed into these D-1 dimensional cells.
    3. To resolve boundaries (where interacting particles might lie in adjacent cells), the algorithm
       performs `2^(D-1)` passes. In each pass, the spatial grid is shifted by `search_limit` along
       combinations of the perpendicular axes.
    4. For each pass, a global hashing key `HASH_global` is constructed. This key encodes both the pass ID
       and the discretized cell index.
    5. A projection is computed along the remaining D-th axis.
    6. A single `argsort` is performed over a uniquely shifted 1D projection: `proj_shifted = proj + HASH_global * L_proj`,
       where `L_proj` is a large constant separating different cells. This `argsort` simultaneously groups particles
       by their macro-cell (bucket sort behavior) and perfectly sorts them along the 1D sweep axis within each cell.
    7. Force computation or neighbor list packing is then executed via fully vectorized relative gathers
       (evaluating `K` neighbors ahead and behind in the sorted list). Masking handles self-interactions,
       redundant pairs (from overlapping cell shifts), and clump connectivity.

    Computational Complexity & Performance Factors:
    ---------------------------------------------
    The theoretical cost is strictly O(N log N) dominated by the single global `argsort`, but practically scales
    as O(N) due to the sorting bucketing particles which are mostly ordered. The neighbor gathering overhead depends on `K`,
    which is the maximum expected occupancy in the 1D sorted list.

    - Packing Fraction: The number of neighbors checked (`K`) linearly depends on the macroscopic packing fraction.
      For standard physics (e.g., 2D packing of 0.9, 3D of 0.74), `K` remains small and constant. If particles
      are highly compressed or overlapping heavily (e.g., inside rigid clumps where `max_occupancy` is high), `K`
      increases to safely capture all overlapping spheres.
    - Max Occupancy: The maximum number of overlapping or clumped spheres directly dictates `K`. By default, `K`
      is inferred dynamically based on system volume and radii.
    - Polydispersity: High polydispersity (ratio of max to min radius) expands the macro-cell size relative to
      small particles. Because the grid resolution must accommodate `2 * max_rad`, highly disparate sizes cause
      small particles to densely populate cells, drastically increasing `K` and reducing the efficiency of the
      1D sweep pruning. The algorithm performs optimally for monodisperse or moderately polydisperse distributions.
    """

    K: int = jax.tree.static()
    max_occupancy: int = jax.tree.static(default=1)

    @classmethod
    def Create(
        cls,
        state: State,
        K: int | None = None,
        max_occupancy: int = 1,
    ):
        if K is None:
            max_rad = jnp.max(state.rad)
            min_rad = jnp.min(state.rad)
            if state.dim == 2:
                particle_volume = jnp.pi * min_rad**2
                # The maximum search volume is a cell of size 2*max_rad.
                # The centers of interacting particles can be up to min_rad outside this volume.
                expanded_cell_volume = (2.0 * max_rad + 2.0 * min_rad) ** 2
                packing_fraction = 0.90  # 2D max packing
            else:
                particle_volume = (4.0 / 3.0) * jnp.pi * min_rad**3
                expanded_cell_volume = (2.0 * max_rad + 2.0 * min_rad) ** 3
                packing_fraction = 0.74  # 3D max packing

            max_particles_in_cell = (
                expanded_cell_volume * packing_fraction
            ) / particle_volume
            K = int(jnp.ceil(max_particles_in_cell * max_occupancy).astype(int))
            # Add a small safety margin for boundary cases
            K = K + 2

        return cls(K=int(K), max_occupancy=int(max_occupancy))

    @staticmethod
    def _compute_single_sort(
        state: State,
        system: System,
        cutoff: float,
        is_neighbor_list: bool = False,
    ):
        collider = cast(SweepAndPrune, system.collider)
        pos = state.pos
        n, dim = pos.shape
        anchor = system.domain.anchor
        box_size = system.domain.box_size

        if system.domain.periodic:
            pos = pos - box_size * jnp.floor((pos - anchor) / box_size)

        max_rad = jnp.max(state.rad)
        search_limit = cutoff if is_neighbor_list else (2.0 * max_rad)
        bin_size = 2.0 * search_limit

        axis = dim - 1
        perp_axes = slice(0, dim - 1)

        proj = pos[:, axis]
        box_perp = box_size[perp_axes]
        anchor_perp = anchor[perp_axes]

        grid_dims = jnp.floor(box_perp / bin_size).astype(int)
        grid_dims = jnp.maximum(1, grid_dims)

        W = box_perp / grid_dims
        grid_strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        inv_W = 1.0 / W
        c_unshifted = ((pos[:, perp_axes] - anchor_perp) * inv_W).astype(int)
        c_unshifted = jnp.minimum(c_unshifted, grid_dims - 1)

        import itertools

        perp_combos = list(itertools.product([0, 1], repeat=dim - 1))
        num_passes = len(perp_combos)

        shifts_perp_list = []
        for combo in perp_combos:
            shift_perp_v = jnp.zeros(dim - 1)
            for i in range(dim - 1):
                val = jnp.where(combo[i] == 1, search_limit, 0.0)
                shift_perp_v = shift_perp_v.at[i].set(val)
            shifts_perp_list.append(shift_perp_v)
        shifts_perp = jnp.stack(shifts_perp_list)
        perp_combos_arr = jnp.array(perp_combos, dtype=int)

        L_proj = box_size[axis] + 2.0 * cutoff + 1.0

        pass_idx = jnp.repeat(jnp.arange(num_passes), n)
        p_idx = jnp.tile(jnp.arange(n), num_passes)

        shift_v = shifts_perp[pass_idx]

        dx_perp = pos[p_idx, perp_axes] - shift_v - anchor_perp
        wrapped_dx_perp = jnp.where(dx_perp < 0.0, dx_perp + box_perp, dx_perp)
        if system.domain.periodic:
            wrapped_dx_perp = jnp.where(
                wrapped_dx_perp >= box_perp, wrapped_dx_perp - box_perp, wrapped_dx_perp
            )

        c_shifted = (wrapped_dx_perp * inv_W).astype(int)
        c_shifted = jnp.minimum(c_shifted, grid_dims - 1)
        HASH_p = jnp.dot(c_shifted, grid_strides)

        max_hash = jnp.prod(grid_dims)
        HASH_global = pass_idx * max_hash + HASH_p

        proj_shifted = proj[p_idx] + HASH_global * L_proj
        order = jnp.argsort(proj_shifted)  # argsort gives the indices!

        sorted_p_idx = p_idx[order]
        sorted_pass_idx = pass_idx[order]
        sorted_hash = HASH_global[order]

        return (
            sorted_p_idx,
            sorted_pass_idx,
            sorted_hash,
            c_unshifted,
            perp_combos_arr,
            num_passes,
            order,
            p_idx,
            max_rad,
            search_limit,
        )

    @staticmethod
    @jax.jit
    def compute_force(state: State, system: System) -> tuple[State, System]:
        collider = cast(SweepAndPrune, system.collider)
        (
            sorted_p_idx,
            sorted_pass_idx,
            sorted_hash,
            c_unshifted,
            perp_combos_arr,
            num_passes,
            order,
            p_idx,
            max_rad,
            search_limit,
        ) = SweepAndPrune._compute_single_sort(
            state, system, cutoff=0.0, is_neighbor_list=False
        )

        sorted_c_unshifted = c_unshifted[sorted_p_idx]
        sorted_combo_p = perp_combos_arr[sorted_pass_idx]
        sorted_clump_id = state.clump_id[sorted_p_idx]
        sorted_bond_id = state.bond_id[sorted_p_idx]
        sorted_unique_id = state.unique_id[sorted_p_idx]

        pos = state.pos
        sorted_pos = pos[sorted_p_idx]
        sorted_rad = state.rad[sorted_p_idx]

        K = collider.K
        iota_k = jnp.concatenate([jnp.arange(-K, 0), jnp.arange(1, K + 1)])

        pos = state.pos
        n, dim = pos.shape
        pos_p = state._pos_p_rot

        box_size = system.domain.box_size
        inv_box_size = 1.0 / box_size

        M = n * num_passes
        idx_base = jnp.arange(M)
        idx_candidates = (idx_base[:, None] + iota_k[None, :]) % M

        neighbor_p_idx = sorted_p_idx[idx_candidates]
        neighbor_hash = sorted_hash[idx_candidates]

        in_same_cell = sorted_hash[:, None] == neighbor_hash

        idx_i = sorted_p_idx[:, None]
        idx_j = neighbor_p_idx

        c_i_unshifted = sorted_c_unshifted[:, None, :]
        c_j_unshifted = sorted_c_unshifted[idx_candidates]

        combo_canonical = c_i_unshifted != c_j_unshifted
        if dim == 2:
            not_in_prev = combo_canonical[..., 0] == sorted_combo_p[:, None, 0]
        else:
            not_in_prev = (combo_canonical[..., 0] == sorted_combo_p[:, None, 0]) & (
                combo_canonical[..., 1] == sorted_combo_p[:, None, 1]
            )

        not_self = idx_i != idx_j

        valid = valid_interaction_mask(
            sorted_clump_id[:, None],
            sorted_clump_id[idx_candidates],
            sorted_bond_id[:, None],
            sorted_unique_id[idx_candidates],
        )

        mask = in_same_cell & not_in_prev & not_self & valid

        dx = sorted_pos[:, None, :] - sorted_pos[idx_candidates]
        if system.domain.periodic:
            dx = dx - box_size * jnp.round(dx * inv_box_size)
        dist_sq = jnp.sum(dx**2, axis=-1)
        overlap_limit = sorted_rad[:, None] + sorted_rad[idx_candidates]
        overlap = dist_sq <= overlap_limit**2

        mask = mask & overlap

        f, t = system.force_model.force(idx_i, idx_j, pos, state, system)

        f_masked = f * mask[:, :, None]
        t_masked = t * mask[:, :, None]

        f_accum = jnp.sum(f_masked, axis=1)
        t_accum = jnp.sum(t_masked, axis=1)

        f_i = jax.ops.segment_sum(f_accum, sorted_p_idx, num_segments=n)
        t_i = jax.ops.segment_sum(t_accum, sorted_p_idx, num_segments=n)

        state.force = f_i
        state.torque = t_i + cross(pos_p, f_i)

        system = replace(
            system, collider=replace(system.collider, overflow=jnp.array(False))
        )

        return state, system

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[State, System, jax.Array, jax.Array]:
        collider = cast(SweepAndPrune, system.collider)
        (
            sorted_p_idx,
            sorted_pass_idx,
            sorted_hash,
            c_unshifted,
            perp_combos_arr,
            num_passes,
            order,
            p_idx,
            max_rad,
            search_limit,
        ) = SweepAndPrune._compute_single_sort(
            state, system, cutoff=cutoff, is_neighbor_list=True
        )

        sorted_c_unshifted = c_unshifted[sorted_p_idx]
        sorted_combo_p = perp_combos_arr[sorted_pass_idx]
        sorted_clump_id = state.clump_id[sorted_p_idx]
        sorted_bond_id = state.bond_id[sorted_p_idx]
        sorted_unique_id = state.unique_id[sorted_p_idx]

        pos = state.pos
        sorted_pos = pos[sorted_p_idx]

        K = collider.K
        iota_k = jnp.concatenate([jnp.arange(-K, 0), jnp.arange(1, K + 1)])

        n, dim = state.pos.shape
        pos = state.pos
        box_size = system.domain.box_size
        inv_box_size = 1.0 / box_size

        M = n * num_passes
        idx_base = jnp.arange(M)
        idx_candidates = (idx_base[:, None] + iota_k[None, :]) % M

        neighbor_p_idx = sorted_p_idx[idx_candidates]
        neighbor_hash = sorted_hash[idx_candidates]

        in_same_cell = sorted_hash[:, None] == neighbor_hash

        idx_i = sorted_p_idx[:, None]
        idx_j = neighbor_p_idx

        c_i_unshifted = sorted_c_unshifted[:, None, :]
        c_j_unshifted = sorted_c_unshifted[idx_candidates]

        combo_canonical = c_i_unshifted != c_j_unshifted
        if dim == 2:
            not_in_prev = combo_canonical[..., 0] == sorted_combo_p[:, None, 0]
        else:
            not_in_prev = (combo_canonical[..., 0] == sorted_combo_p[:, None, 0]) & (
                combo_canonical[..., 1] == sorted_combo_p[:, None, 1]
            )

        not_self = idx_i != idx_j

        valid = valid_interaction_mask(
            sorted_clump_id[:, None],
            sorted_clump_id[idx_candidates],
            sorted_bond_id[:, None],
            sorted_unique_id[idx_candidates],
        )

        mask = in_same_cell & not_in_prev & not_self & valid

        dx = sorted_pos[:, None, :] - sorted_pos[idx_candidates]
        if system.domain.periodic:
            dx = dx - box_size * jnp.round(dx * inv_box_size)
        dist_sq = jnp.sum(dx**2, axis=-1)

        overlap = dist_sq <= search_limit**2
        mask = mask & overlap

        j_list = idx_j
        mask_list = mask

        M = n * num_passes
        rank = jnp.empty_like(order).at[order].set(jnp.arange(M, dtype=order.dtype))

        unsorted_j = j_list[rank]
        unsorted_mask = mask_list[rank]

        all_candidates = (
            unsorted_j.reshape(num_passes, n, -1).transpose((1, 0, 2)).reshape(n, -1)
        )
        all_is_neighbor = (
            unsorted_mask.reshape(num_passes, n, -1).transpose((1, 0, 2)).reshape(n, -1)
        )

        all_candidates = jnp.where(all_is_neighbor, all_candidates, -1)

        # Vectorized prefix-sum packing (replaces expensive row-wise argsort)
        indices = jnp.cumsum(all_is_neighbor, axis=-1) - 1
        col_idx = jnp.where(all_is_neighbor, indices, max_neighbors)
        row_idx = jnp.arange(n)[:, None]

        nl_final = jnp.full((n, max_neighbors), -1, dtype=int)
        nl_final = nl_final.at[row_idx, col_idx].set(all_candidates, mode="drop")

        # Determine overflow: if any particle has more valid neighbors than max_neighbors
        num_valid = jnp.sum(all_is_neighbor, axis=1)
        overflow_flag = jnp.any(num_valid > max_neighbors)

        all_candidates = nl_final

        system = replace(
            system, collider=replace(system.collider, overflow=jnp.array(False))
        )

        return state, system, all_candidates, overflow_flag
