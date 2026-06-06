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


@Collider.register("SweepAndPruneDynamic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SweepAndPruneDynamic(Collider):
    block_size: int = jax.tree.static(default=4)
    max_k_steps: int = jax.tree.static(default=128)

    @classmethod
    def Create(
        cls,
        state: State,
        K: int | None = None,
    ):
        B = K if K is not None else 4
        return cls(block_size=int(B), max_k_steps=128)

    @staticmethod
    def _compute_single_sort(
        state: State,
        system: System,
        cutoff: float,
        is_neighbor_list: bool = False,
    ):
        collider = cast(SweepAndPruneDynamic, system.collider)
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
        grid_strides = jnp.concatenate([jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])])

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
            wrapped_dx_perp = jnp.where(wrapped_dx_perp >= box_perp, wrapped_dx_perp - box_perp, wrapped_dx_perp)

        c_shifted = (wrapped_dx_perp * inv_W).astype(int)
        c_shifted = jnp.minimum(c_shifted, grid_dims - 1)
        HASH_p = jnp.dot(c_shifted, grid_strides)

        max_hash = jnp.prod(grid_dims)
        HASH_global = pass_idx * max_hash + HASH_p
        
        proj_shifted = proj[p_idx] + HASH_global * L_proj
        order = jnp.argsort(proj_shifted)

        sorted_p_idx = p_idx[order]
        sorted_pass_idx = pass_idx[order]
        sorted_hash = HASH_global[order]
        
        return sorted_p_idx, sorted_pass_idx, sorted_hash, c_unshifted, perp_combos_arr, num_passes, order, p_idx, max_rad, search_limit, proj_shifted

    @staticmethod
    @jax.jit
    def compute_force(state: State, system: System) -> tuple[State, System]:
        collider = cast(SweepAndPruneDynamic, system.collider)
        (sorted_p_idx, sorted_pass_idx, sorted_hash, c_unshifted, perp_combos_arr, 
         num_passes, order, p_idx, max_rad, search_limit, proj_shifted) = SweepAndPruneDynamic._compute_single_sort(
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
        sorted_proj = proj_shifted[order]
        
        n, dim = pos.shape
        pos_p = state._pos_p_rot
        box_size = system.domain.box_size
        inv_box_size = 1.0 / box_size
        M = n * num_passes

        B = collider.block_size
        idx_base = jnp.arange(M)
        iota_base_neg = jnp.arange(-B + 1, 1)
        iota_base_pos = jnp.arange(1, B + 1)

        def cond(carry):
            k_start, continue_loop, f_accum, t_accum = carry
            return (k_start <= collider.max_k_steps) & continue_loop

        def body(carry):
            k_start, continue_loop, f_accum, t_accum = carry
            iota_k = jnp.concatenate([iota_base_neg - k_start + 1, iota_base_pos + k_start - 1])
            idx_candidates = (idx_base[:, None] + iota_k[None, :]) % M
            
            neighbor_p_idx = sorted_p_idx[idx_candidates]
            neighbor_hash = sorted_hash[idx_candidates]
            neighbor_proj = sorted_proj[idx_candidates]
            
            proj_diff = jnp.where(iota_k[None, :] > 0, neighbor_proj - sorted_proj[:, None], sorted_proj[:, None] - neighbor_proj)
            active = (proj_diff >= 0.0) & (proj_diff <= search_limit) & (sorted_hash[:, None] == neighbor_hash)
            
            c_j_unshifted = sorted_c_unshifted[idx_candidates]
            combo_canonical = sorted_c_unshifted[:, None, :] != c_j_unshifted
            if dim == 2:
                not_in_prev = combo_canonical[..., 0] == sorted_combo_p[:, None, 0]
            else:
                not_in_prev = (combo_canonical[..., 0] == sorted_combo_p[:, None, 0]) & (combo_canonical[..., 1] == sorted_combo_p[:, None, 1])
                
            idx_i = sorted_p_idx[:, None]
            idx_j = neighbor_p_idx
            not_self = idx_i != idx_j
            
            valid = valid_interaction_mask(
                sorted_clump_id[:, None],
                sorted_clump_id[idx_candidates],
                sorted_bond_id[:, None],
                sorted_unique_id[idx_candidates],
            )
            
            mask = active & not_in_prev & not_self & valid
            
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
            
            f_accum = f_accum + jnp.sum(f_masked, axis=1)
            t_accum = t_accum + jnp.sum(t_masked, axis=1)
            
            new_continue_loop = jnp.any(active[:, 0]) | jnp.any(active[:, -1])
            return k_start + B, new_continue_loop, f_accum, t_accum

        init_carry = (
            jnp.array(1, dtype=int),
            jnp.array(True),
            jnp.zeros((M, dim)),
            jnp.zeros((M, dim))
        )
        
        _, _, f_accum, t_accum = jax.lax.while_loop(cond, body, init_carry)
        
        f_i = jax.ops.segment_sum(f_accum, sorted_p_idx, num_segments=n)
        t_i = jax.ops.segment_sum(t_accum, sorted_p_idx, num_segments=n)
        
        state.force = f_i
        state.torque = t_i + cross(pos_p, f_i)
        
        system = replace(
            system,
            collider=replace(system.collider, overflow=jnp.array(False))
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
        collider = cast(SweepAndPruneDynamic, system.collider)
        (sorted_p_idx, sorted_pass_idx, sorted_hash, c_unshifted, perp_combos_arr, 
         num_passes, order, p_idx, max_rad, search_limit, proj_shifted) = SweepAndPruneDynamic._compute_single_sort(
            state, system, cutoff=cutoff, is_neighbor_list=True
        )
        
        sorted_c_unshifted = c_unshifted[sorted_p_idx]
        sorted_combo_p = perp_combos_arr[sorted_pass_idx]
        sorted_clump_id = state.clump_id[sorted_p_idx]
        sorted_bond_id = state.bond_id[sorted_p_idx]
        sorted_unique_id = state.unique_id[sorted_p_idx]
        
        pos = state.pos
        sorted_pos = pos[sorted_p_idx]
        sorted_proj = proj_shifted[order]
        
        n, dim = pos.shape
        box_size = system.domain.box_size
        inv_box_size = 1.0 / box_size
        M = n * num_passes
        
        rank = jnp.empty_like(order).at[order].set(jnp.arange(M, dtype=order.dtype))
        
        B = collider.block_size
        idx_base = jnp.arange(M)
        iota_base_neg = jnp.arange(-B + 1, 1)
        iota_base_pos = jnp.arange(1, B + 1)
        
        def cond(carry):
            k_start, continue_loop, nl_final, num_neighbors_count = carry
            return (k_start <= collider.max_k_steps) & continue_loop

        def body(carry):
            k_start, continue_loop, nl_final, num_neighbors_count = carry
            iota_k = jnp.concatenate([iota_base_neg - k_start + 1, iota_base_pos + k_start - 1])
            idx_candidates = (idx_base[:, None] + iota_k[None, :]) % M
            
            neighbor_p_idx = sorted_p_idx[idx_candidates]
            neighbor_hash = sorted_hash[idx_candidates]
            neighbor_proj = sorted_proj[idx_candidates]
            
            proj_diff = jnp.where(iota_k[None, :] > 0, neighbor_proj - sorted_proj[:, None], sorted_proj[:, None] - neighbor_proj)
            active = (proj_diff >= 0.0) & (proj_diff <= search_limit) & (sorted_hash[:, None] == neighbor_hash)
            
            c_j_unshifted = sorted_c_unshifted[idx_candidates]
            combo_canonical = sorted_c_unshifted[:, None, :] != c_j_unshifted
            if dim == 2:
                not_in_prev = combo_canonical[..., 0] == sorted_combo_p[:, None, 0]
            else:
                not_in_prev = (combo_canonical[..., 0] == sorted_combo_p[:, None, 0]) & (combo_canonical[..., 1] == sorted_combo_p[:, None, 1])
                
            idx_i = sorted_p_idx[:, None]
            idx_j = neighbor_p_idx
            not_self = idx_i != idx_j
            
            valid = valid_interaction_mask(
                sorted_clump_id[:, None],
                sorted_clump_id[idx_candidates],
                sorted_bond_id[:, None],
                sorted_unique_id[idx_candidates],
            )
            
            mask = active & not_in_prev & not_self & valid
            
            dx = sorted_pos[:, None, :] - sorted_pos[idx_candidates]
            if system.domain.periodic:
                dx = dx - box_size * jnp.round(dx * inv_box_size)
            dist_sq = jnp.sum(dx**2, axis=-1)
            
            overlap = dist_sq <= search_limit**2
            mask = mask & overlap
            
            unsorted_j = idx_j[rank].reshape(num_passes, n, 2*B)
            unsorted_mask = mask[rank].reshape(num_passes, n, 2*B)
            
            block_candidates = unsorted_j.transpose((1, 0, 2)).reshape(n, num_passes * 2 * B)
            block_mask = unsorted_mask.transpose((1, 0, 2)).reshape(n, num_passes * 2 * B)
            
            indices_in_block = jnp.cumsum(block_mask, axis=-1) - 1
            insert_idx = num_neighbors_count[:, None] + indices_in_block
            
            valid_insert = block_mask & (insert_idx < max_neighbors)
            col_idx = jnp.where(valid_insert, insert_idx, max_neighbors)
            row_idx = jnp.arange(n)[:, None]
            
            nl_new = nl_final.at[row_idx, col_idx].set(
                jnp.where(valid_insert, block_candidates, nl_final[row_idx, col_idx]),
                mode="drop"
            )
            
            count_new = num_neighbors_count + jnp.sum(block_mask, axis=-1)
            new_continue_loop = jnp.any(active[:, 0]) | jnp.any(active[:, -1])
            
            return k_start + B, new_continue_loop, nl_new, count_new

        init_carry = (
            jnp.array(1, dtype=int),
            jnp.array(True),
            jnp.full((n, max_neighbors), -1, dtype=int),
            jnp.zeros(n, dtype=int)
        )
        
        _, _, nl_final, num_neighbors_count = jax.lax.while_loop(cond, body, init_carry)

        overflow_flag = jnp.any(num_neighbors_count > max_neighbors)
        
        system = replace(
            system,
            collider=replace(system.collider, overflow=jnp.array(False))
        )
        
        return state, system, nl_final, overflow_flag
