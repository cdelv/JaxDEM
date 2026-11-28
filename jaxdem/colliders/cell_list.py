# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Cell List :math:`O(N \log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.typing import ArrayLike

import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from ..domains import SearchGeometry
from ..utils.linalg import cross, norm2
from . import Collider, valid_interaction_mask
from ._cell_lookup import _cell_starts
from ._cell_options import (
    resolve_cell_tile_capacity,
    validate_cell_capacity_estimate,
    validate_cell_options,
)
from ._partition import (
    NEIGHBOR_QUERY_BATCH_SIZE,
    PAIR_TRAVERSAL_BATCH_SIZE,
    PAIR_UNROLL,
    _build_dense_cell_table,
    _can_use_sorted_order,
    _compacted_stencil_decode,
    _dedup_periodic_stencil,
    _dedup_stencil_hashes,
    _dense_grid_stats,
    _dense_stencil_prefix_batch,
    _effective_cell_size,
    _energy_pair_fn,
    _force_only_pair_fn,
    _force_pair_fn,
    _get_spatial_partition,
    _loose_cell_aabbs,
    _neighbor_hashes_from_coords,
    _permute_state,
    _prepare_dense_grid,
    _shared_max,
    _sorted_cell_run_ends,
    _traverse_neighbor_row,
    _vec_sum,
    shared_bounded_loop,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


def _can_share_pair_loop(pair_fn: Callable[..., Any]) -> bool:
    """Recognize kernels whose system can enter the loop as an operand."""
    return (
        isinstance(pair_fn, partial)
        and pair_fn.func in (_force_pair_fn, _force_only_pair_fn, _energy_pair_fn)
        and not pair_fn.args
        and "system" in pair_fn.keywords
        and set(pair_fn.keywords).issubset({"system", "quasistatic"})
    )


def _dense_sorted_traverse(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
    *,
    prune_cells: bool,
    tile_capacity: int | None = None,
    fallback_fn: Callable[[State, System], tuple[Any, jax.Array]] | None = None,
) -> tuple[Any, jax.Array]:
    """Fused cell-sorted traversal using a compact dense cell-offset table.

    Uses an outer ``lax.scan`` over the stencil offsets so every GPU warp reads
    contiguous sorted particles and accumulates forces in registers with no
    ``(N, M, W)`` intermediate materialization and no ``while_loop`` syncs.
    When ``tile_capacity`` is provided, also tracks saturating static tile
    usage and reports tile or hash overflow.
    """
    N = state.N
    pos = state.pos
    iota = jax.lax.iota(dtype=int, size=N)
    meta, use_dense = _prepare_dense_grid(
        pos, system, cell_size, neighbor_mask, iota, prune_cells=prune_cells
    )
    perm = meta.perm
    p_cell_hash = meta.p_cell_hash
    p_cell_coords = meta.p_cell_coords
    grid_dims = meta.grid_dims
    grid_strides = meta.grid_strides
    eff_cell_size = meta.eff_cell_size
    hash_overflow = meta.hash_overflow
    cap_dense = meta.cap_dense
    periodic = meta.periodic
    has_shear = meta.has_shear
    do_prune = meta.do_prune
    shear_r = meta.shear_r
    use_compact_shear = meta.use_compact_shear

    def run_dense(cap: int) -> tuple[Any, jax.Array]:
        pos_s = pos[perm]
        coords_s = p_cell_coords[perm].astype(int)
        state_s = _permute_state(state, perm)
        dims_i = grid_dims.astype(int)
        strides_i = grid_strides.astype(int)
        offsets_mask = neighbor_mask.astype(int)

        counts, cell_offsets = _build_dense_cell_table(p_cell_hash, cap)

        search_rad_s: jax.Array | None
        cell_center: jax.Array | None
        cell_half: jax.Array | None
        if do_prune:
            search_rad_s = system.force_model.search_radii(cast(Any, state_s), system)
            cell_center, cell_half = _loose_cell_aabbs(
                pos_s - search_rad_s[:, None],
                pos_s + search_rad_s[:, None],
                p_cell_hash,
            )
        else:
            search_rad_s = None
            cell_center = cell_half = None

        zero_single = jax.tree.map(jnp.zeros_like, init_acc)
        interact_same = system.interact_same_bond_id
        # Bound per-chunk workspace for non-pruned grids so batched vmap stays
        # in L2; keep pruned grids single-pass to avoid repeating AABB checks.
        chunk_size = min(N, 65_536 if do_prune else 16_384)

        pair_partial = pair_fn if isinstance(pair_fn, partial) else None
        is_force_pair = pair_partial is not None and pair_partial.func in (
            _force_only_pair_fn,
            _force_pair_fn,
        )
        want_torque = (
            is_force_pair
            and pair_partial is not None
            and (pair_partial.func is _force_pair_fn)
        )
        quasistatic_p = (
            bool(pair_partial.keywords.get("quasistatic", False))
            if (is_force_pair and pair_partial is not None)
            else False
        )
        r_stencil = shear_r

        def eval_pure_dense(
            st_s: Any, sy: System, ps: jax.Array, disc: Any
        ) -> tuple[Any, jax.Array]:
            (
                d_coords_s,
                d_cell_offsets,
                d_dims_i,
                d_strides_i,
                d_eff_cell_size,
                d_perm,
                d_hash_overflow,
                d_cell_center,
                d_cell_half,
                d_search_rad_s,
                d_counts,
                d_grid_dims,
                d_grid_strides,
                d_neighbor_mask,
                d_offsets_mask,
                d_iota,
                d_zero_single,
                d_init_acc,
                d_interact_same,
                d_orig_state,
            ) = disc

            def stencil_prefix_batch(
                b_iota: jax.Array, b_coords: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                prune_rad = (
                    d_search_rad_s[b_iota, None, None]
                    if (do_prune and d_search_rad_s is not None)
                    else None
                )
                return _dense_stencil_prefix_batch(
                    b_iota,
                    b_coords,
                    n=N,
                    pos_s=ps,
                    dims_i=d_dims_i,
                    strides_i=d_strides_i,
                    eff_cell_size=d_eff_cell_size,
                    grid_dims=d_grid_dims,
                    grid_strides=d_grid_strides,
                    neighbor_mask=d_neighbor_mask,
                    offsets_mask=d_offsets_mask,
                    cell_offsets=d_cell_offsets,
                    system=sy,
                    cap=cap,
                    periodic=periodic,
                    has_shear=has_shear,
                    use_compact_shear=use_compact_shear,
                    shear_r=shear_r,
                    do_prune=do_prune,
                    cell_center=d_cell_center,
                    cell_half=d_cell_half,
                    prune_rad=prune_rad,
                )

            (
                pre_starts,
                _pre_lens,
                pre_pref,
                pre_totals,
                max_total,
                dense_overflow,
            ) = _dense_grid_stats(
                N,
                chunk_size,
                d_iota,
                d_coords_s,
                stencil_prefix_batch,
                cell_counts=d_counts,
                dims_i=d_dims_i,
                strides_i=d_strides_i,
                offsets_mask=d_offsets_mask,
                periodic=periodic,
                has_shear=has_shear,
                do_prune=do_prune,
                tile_capacity=tile_capacity,
                hash_overflow=d_hash_overflow,
            )

            pair_eval = (
                partial(pair_partial.func, system=sy, quasistatic=quasistatic_p)
                if (is_force_pair and pair_partial is not None)
                else pair_fn
            )

            def eval_compacted_chunk(
                b_iota: jax.Array,
                starts_m: jax.Array,
                pref_m: jax.Array,
                totals: jax.Array,
                slots: jax.Array,
            ) -> Any:
                def per_particle(
                    s: jax.Array, st_m: jax.Array, pr_m: jax.Array, tot: jax.Array
                ) -> Any:
                    k = _compacted_stencil_decode(st_m, pr_m, slots)
                    sk = jnp.minimum(k, N - 1)
                    if st_s.only_spheres:
                        eligible = sk != s
                        if st_s.bond_id.shape[-1] > 0:
                            is_bonded = jnp.any(
                                st_s.bond_id[sk] == st_s.orig_id[s][..., None], axis=-1
                            )
                            eligible = eligible & (~is_bonded | d_interact_same)
                    else:
                        eligible = valid_interaction_mask(
                            st_s.clump_id[sk],
                            st_s.clump_id[s],
                            st_s.bond_id[sk],
                            st_s.orig_id[s],
                            d_interact_same,
                        ).astype(bool)
                    valid = (slots < tot) & eligible
                    return pair_eval(d_zero_single, s, sk, ps, st_s, valid)

                raw = jax.vmap(per_particle)(b_iota, starts_m, pref_m, totals)
                return jax.tree.map(lambda a: _vec_sum(a, 1), raw)

            def eval_compacted_width(width: int) -> tuple[Any, jax.Array]:
                if width == 0:
                    zero_out = jax.tree.map(
                        lambda x: jnp.zeros((N,) + x.shape, dtype=x.dtype), d_init_acc
                    )
                    return zero_out, dense_overflow
                slots = jax.lax.iota(dtype=int, size=width)
                if N <= chunk_size:
                    assert (
                        pre_starts is not None
                        and pre_pref is not None
                        and pre_totals is not None
                    )
                    acc_s = eval_compacted_chunk(
                        d_iota, pre_starts, pre_pref, pre_totals, slots
                    )
                else:
                    full = N // chunk_size * chunk_size
                    iota_g = d_iota[:full].reshape(-1, chunk_size)
                    coords_g = d_coords_s[:full].reshape(
                        -1, chunk_size, d_coords_s.shape[-1]
                    )

                    def do_chunk(x: tuple[jax.Array, jax.Array]) -> Any:
                        bi, bc = x
                        st_c, _, pr_c, tot_c = stencil_prefix_batch(bi, bc)
                        return eval_compacted_chunk(bi, st_c, pr_c, tot_c, slots)

                    acc_s = jax.lax.map(do_chunk, (iota_g, coords_g))
                    acc_s = jax.tree.map(
                        lambda a: a.reshape((full,) + a.shape[2:]), acc_s
                    )
                    if full < N:
                        tail = do_chunk((d_iota[full:], d_coords_s[full:]))
                        acc_s = jax.tree.map(
                            lambda a, b: jnp.concatenate((a, b), axis=0), acc_s, tail
                        )
                acc_out = jax.tree.map(
                    lambda a: jnp.empty_like(a).at[d_perm].set(a, unique_indices=True),
                    acc_s,
                )
                return acc_out, dense_overflow

            def eval_general() -> tuple[Any, jax.Array]:
                if fallback_fn is not None:
                    return fallback_fn(d_orig_state, sy)
                step_w = 32
                steps = (max_total + step_w - 1) // step_w
                base_slots = jax.lax.iota(dtype=int, size=step_w)

                def eval_chunk_loop(
                    bi: jax.Array, st_c: jax.Array, pr_c: jax.Array, tot_c: jax.Array
                ) -> Any:
                    zero_c = jax.tree.map(
                        lambda x: jnp.zeros((bi.shape[0],) + x.shape, dtype=x.dtype),
                        d_init_acc,
                    )

                    def step_body(step_idx: jax.Array, a: Any) -> Any:
                        sl = step_idx * step_w + base_slots
                        delta = eval_compacted_chunk(bi, st_c, pr_c, tot_c, sl)
                        return jax.tree.map(lambda old, d: old + d, a, delta)

                    return jax.lax.fori_loop(0, steps, step_body, zero_c)

                if N <= chunk_size:
                    assert (
                        pre_starts is not None
                        and pre_pref is not None
                        and pre_totals is not None
                    )
                    acc_s = eval_chunk_loop(d_iota, pre_starts, pre_pref, pre_totals)
                else:
                    full = N // chunk_size * chunk_size
                    iota_g = d_iota[:full].reshape(-1, chunk_size)
                    coords_g = d_coords_s[:full].reshape(
                        -1, chunk_size, d_coords_s.shape[-1]
                    )

                    def do_chunk(x: tuple[jax.Array, jax.Array]) -> Any:
                        bi, bc = x
                        st_c, _, pr_c, tot_c = stencil_prefix_batch(bi, bc)
                        return eval_chunk_loop(bi, st_c, pr_c, tot_c)

                    acc_s = jax.lax.map(do_chunk, (iota_g, coords_g))
                    acc_s = jax.tree.map(
                        lambda a: a.reshape((full,) + a.shape[2:]), acc_s
                    )
                    if full < N:
                        tail = do_chunk((d_iota[full:], d_coords_s[full:]))
                        acc_s = jax.tree.map(
                            lambda a, b: jnp.concatenate((a, b), axis=0), acc_s, tail
                        )
                acc_out = jax.tree.map(
                    lambda a: jnp.empty_like(a).at[d_perm].set(a, unique_indices=True),
                    acc_s,
                )
                return acc_out, dense_overflow

            widths = (0, 16, 32, 40, 48, 64, 96)
            branch_idx = sum((max_total > w).astype(int) for w in widths)
            branches = tuple(
                partial(lambda w, _: eval_compacted_width(w), w) for w in widths
            ) + (lambda _: eval_general(),)
            return jax.lax.switch(branch_idx, branches, None)

        discrete = (
            coords_s,
            cell_offsets,
            dims_i,
            strides_i,
            eff_cell_size,
            perm,
            hash_overflow,
            cell_center,
            cell_half,
            search_rad_s,
            counts,
            grid_dims,
            grid_strides,
            neighbor_mask,
            offsets_mask,
            iota,
            zero_single,
            init_acc,
            interact_same,
            state,
        )

        if is_force_pair and r_stencil >= 1 and (not has_shear or use_compact_shear):
            from ._pallas import try_pallas_cell_forces

            @jax.custom_jvp
            def _eval_dense_maybe_pallas(
                st_s: Any, sy: System, ps: jax.Array, disc: Any
            ) -> tuple[Any, jax.Array]:
                (
                    d_coords_s,
                    d_cell_off,
                    d_dims_i,
                    d_strides_i,
                    d_eff_cs,
                    d_perm,
                    d_hov,
                    d_cc,
                    d_ch,
                    d_sr,
                    *_rest,
                ) = disc
                res = try_pallas_cell_forces(
                    st_s,
                    sy,
                    ps,
                    d_coords_s,
                    d_cell_off,
                    d_dims_i,
                    d_strides_i,
                    d_eff_cs,
                    d_perm,
                    d_hov,
                    r_stencil,
                    do_prune=do_prune,
                    cell_center=d_cc,
                    cell_half=d_ch,
                    search_rad_s=d_sr,
                    tile_capacity=tile_capacity,
                    want_torque=want_torque,
                    quasistatic=quasistatic_p,
                    pure_fallback=lambda: eval_pure_dense(st_s, sy, ps, disc),
                )
                if res is not None:
                    return res
                return eval_pure_dense(st_s, sy, ps, disc)

            @_eval_dense_maybe_pallas.defjvp
            def _eval_dense_jvp(primals: Any, tangents: Any) -> Any:
                p_st, p_sy, p_ps, p_disc = primals
                t_st, t_sy, t_ps, _ = tangents
                return (
                    _eval_dense_maybe_pallas(p_st, p_sy, p_ps, p_disc),
                    jax.jvp(
                        lambda a, b, c: eval_pure_dense(a, b, c, p_disc),
                        (p_st, p_sy, p_ps),
                        (t_st, t_sy, t_ps),
                    )[1],
                )

            return _eval_dense_maybe_pallas(state_s, system, pos_s, discrete)

        return eval_pure_dense(state_s, system, pos_s, discrete)

    def run_fallback() -> tuple[Any, jax.Array]:
        if fallback_fn is not None:
            return fallback_fn(state, system)
        stencil = _neighbor_hashes_from_coords(
            p_cell_coords[perm],
            grid_dims,
            grid_strides,
            eff_cell_size,
            neighbor_mask,
            system.domain,
        )
        stencil = _dedup_periodic_stencil(
            stencil, grid_dims, neighbor_mask, system.domain
        )
        starts = _cell_starts(p_cell_hash, stencil)
        safe = jnp.minimum(starts, N - 1)
        present = (starts < N) & (p_cell_hash[safe] == stencil)
        ends = jnp.where(present, _sorted_cell_run_ends(p_cell_hash)[safe], starts)
        pos_s = pos[perm]
        state_s = _permute_state(state, perm)
        if prune_cells and not has_shear:
            search_rad_s = system.force_model.search_radii(cast(Any, state_s), system)
            cell_center, cell_half = _loose_cell_aabbs(
                pos_s - search_rad_s[:, None],
                pos_s + search_rad_s[:, None],
                p_cell_hash,
            )
            dr_cell = system.domain.displacement(
                pos_s[:, None, :], cell_center[safe], system
            )
            overlap = jnp.all(
                jnp.abs(dr_cell) <= search_rad_s[:, None, None] + cell_half[safe],
                axis=-1,
            )
            ends = jnp.where(overlap, ends, starts)

        zero_single = jax.tree.map(jnp.zeros_like, init_acc)
        zero_N = jax.tree.map(
            lambda x: jnp.zeros((N,) + x.shape, dtype=x.dtype), init_acc
        )
        interact_same = system.interact_same_bond_id
        max_run = _shared_max(jnp.max(ends - starts, initial=0))
        steps = (max_run + PAIR_UNROLL - 1) // PAIR_UNROLL
        cols = (starts.T, ends.T)
        unroll = min(int(starts.shape[1]), 3)

        def col_step(acc: Any, col: tuple[jax.Array, jax.Array]) -> tuple[Any, None]:
            st, en = col

            def step_body(step_idx: jax.Array, a: Any) -> Any:
                base = st + step_idx * PAIR_UNROLL
                for lane in range(PAIR_UNROLL):
                    k = base + lane
                    safe_k = jnp.minimum(k, N - 1)
                    valid = (k < en) & valid_interaction_mask(
                        state_s.clump_id[safe_k],
                        state_s.clump_id,
                        state_s.bond_id[safe_k],
                        state_s.orig_id,
                        interact_same,
                    ).astype(bool)
                    delta = jax.vmap(
                        lambda s, sk, v: pair_fn(zero_single, s, sk, pos_s, state_s, v)
                    )(iota, safe_k, valid)
                    a = jax.tree.map(lambda x, d: x + d, a, delta)
                return a

            acc = jax.lax.fori_loop(0, steps, step_body, acc)
            return acc, None

        acc_s, _ = jax.lax.scan(col_step, zero_N, cols, unroll=unroll)
        acc_out = jax.tree.map(
            lambda a: jnp.empty_like(a).at[perm].set(a, unique_indices=True),
            acc_s,
        )
        return acc_out, hash_overflow

    return jax.lax.cond(
        use_dense,
        lambda _: run_dense(cap_dense),
        lambda _: run_fallback(),
        None,
    )


@jax.jit(inline=True, static_argnames=("pair_fn", "prune_cells"))
def _traverse_pairs_legacy(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
    *,
    prune_cells: bool = False,
) -> tuple[Any, jax.Array]:
    """Fold a per-pair kernel over candidate pairs of the cell partition."""
    N = state.N
    if N == 0:
        empty = jax.tree.map(lambda x: jnp.empty((0,) + x.shape, x.dtype), init_acc)
        return empty, jnp.asarray(False)
    pos = state.pos
    j_arr = jax.lax.iota(dtype=int, size=PAIR_UNROLL)
    iota = jax.lax.iota(dtype=int, size=N)
    (
        perm,
        p_cell_hash,
        p_neighbor_cell_hashes,
        hash_overflow,
    ) = _get_spatial_partition(pos, system, cell_size, neighbor_mask, iota)
    cell_starts_fn = getattr(
        sys.modules[type(system.collider).__module__], "_cell_starts", _cell_starts
    )
    p_neighbor_cell_starts = cell_starts_fn(p_cell_hash, p_neighbor_cell_hashes)

    # Shear-periodic domains skip orthogonal loose-cell AABB pruning.
    prune = prune_cells and not hasattr(system.domain, "gamma")
    if prune:
        search_rad = system.force_model.search_radii(state, system)
        cell_center, cell_half = _loose_cell_aabbs(
            (pos - search_rad[:, None])[perm],
            (pos + search_rad[:, None])[perm],
            p_cell_hash,
        )

    def per_particle(
        orig_idx: jax.Array,
        neighbor_hashes: jax.Array,
        neighbor_starts: jax.Array,
    ) -> Any:
        if system.domain.periodic:
            neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes)

        def per_cell(target_hash: jax.Array, start_idx: jax.Array) -> Any:
            if prune:
                safe_start = jnp.minimum(start_idx, N - 1)
                dr_cell = system.domain.displacement(
                    pos[orig_idx], cell_center[safe_start], system
                )
                aabb_overlap = jnp.all(
                    jnp.abs(dr_cell) <= search_rad[orig_idx] + cell_half[safe_start]
                )
                cell_overlap = (
                    (start_idx < N)
                    & (p_cell_hash[safe_start] == target_hash)
                    & aabb_overlap
                )

            def cond_fun(val: tuple[jax.Array, Any]) -> bool:
                k, _ = val
                in_run = (k < N) & (p_cell_hash[jnp.minimum(k, N - 1)] == target_hash)
                return cast(bool, in_run & cell_overlap if prune else in_run)

            def body_fun(val: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
                k, acc = val
                kj = jnp.minimum(k + j_arr, N - 1)
                orig_kj = perm[kj]
                in_cell = ((k + j_arr) < N) & (p_cell_hash[kj] == target_hash)
                valid = in_cell * valid_interaction_mask(
                    state.clump_id[orig_kj],
                    state.clump_id[orig_idx],
                    state.bond_id[orig_kj],
                    orig_idx,
                    system.interact_same_bond_id,
                )
                zero_acc = jax.tree.map(jnp.zeros_like, acc)
                vec_acc = pair_fn(zero_acc, orig_idx, orig_kj, pos, state, valid)
                acc = jax.tree.map(lambda a, v: a + jnp.sum(v, axis=0), acc, vec_acc)
                return k + PAIR_UNROLL, acc

            _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (start_idx, init_acc))
            return final_acc

        cell_results = jax.vmap(per_cell)(neighbor_hashes, neighbor_starts)
        return jax.tree.map(lambda x: x.sum(axis=0), cell_results)

    batch_size = getattr(
        sys.modules[type(system.collider).__module__],
        "PAIR_TRAVERSAL_BATCH_SIZE",
        PAIR_TRAVERSAL_BATCH_SIZE,
    )
    acc = jax.lax.map(
        lambda row: per_particle(*row),
        (iota, p_neighbor_cell_hashes, p_neighbor_cell_starts),
        batch_size=min(N, batch_size),
    )
    return acc, hash_overflow


@partial(
    jax.jit,
    inline=True,
    static_argnames=("pair_impl", "pair_keywords", "prune_cells"),
)
def _traverse_shared_pairs(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_impl: Callable[..., Any],
    pair_keywords: tuple[tuple[str, Any], ...],
    init_acc: Any,
    *,
    pair_system: System,
    prune_cells: bool,
) -> tuple[Any, jax.Array]:
    """Fold known contact kernels with one bound across explicit batch axes."""
    kwargs = dict(pair_keywords)
    n = state.N
    if n == 0:
        return jax.tree.map(
            lambda x: jnp.empty((0,) + x.shape, x.dtype), init_acc
        ), jnp.asarray(False)
    rows = jnp.arange(n)
    pos = state.pos
    perm, hashes, targets, overflow = _get_spatial_partition(
        pos, system, cell_size, neighbor_mask, rows
    )
    cell_starts_fn = getattr(
        sys.modules[type(system.collider).__module__], "_cell_starts", _cell_starts
    )
    starts = cell_starts_fn(hashes, targets)
    if system.domain.periodic:
        targets = jax.vmap(_dedup_stencil_hashes)(targets)
    safe = jnp.minimum(starts, n - 1)
    present = (starts < n) & (hashes[safe] == targets)
    ends = jnp.where(present, _sorted_cell_run_ends(hashes)[safe], starts)
    if prune_cells and not hasattr(system.domain, "gamma"):
        radii = system.force_model.search_radii(state, system)
        center, half = _loose_cell_aabbs(
            (pos - radii[:, None])[perm], (pos + radii[:, None])[perm], hashes
        )
        delta = system.domain.displacement(pos[:, None, :], center[safe], system)
        overlap = jnp.all(jnp.abs(delta) <= radii[:, None, None] + half[safe], axis=-1)
        ends = jnp.where(overlap, ends, starts)

    def body(step: jax.Array, accumulators: Any, operands: Any) -> Any:
        indices, begin, end, perm, pos, state, system, bound_system = operands
        cursor = begin + step * PAIR_UNROLL

        def row(
            orig: jax.Array, row_cursor: jax.Array, row_end: jax.Array, row_acc: Any
        ) -> Any:
            def cell(k: jax.Array, stop: jax.Array, acc: Any) -> Any:
                candidate = k + jnp.arange(PAIR_UNROLL)
                other = perm[jnp.minimum(candidate, n - 1)]
                valid = (candidate < stop) & valid_interaction_mask(
                    state.clump_id[other],
                    state.clump_id[orig],
                    state.bond_id[other],
                    orig,
                    system.interact_same_bond_id,
                ).astype(bool)
                values = pair_impl(
                    jax.tree.map(jnp.zeros_like, acc),
                    orig,
                    other,
                    pos,
                    state,
                    valid,
                    system=bound_system,
                    **kwargs,
                )
                return jax.tree.map(
                    lambda old, values: old + values.sum(axis=0), acc, values
                )

            return jax.vmap(cell)(row_cursor, row_end, row_acc)

        return jax.vmap(row)(indices, cursor, end, accumulators)

    loop = shared_bounded_loop(body)

    def batch(args: tuple[jax.Array, jax.Array, jax.Array]) -> Any:
        indices, begin, end = args
        steps = jnp.max((end - begin + PAIR_UNROLL - 1) // PAIR_UNROLL, initial=0)
        initial = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (indices.shape[0], begin.shape[1]) + x.shape),
            init_acc,
        )
        result = loop(
            steps, initial, (indices, begin, end, perm, pos, state, system, pair_system)
        )
        return jax.tree.map(lambda x: x.sum(axis=1), result)

    batch_size = getattr(
        sys.modules[type(system.collider).__module__],
        "PAIR_TRAVERSAL_BATCH_SIZE",
        PAIR_TRAVERSAL_BATCH_SIZE,
    )
    width = min(n, batch_size)
    full = n // width * width
    inputs = (rows, starts, ends)
    grouped = jax.tree.map(
        lambda x: x[:full].reshape((full // width, width) + x.shape[1:]), inputs
    )
    values = jax.lax.map(batch, grouped)
    values = jax.tree.map(lambda x: x.reshape((full,) + x.shape[2:]), values)
    if full < n:
        tail = batch(jax.tree.map(lambda x: x[full:], inputs))
        values = jax.tree.map(lambda x, y: jnp.concatenate((x, y)), values, tail)
    return values, overflow


@jax.jit(inline=True, static_argnames=("pair_fn", "prune_cells"))
def _traverse_pairs_impl(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
    *,
    prune_cells: bool,
) -> tuple[Any, jax.Array]:
    """Keep scalar traversal and share loop bounds under explicit batching."""
    if state.N == 0:
        empty = jax.tree.map(lambda x: jnp.empty((0,) + x.shape, x.dtype), init_acc)
        return empty, jnp.asarray(False)

    mod = sys.modules[type(system.collider).__module__]
    cell_starts_fn = getattr(mod, "_cell_starts", _cell_starts)
    batch_size = getattr(mod, "PAIR_TRAVERSAL_BATCH_SIZE", PAIR_TRAVERSAL_BATCH_SIZE)
    if (
        _can_share_pair_loop(pair_fn)
        and _can_use_sorted_order(state, system)
        and cell_starts_fn is _cell_starts
        and batch_size == PAIR_TRAVERSAL_BATCH_SIZE
    ):
        # Preserve manual-axis variance from pair_fn without extra sorting.
        prototype = pair_fn(
            jax.tree.map(jnp.zeros_like, init_acc),
            jnp.asarray(0, dtype=int),
            jnp.zeros((1,), dtype=int),
            state.pos,
            state,
            jnp.zeros((1,), dtype=bool),
        )
        init_acc = jax.tree.map(
            lambda start, result: start + jnp.zeros_like(result.sum(axis=0)),
            init_acc,
            prototype,
        )
        return _dense_sorted_traverse(
            state,
            system,
            cell_size,
            neighbor_mask,
            pair_fn,
            init_acc,
            prune_cells=prune_cells,
        )

    if not _can_share_pair_loop(pair_fn):
        return _traverse_pairs_legacy(
            state,
            system,
            cell_size,
            neighbor_mask,
            pair_fn,
            init_acc,
            prune_cells=prune_cells,
        )

    partial_fn = cast(partial[Any], pair_fn)
    pair_impl = partial_fn.func
    pair_system = partial_fn.keywords["system"]
    pair_keywords = tuple(
        (k, v) for k, v in partial_fn.keywords.items() if k != "system"
    )
    pair_kw_dict = dict(pair_keywords)

    args_leaves, args_tree = jax.tree_util.tree_flatten(
        (state, system, pair_system, cell_size, neighbor_mask, init_acc)
    )

    def scalar_dispatch(leaves: list[Any]) -> Any:
        state, system, bound_system, cell_size, neighbor_mask, initial = (
            jax.tree_util.tree_unflatten(args_tree, leaves)
        )
        rebound = partial(pair_impl, system=bound_system, **pair_kw_dict)
        if state.N and neighbor_mask.shape[0]:
            prototype = rebound(
                jax.tree.map(jnp.zeros_like, initial),
                jnp.asarray(0, dtype=int),
                jnp.zeros((PAIR_UNROLL,), dtype=int),
                state.pos,
                state,
                jnp.zeros((PAIR_UNROLL,), dtype=bool),
            )
            initial = jax.tree.map(
                lambda start, result: start + jnp.zeros_like(result.sum(axis=0)),
                initial,
                prototype,
            )
        return _traverse_pairs_legacy(
            state,
            system,
            cell_size,
            neighbor_mask,
            rebound,
            initial,
            prune_cells=prune_cells,
        )

    @custom_vmap
    def dispatch(leaves: list[Any]) -> Any:
        return scalar_dispatch(leaves)

    @dispatch.def_vmap
    def mapped(axis_size: int, in_batched: Any, leaves: list[Any]) -> Any:
        (leaves_batched,) = in_batched
        if not any(leaves_batched):
            result = scalar_dispatch(leaves)
            return result, jax.tree.map(lambda _: False, result)
        axes = [0 if batched else None for batched in leaves_batched]

        def one(local_leaves: list[Any]) -> Any:
            st, sys_, bsys, cs, nm, init = jax.tree_util.tree_unflatten(
                args_tree, local_leaves
            )
            return _traverse_shared_pairs(
                st,
                sys_,
                cs,
                nm,
                pair_impl,
                pair_keywords,
                init,
                pair_system=bsys,
                prune_cells=prune_cells,
            )

        result = jax.vmap(one, in_axes=(axes,), axis_size=axis_size)(leaves)
        return result, jax.tree.map(lambda _: True, result)

    return dispatch(args_leaves)


@jax.jit(inline=True, static_argnames=("pair_fn",))
def _traverse_pairs(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
) -> tuple[Any, jax.Array]:
    """Fold a per-pair kernel over candidate pairs of the cell partition."""
    return _traverse_pairs_impl(
        state, system, cell_size, neighbor_mask, pair_fn, init_acc, prune_cells=False
    )


def _prune_stencil(
    pos_i: jax.Array,
    stencil: jax.Array,
    cell_starts: jax.Array,
    cell_center: jax.Array,
    cell_half: jax.Array,
    n_db: int,
    cutoff: float,
    system: System,
) -> jax.Array:
    """Mask stencil cells whose loose AABB cannot reach within ``cutoff``."""
    safe_starts = jnp.minimum(cell_starts, n_db - 1)
    dr_cell = system.domain.displacement(pos_i, cell_center[safe_starts], system)
    overlap = jnp.all(jnp.abs(dr_cell) <= cutoff + cell_half[safe_starts], axis=-1)
    sentinel = jnp.bitwise_not(jnp.asarray(0, stencil.dtype))
    return jnp.where(overlap, stencil, sentinel)


@dataclass(slots=True)
class _CellListBase(Collider):
    """Shared implementation for grid-based cell-list colliders."""

    supported_search_geometries = frozenset(
        (SearchGeometry.ORTHOGONAL, SearchGeometry.SHEAR_PERIODIC)
    )

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a grid cell (scalar)."""

    tile_capacity: int | None = jax.tree.static(default=None)
    """Global tile budget; None selects the occupancy-dependent traversal."""

    clump_aware: bool = jax.tree.static(default=False)
    """Exclude same-clump members before static tile allocation."""

    _auto_capacity: bool = jax.tree.static(default=False)
    """Refine the constructor estimate from the full reference system."""

    _prune_cells: ClassVar[bool] = False

    @property
    def stateful(self) -> bool:
        return True

    @classmethod
    def _build(
        cls,
        state: State,
        cell_size: jax.Array,
        max_rad: jax.Array,
        search_range: ArrayLike | None,
        box_size: ArrayLike | None,
        tile_capacity: int | None,
        clump_aware: bool,
        static: bool,
        _auto_capacity: bool,
    ) -> Self:
        cell_size = jnp.asarray(cell_size, dtype=float)
        if cell_size.ndim != 0 or (
            not isinstance(cell_size, jax.core.Tracer)
            and not bool(jnp.isfinite(cell_size) & (cell_size > 0))
        ):
            raise ValueError("cell_size must be a finite positive scalar")

        search_range_int: int | None = None
        if search_range is not None:
            if isinstance(search_range, bool):
                raise ValueError("search_range must be a positive integer")
            if isinstance(search_range, int):
                search_range_int = search_range
            else:
                sr_arr = jnp.asarray(search_range)
                if sr_arr.ndim != 0:
                    raise ValueError("search_range must be a positive integer")
                if isinstance(sr_arr, jax.core.Tracer):
                    raise ValueError(
                        f"{cls.__name__}.Create requires a Python integer `search_range` "
                        "when called while tracing."
                    )
                sr_val = float(sr_arr)
                if not bool(jnp.isfinite(sr_arr)) or sr_val != int(sr_val):
                    raise ValueError("search_range must be a positive integer")
                search_range_int = int(sr_val)
            if search_range_int < 1:
                raise ValueError("search_range must be a positive integer")

        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            for _ in range(2):
                sr = (
                    jnp.maximum(1, jnp.ceil(2 * max_rad / cell_size).astype(int))
                    if search_range is None
                    else jnp.asarray(search_range_int, dtype=int)
                )
                grid_dims = jnp.maximum(
                    jnp.floor(box_size / cell_size).astype(int), 2 * sr + 1
                )
                cell_size = jnp.min(box_size / grid_dims)

        if search_range is None:
            if isinstance(cell_size, jax.core.Tracer) or isinstance(
                max_rad, jax.core.Tracer
            ):
                raise ValueError(
                    f"{cls.__name__}.Create requires explicit static `search_range` "
                    "when `cell_size` is traced."
                )
            search_range_int = int(
                jnp.maximum(1, jnp.ceil(2 * max_rad / cell_size).astype(int)).item()
            )
        resolved_sr = cast(int, search_range_int)
        r = jnp.arange(-resolved_sr, resolved_sr + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)
        auto_capacity = _auto_capacity or (static and tile_capacity is None)
        tile_capacity = resolve_cell_tile_capacity(
            state, cell_size, neighbor_mask.shape[0], tile_capacity, static
        )

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
            tile_capacity=tile_capacity,
            clump_aware=clump_aware,
            _auto_capacity=auto_capacity,
        )

    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
    @partial(jax.named_call, name="DynamicCellList.compute_force")
    def compute_force(
        state: State,
        system: System,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[State, System]:
        """Compute pairwise contact forces and torques with a cell-list collider."""
        collider = cast(_CellListBase, system.collider)
        if collider.tile_capacity is not None:
            from ._static_cell import _static_compute_force

            return _static_compute_force(
                state, system, advance_history=advance_history, quasistatic=quasistatic
            )

        search_radius = jnp.max(
            system.force_model.search_radii(state, system), initial=0.0
        )
        system = system.domain.update_bounds(state.pos, system, padding=search_radius)
        cell_size = _effective_cell_size(collider, 2.0 * search_radius)
        if not system.force_model.has_pair_torque:
            sum_f, hash_overflow = _traverse_pairs_impl(
                state,
                system,
                cell_size,
                collider.neighbor_mask,
                partial(_force_only_pair_fn, system=system, quasistatic=quasistatic),
                jnp.zeros_like(state.force, shape=state.force.shape[1:]),
                prune_cells=collider._prune_cells,
            )
            state.force = sum_f
            state.torque = (
                jnp.zeros_like(state.torque)
                if state.only_spheres
                else cross(state._pos_p_rot, sum_f)
            )
        else:
            (sum_f, sum_t), hash_overflow = _traverse_pairs_impl(
                state,
                system,
                cell_size,
                collider.neighbor_mask,
                partial(_force_pair_fn, system=system, quasistatic=quasistatic),
                (
                    jnp.zeros_like(state.force, shape=state.force.shape[1:]),
                    jnp.zeros_like(state.torque, shape=state.torque.shape[1:]),
                ),
                prune_cells=collider._prune_cells,
            )
            state.force = sum_f
            state.torque = (
                sum_t if state.only_spheres else sum_t + cross(state._pos_p_rot, sum_f)
            )
        system.collider.overflow = hash_overflow
        return state, system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="DynamicCellList.compute_potential_energy")
    def compute_potential_energy(
        state: State, system: System
    ) -> tuple[State, System, jax.Array]:
        """Compute the total non-bonded potential energy of the system."""
        collider = cast(_CellListBase, system.collider)
        if collider.tile_capacity is not None:
            from ._static_cell import _static_compute_potential_energy

            return _static_compute_potential_energy(state, system)

        search_radius = jnp.max(
            system.force_model.search_radii(state, system), initial=0.0
        )
        system = system.domain.update_bounds(state.pos, system, padding=search_radius)
        cell_size = _effective_cell_size(collider, 2.0 * search_radius)
        energy, hash_overflow = _traverse_pairs_impl(
            state,
            system,
            cell_size,
            collider.neighbor_mask,
            partial(_energy_pair_fn, system=system),
            jnp.zeros_like(state.pos_c, shape=(), dtype=float),
            prune_cells=collider._prune_cells,
        )
        system.collider.overflow = hash_overflow
        return state, system, jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        """Create a neighbor list of shape (N, max_neighbors)."""
        collider = cast(_CellListBase, system.collider)
        if collider.tile_capacity is not None:
            from ._static_cell import _static_create_neighbor_list

            return _static_create_neighbor_list(state, system, cutoff, max_neighbors)

        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        cutoff_sq = cutoff**2
        N = state.N
        if N == 0:
            return (
                state,
                system,
                jnp.empty((0, max_neighbors), dtype=int),
                jnp.asarray(False),
            )
        system = system.domain.update_bounds(state.pos, system, padding=cutoff)
        iota = jax.lax.iota(int, N)
        pos = state.pos
        cell_size = _effective_cell_size(collider, jnp.asarray(cutoff))
        (
            perm,
            p_cell_hash,
            p_neighbor_hashes,
            hash_overflow,
        ) = _get_spatial_partition(pos, system, cell_size, collider.neighbor_mask, iota)
        mod = sys.modules[type(collider).__module__]
        cell_starts_fn = getattr(mod, "_cell_starts", _cell_starts)
        query_batch_size = getattr(
            mod, "NEIGHBOR_QUERY_BATCH_SIZE", NEIGHBOR_QUERY_BATCH_SIZE
        )
        p_neighbor_starts = cell_starts_fn(p_cell_hash, p_neighbor_hashes)

        prune = collider._prune_cells and not hasattr(system.domain, "gamma")
        if prune:
            permuted_pos = pos[perm]
            cell_center, cell_half = _loose_cell_aabbs(
                permuted_pos, permuted_pos, p_cell_hash
            )

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            stencil: jax.Array,
            cell_starts: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil)
            if prune:
                stencil = _prune_stencil(
                    pos_i,
                    stencil,
                    cell_starts,
                    cell_center,
                    cell_half,
                    N,
                    cutoff,
                    system,
                )

            def candidate_valid(k: jax.Array) -> jax.Array:
                orig_k = perm[k]
                dr = system.domain.displacement(pos_i, pos[orig_k], system)
                return valid_interaction_mask(
                    state.clump_id[orig_k],
                    state.clump_id[idx],
                    state.bond_id[orig_k],
                    idx,
                    system.interact_same_bond_id,
                ).astype(bool) & (norm2(dr) <= cutoff_sq)

            return _traverse_neighbor_row(
                stencil, cell_starts, p_cell_hash, N, max_neighbors, candidate_valid
            )

        if max_neighbors == 0:
            topk, row_overflows = jax.vmap(traverse)(
                iota, pos, p_neighbor_hashes, p_neighbor_starts
            )
        else:
            topk, row_overflows = jax.lax.map(
                lambda args: traverse(*args),
                (iota, pos, p_neighbor_hashes, p_neighbor_starts),
                batch_size=min(N, query_batch_size),
            )

        topk = jnp.where(topk != -1, perm[jnp.maximum(topk, 0)], -1)
        return state, system, topk, jnp.any(row_overflows) | hash_overflow

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="DynamicCellList.create_cross_neighbor_list")
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Create a cross-neighbor list between pos_a (query) and pos_b (database)."""
        collider = cast(_CellListBase, system.collider)
        if collider.tile_capacity is not None:
            from ._static_cell import _static_create_cross_neighbor_list

            return _static_create_cross_neighbor_list(
                pos_a, pos_b, system, cutoff, max_neighbors
            )

        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        n_a, n_b = pos_a.shape[0], pos_b.shape[0]
        if n_a == 0:
            return jnp.empty((0, max_neighbors), dtype=int), jnp.asarray(False)
        if n_b == 0:
            return jnp.full((n_a, max_neighbors), -1, dtype=int), jnp.asarray(False)

        system = system.domain.update_bounds(
            jnp.concatenate((pos_a, pos_b), axis=0), system, padding=cutoff
        )
        cell_size = _effective_cell_size(collider, jnp.asarray(cutoff))

        perm_b, p_cell_hash_b, _, hash_overflow_b = _get_spatial_partition(
            pos_b, system, cell_size, collider.neighbor_mask, jax.lax.iota(int, n_b)
        )
        _, _, p_neighbor_hashes_a, hash_overflow_a = _get_spatial_partition(
            pos_a, system, cell_size, collider.neighbor_mask, jax.lax.iota(int, n_a)
        )
        mod = sys.modules[type(collider).__module__]
        cell_starts_fn = getattr(mod, "_cell_starts", _cell_starts)
        query_batch_size = getattr(
            mod, "NEIGHBOR_QUERY_BATCH_SIZE", NEIGHBOR_QUERY_BATCH_SIZE
        )
        p_neighbor_starts_a = cell_starts_fn(p_cell_hash_b, p_neighbor_hashes_a)

        prune = collider._prune_cells and not hasattr(system.domain, "gamma")
        if prune:
            pos_b_permuted = pos_b[perm_b]
            cell_center_b, cell_half_b = _loose_cell_aabbs(
                pos_b_permuted, pos_b_permuted, p_cell_hash_b
            )
        cutoff_sq = cutoff**2

        def traverse(
            pos_ai: jax.Array,
            stencil: jax.Array,
            cell_starts: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil)
            if prune:
                stencil = _prune_stencil(
                    pos_ai,
                    stencil,
                    cell_starts,
                    cell_center_b,
                    cell_half_b,
                    n_b,
                    cutoff,
                    system,
                )

            def candidate_valid(k: jax.Array) -> jax.Array:
                dr = system.domain.displacement(pos_ai, pos_b[perm_b[k]], system)
                return norm2(dr) <= cutoff_sq

            return _traverse_neighbor_row(
                stencil, cell_starts, p_cell_hash_b, n_b, max_neighbors, candidate_valid
            )

        if max_neighbors == 0:
            topk, row_overflows = jax.vmap(traverse)(
                pos_a, p_neighbor_hashes_a, p_neighbor_starts_a
            )
        else:
            topk, row_overflows = jax.lax.map(
                lambda args: traverse(*args),
                (pos_a, p_neighbor_hashes_a, p_neighbor_starts_a),
                batch_size=min(n_a, query_batch_size),
            )

        topk = jnp.where(topk != -1, perm_b[jnp.maximum(topk, 0)], -1)
        return topk, jnp.any(row_overflows) | hash_overflow_a | hash_overflow_b


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicCellList(_CellListBase):
    r"""Cell-list collider with dynamic or fixed-capacity static traversal.

    A regular grid assigns each particle center to one cell. Sorted cell
    hashes place each cell's members in one contiguous run. A fixed stencil
    selects the cells that can contain contact candidates.

    With ``static=False`` and ``tile_capacity=None``, occupancy sets the dynamic
    traversal bound.
    Explicit environment batches share the longest active bound within each
    bounded query batch. The dynamic path has no per-cell occupancy limit.

    Set ``static=True`` for an automatic fixed tile budget. With a reference
    state, ``System.create`` measures the actual cell work and adds headroom.
    Standalone construction uses a density and packing estimate. Later
    geometry changes or particle clustering can still cause overflow.
    Set ``tile_capacity`` to a nonnegative Python integer for an explicit budget.
    This global budget covers four-candidate tiles across all source particles
    and stencil cells in one environment. Dense cells can use spare capacity
    from other cells. The static path has no per-cell occupancy limit.
    Every environment in a batch uses the same static budget.

    Set ``clump_aware=True`` to exclude same-clump members before static tile
    allocation. All paths exclude same-clump interactions during evaluation.
    External candidates can still exhaust the global tile budget.

    An overflow invalidates the result. After force or energy evaluation,
    call ``system.check_overflow()``. For neighbor queries, check the returned
    overflow flag. The static path reports both tile and hash overflow.
    The dynamic path reports hash overflow.

    For a fixed stencil, dynamic storage scales as :math:`O(N)`.
    Static storage scales as :math:`O(N + C)`, where :math:`C` is
    ``tile_capacity``. Sorting costs :math:`O(N \log N)` in both paths.
    Traversal work depends on cell occupancy and the selected tile budget.

    The dynamic path supports forward differentiation. Its occupancy-dependent
    loops do not support reverse differentiation. Set ``tile_capacity`` for
    reverse differentiation, or use a ``NeighborList`` cache.
    """

    _prune_cells: ClassVar[bool] = False

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        search_range: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
        tile_capacity: int | None = None,
        clump_aware: bool = False,
        static: bool = False,
        *,
        _auto_capacity: bool = False,
    ) -> Self:
        """Create a DynamicCellList instance from the reference state."""
        validate_cell_options(tile_capacity, clump_aware, static, _auto_capacity)
        validate_cell_capacity_estimate(state, cell_size, tile_capacity, static)
        max_rad = jnp.max(state._rad, initial=0.0)
        if cell_size is None:
            min_rad = jnp.min(state._rad, initial=jnp.inf)
            alpha = max_rad / jnp.where(min_rad > 0, min_rad, 1.0)
            cell_size = jnp.where(
                max_rad > 0, jnp.where(alpha < 2.5, 2.0 * max_rad, 0.5 * max_rad), 1.0
            )
        return cls._build(
            state,
            jnp.asarray(cell_size, dtype=float),
            max_rad,
            search_range,
            box_size,
            tile_capacity,
            clump_aware,
            static,
            _auto_capacity,
        )


__all__ = ["DynamicCellList"]
