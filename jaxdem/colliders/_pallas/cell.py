# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Pallas Triton sorted-grid kernels: CellList/MultiCellList forces & pair build."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp

from .config import (
    has_manual_axes,
    is_pallas_enabled,
    pl,
    platform_dispatch,
    shape_dtype,
    triton_compiler_params,
)
from .domains import (
    DomainSpec,
    extract_domain_spec,
    load_domain_scalars,
    triton_displacement,
)
from .forces import ForceSpec, eval_law_node, extract_force_spec
from .refs import (
    ArgPack,
    PackLayout,
    RefView,
    batched_pallas,
    store1,
    store_vec,
    triton_bond_ok,
    triton_dot,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...system import System

_CELL_BLOCK = 128
_BUILD_BLOCK = 128


# ---------------------------------------------------------------------------
# Shared sorted-grid stencil traversal (with stride-1 X-slab merging)
# ---------------------------------------------------------------------------


def _le_wrap_a(v: jax.Array, dims_a: jax.Array) -> jax.Array:
    t = jnp.where(v >= dims_a, v - dims_a, v)
    return jnp.where(t < 0, t + dims_a, jnp.where(t >= dims_a, t - dims_a, t))


def prepare_le_cell_shifts(
    dspec: DomainSpec,
    grid_dims: jax.Array,
    eff_cell_size: jax.Array,
    domain: Any,
) -> jax.Array:
    """Compute compact Lees-Edwards cell-shift metadata ``(lo_pos, lo_neg, hi_pos, hi_neg)``."""
    a, b_ax = dspec.alpha, dspec.beta
    dims_a = grid_dims[a].astype(int)
    shift_mag = domain.gamma * domain.box_size[b_ax] / eff_cell_size[a]
    s_pos, s_neg = -shift_mag, shift_mag
    lo_pos, hi_pos = jnp.floor(s_pos).astype(int), jnp.ceil(s_pos).astype(int)
    lo_neg, hi_neg = jnp.floor(s_neg).astype(int), jnp.ceil(s_neg).astype(int)
    return jnp.stack(
        (
            jnp.mod(lo_pos, dims_a),
            jnp.mod(lo_neg, dims_a),
            (hi_pos != lo_pos).astype(int),
            (hi_neg != lo_neg).astype(int),
        )
    )


def _visit_stencil_cells(
    dspec: DomainSpec,
    r: int,
    coords_i: tuple[jax.Array, ...],
    dims_i: tuple[jax.Array, ...],
    strides_i: tuple[jax.Array, ...],
    le_vals: tuple[jax.Array, ...] | None,
    rv: RefView,
    m_i: jax.Array,
    init_carry: Any,
    visit_cell_fn: Callable[[jax.Array, jax.Array, jax.Array, Any], Any],
    *,
    merge_x_slab: bool = False,
) -> Any:
    """Iterate over every distinct stencil cell (or merged stride-1 X-slab)."""
    dim = dspec.dim
    width = 2 * r + 1
    periodic = dspec.kind != "nonperiodic"
    has_shear = dspec.kind == "lees_edwards"

    def wrap_coord(raw: jax.Array, lim: jax.Array) -> tuple[jax.Array, jax.Array]:
        if periodic:
            return jnp.where(raw < 0, raw + lim, jnp.where(raw >= lim, raw - lim, raw)), m_i
        ok = m_i & (raw >= 0) & (raw < lim)
        return jnp.where(ok, raw, 0), ok

    def visit_hash(h: jax.Array, ok: jax.Array, carry: Any) -> Any:
        c_start = rv.load1("cell_off", h, ok, other=0)
        c_end = rv.load1("cell_off", h + 1, ok, other=0)
        return visit_cell_fn(c_start, c_end, ok, carry)

    def visit_x_slab(
        base_h: jax.Array,
        cx_center: jax.Array,
        span_hi: jax.Array,
        dx_lim: jax.Array,
        ok: jax.Array,
        carry: Any,
    ) -> Any:
        if not periodic:
            lo = jnp.maximum(cx_center - r, 0)
            hi = jnp.minimum(cx_center + span_hi + 1, dx_lim)
            c_st = rv.load1("cell_off", base_h + lo, ok, other=0)
            c_en = rv.load1("cell_off", base_h + hi, ok, other=0)
            return visit_cell_fn(c_st, c_en, ok, carry)

        raw_lo = cx_center - r
        raw_hi = cx_center + span_hi + 1
        wrap_lo, wrap_hi = raw_lo < 0, raw_hi > dx_lim
        lo1 = jnp.where(wrap_lo, 0, raw_lo)
        hi1 = jnp.where(wrap_hi, dx_lim, raw_hi)
        carry = visit_cell_fn(
            rv.load1("cell_off", base_h + lo1, ok, other=0),
            rv.load1("cell_off", base_h + hi1, ok, other=0),
            ok,
            carry,
        )
        has_wrap = ok & (wrap_lo | wrap_hi)
        lo2 = jnp.where(wrap_lo, raw_lo + dx_lim, 0)
        hi2 = jnp.where(wrap_lo, dx_lim, raw_hi - dx_lim)
        return visit_cell_fn(
            rv.load1("cell_off", base_h + lo2, has_wrap, other=0),
            rv.load1("cell_off", base_h + hi2, has_wrap, other=0),
            has_wrap,
            carry,
        )

    def _for_axis(
        axis: int,
        base_h: jax.Array,
        ok_in: jax.Array,
        acc: Any,
        body_fn: Callable[[jax.Array, jax.Array, Any], Any],
    ) -> Any:
        c_ax, d_ax, s_ax = coords_i[axis], dims_i[axis], strides_i[axis]

        def step(off: jax.Array, carry: Any) -> Any:
            w, ok_w = wrap_coord(c_ax + (off - r), d_ax)
            return body_fn(base_h + w * s_ax, ok_in & ok_w, carry)

        return jax.lax.fori_loop(0, width, step, acc)

    if not has_shear:
        zero_h = jnp.zeros_like(coords_i[0])
        if merge_x_slab:
            cx, dx_lim = coords_i[0], dims_i[0]
            r_arr = jnp.asarray(r, dtype=cx.dtype)

            def leaf(h: jax.Array, ok: jax.Array, c: Any) -> Any:
                return visit_x_slab(h, cx, r_arr, dx_lim, ok, c)
        else:

            def leaf(h: jax.Array, ok: jax.Array, c: Any) -> Any:
                return _for_axis(0, h, ok, c, visit_hash)

        def y_step(h: jax.Array, ok: jax.Array, c: Any) -> Any:
            return _for_axis(1, h, ok, c, leaf)

        if dim == 2:
            return y_step(zero_h, m_i, init_carry)
        return _for_axis(2, zero_h, m_i, init_carry, y_step)

    # Lees-Edwards compact duplicate-free stencil.
    assert le_vals is not None
    lo_pos_m, lo_neg_m, need_hi_pos, need_hi_neg = le_vals
    a, b_ax = dspec.alpha, dspec.beta
    ca, cb = coords_i[a], coords_i[b_ax]
    da_lim, db_lim = dims_i[a], dims_i[b_ax]
    sa_st, sb_st = strides_i[a], strides_i[b_ax]

    def _le_beta_params(ob: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        raw_b = cb + (ob - r)
        img_b = jnp.where(raw_b < 0, -1, jnp.where(raw_b >= db_lim, 1, 0))
        wb = raw_b - img_b * db_lim
        lo_m = jnp.where(img_b == 0, 0, jnp.where(img_b > 0, lo_pos_m, lo_neg_m))
        return wb, lo_m, img_b

    if merge_x_slab and a == 0:

        def le_slab_b(base_h: jax.Array, ok_o: jax.Array, acc_o: Any) -> Any:
            def step_b(ob: jax.Array, acc_b: Any) -> Any:
                wb, lo_m, img_b = _le_beta_params(ob)
                extra_hi = jnp.where(
                    img_b == 0,
                    0,
                    jnp.where(img_b > 0, need_hi_pos != 0, need_hi_neg != 0).astype(ca.dtype),
                )
                ca_s = _le_wrap_a(ca + lo_m, da_lim)
                return visit_x_slab(base_h + wb * sb_st, ca_s, r + extra_hi, da_lim, ok_o, acc_b)

            return jax.lax.fori_loop(0, width, step_b, acc_o)

        zero_h = jnp.zeros_like(ca)
        if dim == 2:
            return le_slab_b(zero_h, m_i, init_carry)
        return _for_axis(3 - a - b_ax, zero_h, m_i, init_carry, le_slab_b)

    def _visit_ab(h_ab: jax.Array, ok_ab: jax.Array, acc: Any) -> Any:
        if dim == 2:
            return visit_hash(h_ab, ok_ab, acc)
        return _for_axis(3 - a - b_ax, h_ab, ok_ab, acc, visit_hash)

    def le_base_b(ob: jax.Array, acc_b: Any) -> Any:
        wb, lo_m, _ = _le_beta_params(ob)
        h_b = wb * sb_st

        def le_base_a(oa: jax.Array, acc_a: Any) -> Any:
            wa = _le_wrap_a(ca + (oa - r) + lo_m, da_lim)
            return _visit_ab(wa * sa_st + h_b, m_i, acc_a)

        return jax.lax.fori_loop(0, width, le_base_a, acc_b)

    carry = jax.lax.fori_loop(0, width, le_base_b, init_carry)

    def le_ext(ok_idx: jax.Array, acc_e: Any) -> Any:
        off_k = ok_idx + 1
        v_neg = (cb - off_k < 0) & (need_hi_neg != 0)
        v_pos = (cb + off_k >= db_lim) & (need_hi_pos != 0)
        v_bk = m_i & (v_neg | v_pos)
        wb_bk = jnp.where(v_bk, jnp.where(v_neg, cb - off_k + db_lim, cb + off_k - db_lim), 0)
        lo_bk = jnp.where(v_neg, lo_neg_m, lo_pos_m)
        wa_bk = _le_wrap_a(ca + (r + 1) + lo_bk, da_lim)
        return _visit_ab(wa_bk * sa_st + wb_bk * sb_st, v_bk, acc_e)

    return jax.lax.fori_loop(0, r, le_ext, carry)


def _prune_cell_aabb(
    rv: RefView,
    dspec: DomainSpec,
    dom: dict[str, Any],
    ps: tuple[jax.Array, ...],
    reach_i: jax.Array,
    c_start: jax.Array,
    c_end: jax.Array,
    ok_cell: jax.Array,
    c_len: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Prune a candidate cell whose loose AABB cannot overlap ``reach_i``."""
    dim = dspec.dim
    non_empty = ok_cell & (c_len > 0)
    safe_st = jnp.where(non_empty, c_start, 0)
    cc = rv.load_vec("cell_c", safe_st, dim, non_empty)
    ch = rv.load_vec("cell_h", safe_st, dim, non_empty)
    dr_c, _, _ = triton_displacement(dspec, ps, cc, dom)
    ov = non_empty
    for d in range(dim):
        ov = ov & (jnp.abs(dr_c[d]) <= reach_i + ch[d])
    return jnp.where(ov, c_len, 0), jnp.where(ov, c_end, c_start)


# ---------------------------------------------------------------------------
# Kernel 2: Sorted-grid CellList / MultiCellList forces
# ---------------------------------------------------------------------------


def _run_pallas_cell_forces(
    fspec: ForceSpec,
    dspec: DomainSpec,
    layout: PackLayout,
    r: int,
    bond_w: int,
    do_prune: bool,
    count_tiles: bool,
    args: tuple[Any, ...],
    batched: tuple[bool, ...],
    batch_size: int,
) -> tuple[jax.Array, ...]:
    from .._partition import PAIR_UNROLL

    dim, ang_dim = fspec.dim, fspec.ang_dim
    has_batch = batch_size > 0
    pos_s_flat = args[layout.index["pos"]]
    N = pos_s_flat.shape[-1] // dim
    dtype = pos_s_flat.dtype
    int_dtype = args[layout.index["cell_off"]].dtype

    f_shape = (batch_size, N * dim) if has_batch else (N * dim,)
    t_shape = (batch_size, N * ang_dim) if has_batch else (N * ang_dim,)
    tile_shape = (batch_size, N) if has_batch else (N,)
    out_shapes: list[Any] = [shape_dtype(f_shape, dtype)]
    if fspec.has_torque:
        out_shapes.append(shape_dtype(t_shape, dtype))
    if count_tiles:
        out_shapes.append(shape_dtype(tile_shape, int_dtype))

    BLOCK = _CELL_BLOCK

    def kernel(*refs: Any) -> None:
        n_in = len(args)
        out_refs = refs[n_in:]
        f_out_ref = out_refs[0]
        t_out_ref = out_refs[1] if fspec.has_torque else None
        tile_out_ref = out_refs[-1] if count_tiles else None

        pid = pl.program_id(0)
        b = pl.program_id(1) if has_batch else 0
        rv = RefView(refs[:n_in], batched, layout, b, has_batch)

        s = pid * BLOCK + jnp.arange(BLOCK)
        m_s = s < N
        isame = rv.scalar("isame") != 0

        ps = rv.load_vec("pos", s, dim, m_s)
        rs = rv.load1("rad", s, m_s, other=0.0)
        cs_coords = rv.load_vec("coords", s, dim, m_s, other=0)
        clump_i = rv.load1("clump", s, m_s, other=-1)
        orig_i = rv.load1("orig", s, m_s, other=-1) if bond_w > 0 else jnp.zeros_like(s)
        dims_i = tuple(rv.scalar("dims", d) for d in range(dim))
        strides_i = tuple(rv.scalar("strides", d) for d in range(dim))
        le_vals = tuple(rv.scalar("le", d) for d in range(4)) if rv.has("le") else None
        sr_i = rv.load1("sr", s, m_s, other=0.0) if do_prune else jnp.zeros_like(rs)

        mi = rv.load1("mat", s, m_s, other=0) if rv.has("mat") else jnp.zeros_like(s)
        si = rv.load1("sp", s, m_s, other=0) if rv.has("sp") else jnp.zeros_like(s)
        vel_i = rv.load_vec("vel", s, dim, m_s) if rv.has("vel") else ()
        ang_i = rv.load_vec("ang", s, ang_dim, m_s) if rv.has("ang") else ()
        ppr_i = rv.load_vec("ppr", s, dim, m_s) if rv.has("ppr") else ()
        mass_i = rv.load1("mass", s, m_s, other=1.0) if rv.has("mass") else jnp.ones_like(rs)
        dom = load_domain_scalars(rv, dspec, need_vel_rates=fspec.needs_vel)

        z_f = tuple(jnp.zeros((BLOCK,), dtype=dtype) for _ in range(dim))
        z_t = tuple(jnp.zeros((BLOCK,), dtype=dtype) for _ in range(ang_dim))
        z_tile = (jnp.zeros((BLOCK,), dtype=int_dtype),)
        init_carry = (
            z_f
            + (z_t if fspec.has_torque else ())
            + (z_tile if count_tiles else ())
        )

        def visit_cell(
            c_start: jax.Array,
            c_end: jax.Array,
            ok_cell: jax.Array,
            carry: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, ...]:
            c_len = jnp.where(ok_cell, c_end - c_start, 0)
            if do_prune:
                c_len, c_end = _prune_cell_aabb(
                    rv, dspec, dom, ps, sr_i, c_start, c_end, ok_cell, c_len
                )
            if count_tiles:
                tile_inc = (c_len + (PAIR_UNROLL - 1)) // PAIR_UNROLL
                carry = carry[:-1] + (carry[-1] + tile_inc,)

            max_c = jnp.max(c_len)

            def inner(t: jax.Array, acc: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
                k = c_start + t
                in_cell = ok_cell & (k < c_end)
                ck = rv.load1("clump", k, in_cell, other=-2)
                valid = in_cell & (clump_i != ck)
                if bond_w > 0:
                    valid = valid & triton_bond_ok(bond_w, rv, k, orig_i, valid, isame)
                pk = rv.load_vec("pos", k, dim, valid)
                rk = rv.load1("rad", k, valid, other=0.0)
                dr, img, shear_img = triton_displacement(dspec, ps, pk, dom)
                ctx = {
                    "j": k,
                    "ri": rs,
                    "rj": rk,
                    "R": rs + rk,
                    "dr": dr,
                    "n2": triton_dot(dr, dr),
                    "img": img,
                    "shear_img": shear_img,
                    "valid": valid,
                    "mi": mi,
                    "mj": rv.load1("mat", k, valid, other=0) if rv.has("mat") else jnp.zeros_like(k),
                    "si": si,
                    "sj": rv.load1("sp", k, valid, other=0) if rv.has("sp") else jnp.zeros_like(k),
                    "vel_i": vel_i,
                    "ang_i": ang_i,
                    "ppr_i": ppr_i,
                    "mass_i": mass_i,
                    "dt": jnp.asarray(0.0, dtype=dtype),
                    "slot": None,
                    "hist_out_ref": None,
                }
                df, dt_vec = eval_law_node(
                    fspec.root, rv, fspec, dspec, dom, ctx, valid & (s != k), jnp.zeros_like(valid)
                )
                new_f = tuple(acc[d] + df[d] for d in range(dim))
                if fspec.has_torque:
                    new_t = tuple(acc[dim + d] + dt_vec[d] for d in range(ang_dim))
                    return new_f + new_t + (acc[dim + ang_dim :] if count_tiles else ())
                return new_f + (acc[dim:] if count_tiles else ())

            return jax.lax.fori_loop(0, max_c, inner, carry)

        res = _visit_stencil_cells(
            dspec,
            r,
            cs_coords,
            dims_i,
            strides_i,
            le_vals,
            rv,
            m_s,
            init_carry,
            visit_cell,
            merge_x_slab=(not do_prune) and (not count_tiles),
        )
        # Scatter directly to original particle indices (perm[s]) to avoid an
        # extra full-array unpermute pass in VRAM.
        out_idx = rv.load1("orig", s, m_s, other=0)
        store_vec(f_out_ref, b, has_batch, out_idx, dim, res[:dim], m_s)
        if fspec.has_torque and t_out_ref is not None:
            store_vec(t_out_ref, b, has_batch, out_idx, ang_dim, res[dim : dim + ang_dim], m_s)
        if count_tiles and tile_out_ref is not None:
            store1(tile_out_ref, b, has_batch, s, res[-1], m_s)

    grid = (pl.cdiv(N, BLOCK), batch_size) if has_batch else (pl.cdiv(N, BLOCK),)
    out_spec = tuple(out_shapes) if len(out_shapes) > 1 else out_shapes[0]
    raw = pl.pallas_call(
        kernel,
        out_shape=out_spec,
        grid=grid,
        compiler_params=triton_compiler_params(num_warps=4, num_stages=2),
    )(*args)
    return raw if isinstance(raw, tuple) else (raw,)


def try_pallas_cell_forces(
    state_s: Any,
    system: System,
    pos_s: jax.Array,
    coords_s: jax.Array,
    cell_offsets: jax.Array,
    dims_i: jax.Array,
    strides_i: jax.Array,
    eff_cell_size: jax.Array,
    perm: jax.Array,
    hash_overflow: jax.Array,
    r: int,
    *,
    do_prune: bool,
    cell_center: jax.Array | None,
    cell_half: jax.Array | None,
    search_rad_s: jax.Array | None,
    tile_capacity: int | None,
    want_torque: bool,
    quasistatic: bool,
    pure_fallback: Callable[[], tuple[Any, jax.Array]],
) -> tuple[Any, jax.Array] | None:
    """Run sorted-grid CellList forces via Pallas Triton when supported, else ``None``."""
    if not is_pallas_enabled() or r < 1 or has_manual_axes(state_s, system, pos_s, cell_offsets):
        return None
    f_res = extract_force_spec(
        system.force_model,
        state_s,
        system,
        advance_history=False,
        quasistatic=quasistatic,
    )
    if f_res is None:
        return None
    fspec, f_params = f_res
    d_res = extract_domain_spec(system.domain, state_s.dim, need_vel_rates=fspec.needs_vel)
    if d_res is None or fspec.has_torque != want_torque:
        return None
    dspec, d_params = d_res

    dim, ang_dim = fspec.dim, fspec.ang_dim
    n = pos_s.shape[0]
    bond_w = int(state_s.bond_id.shape[-1])
    count_tiles = tile_capacity is not None

    pack = ArgPack()
    pack.add("pos", pos_s.reshape(-1))
    pack.add("rad", state_s.rad)
    pack.add("coords", coords_s.reshape(-1))
    pack.add("cell_off", cell_offsets)
    pack.add("dims", dims_i)
    pack.add("strides", strides_i)
    pack.add("clump", state_s.clump_id)
    pack.add("bond", state_s.bond_id.reshape(-1) if bond_w > 0 else jnp.zeros((1,), dtype=int))
    pack.add("orig", state_s.orig_id)
    pack.add("isame", jnp.reshape(jnp.asarray(system.interact_same_bond_id, dtype=int), (1,)))
    if dspec.kind == "lees_edwards":
        pack.add("le", prepare_le_cell_shifts(dspec, dims_i, eff_cell_size, system.domain))
    if do_prune:
        assert cell_center is not None and cell_half is not None and search_rad_s is not None
        pack.add("cell_c", cell_center.reshape(-1))
        pack.add("cell_h", cell_half.reshape(-1))
        pack.add("sr", search_rad_s)
    if fspec.needs_mat_id:
        pack.add("mat", state_s.mat_id)
    if fspec.needs_species_id:
        pack.add("sp", state_s.species_id)
    if fspec.needs_vel:
        pack.add("vel", state_s.vel.reshape(-1))
        pack.add("ang", state_s.ang_vel.reshape(-1))
        if not fspec.only_spheres:
            pack.add("ppr", state_s._pos_p_rot.reshape(-1))
    if fspec.needs_mass:
        pack.add("mass", state_s.mass)
    pack.add_group("dom", d_params)
    pack.add_group("params", f_params)
    flat_args, layout = pack.build()

    gpu_eval = batched_pallas(
        lambda kargs, batched, bsz: _run_pallas_cell_forces(
            fspec, dspec, layout, r, bond_w, do_prune, count_tiles, kargs, batched, bsz
        )
    )

    def gpu_wrapped(_pm: Any, hov: Any, *kargs: Any) -> tuple[Any, jax.Array]:
        out = gpu_eval(*kargs)
        f_out = out[0].reshape(n, dim)
        acc_out: Any = (f_out, out[1].reshape(n, ang_dim)) if want_torque else f_out
        if count_tiles:
            assert tile_capacity is not None
            ov = (jnp.sum(out[-1], dtype=cell_offsets.dtype) > tile_capacity) | hov
        else:
            ov = hov
        return acc_out, ov

    def cpu_wrapped(_pm: Any, _hov: Any, *_kargs: Any) -> tuple[Any, jax.Array]:
        return pure_fallback()

    return platform_dispatch(gpu_wrapped, cpu_wrapped, perm, hash_overflow, *flat_args)


# ---------------------------------------------------------------------------
# Kernel 3: Unified sorted-grid CSR pair builder (Count + Fill)
# ---------------------------------------------------------------------------


def _run_pallas_build(
    mode: str,  # "count" | "fill"
    dspec: DomainSpec,
    layout: PackLayout,
    r: int,
    bond_w: int,
    has_radii: bool,
    do_prune: bool,
    count_tiles: bool,
    args: tuple[Any, ...],
    batched: tuple[bool, ...],
    batch_size: int,
) -> Any:
    """Unified Triton sorted-grid neighbor pair counter and CSR filler."""
    from .._partition import PAIR_UNROLL

    dim = dspec.dim
    has_batch = batch_size > 0
    pos_s_flat = args[layout.index["pos"]]
    N = pos_s_flat.shape[-1] // dim
    int_dtype = args[layout.index["cell_off"]].dtype
    is_fill = mode == "fill"

    k_args = list(args)
    k_batched = list(batched)
    in_out_aliases: dict[int, int] = {}
    if is_fill:
        idx_nbr = layout.index["nbr_init"]
        cap = k_args[idx_nbr].shape[-1]
        out_shape = (batch_size, cap) if has_batch else (cap,)
        if has_batch and not k_batched[idx_nbr]:
            k_args[idx_nbr] = jnp.broadcast_to(k_args[idx_nbr][None, :], out_shape)
            k_batched[idx_nbr] = True
        out_shapes: list[Any] = [shape_dtype(out_shape, k_args[idx_nbr].dtype)]
        in_out_aliases[idx_nbr] = 0
    else:
        c_shape = (batch_size, N) if has_batch else (N,)
        out_shapes = [shape_dtype(c_shape, int_dtype)]
        if count_tiles:
            out_shapes.append(shape_dtype(c_shape, int_dtype))

    BLOCK = _BUILD_BLOCK

    def kernel(*refs: Any) -> None:
        n_in = len(k_args)
        out_refs = refs[n_in:]
        pid = pl.program_id(0)
        b = pl.program_id(1) if has_batch else 0
        rv = RefView(refs[:n_in], tuple(k_batched), layout, b, has_batch)

        s = pid * BLOCK + jnp.arange(BLOCK)
        m_s = s < N
        cutoff_sq = rv.scalar("cutoff_sq")
        isame = rv.scalar("isame") != 0

        ps = rv.load_vec("pos", s, dim, m_s)
        cs_coords = rv.load_vec("coords", s, dim, m_s, other=0)
        clump_i = rv.load1("clump", s, m_s, other=-1)
        perm_i = rv.load1("perm", s, m_s, other=-1) if bond_w > 0 else jnp.zeros_like(s)
        dims_i = tuple(rv.scalar("dims", d) for d in range(dim))
        strides_i = tuple(rv.scalar("strides", d) for d in range(dim))
        le_vals = tuple(rv.scalar("le", d) for d in range(4)) if rv.has("le") else None
        sr_i = rv.load1("sr", s, m_s, other=0.0) if has_radii else jnp.zeros_like(ps[0])
        skin_val = rv.scalar("skin") if has_radii else jnp.asarray(0.0, dtype=ps[0].dtype)
        cut_val = rv.scalar("cut") if do_prune else jnp.asarray(0.0, dtype=ps[0].dtype)
        dom = load_domain_scalars(rv, dspec, need_vel_rates=False)

        if is_fill:
            base_s = rv.load1("base_s", s, m_s, other=0)
            end_s = rv.load1("end_s", s, m_s, other=0)
            init_carry: Any = base_s
            nbr_out = out_refs[0]
        else:
            z_c = jnp.zeros((BLOCK,), dtype=int_dtype)
            init_carry = (z_c, z_c) if count_tiles else (z_c,)

        def visit_cell(
            c_start: jax.Array, c_end: jax.Array, ok_cell: jax.Array, carry: Any
        ) -> Any:
            c_len = jnp.where(ok_cell, c_end - c_start, 0)
            if do_prune:
                c_len, c_end = _prune_cell_aabb(
                    rv, dspec, dom, ps, cut_val, c_start, c_end, ok_cell, c_len
                )
            if (not is_fill) and count_tiles:
                tile_inc = (c_len + (PAIR_UNROLL - 1)) // PAIR_UNROLL
                carry = (carry[0], carry[1] + tile_inc)

            max_c = jnp.max(c_len)

            def inner(t: jax.Array, acc: Any) -> Any:
                k = c_start + t
                in_cell = ok_cell & (k < c_end)
                ck = rv.load1("clump", k, in_cell, other=-2)
                valid = in_cell & (clump_i != ck)
                if bond_w > 0:
                    valid = valid & triton_bond_ok(bond_w, rv, k, perm_i, valid, isame)
                pk = rv.load_vec("pos", k, dim, valid)
                dr, _, _ = triton_displacement(dspec, ps, pk, dom)
                d2 = triton_dot(dr, dr)
                ok = valid & (d2 <= cutoff_sq)
                if has_radii:
                    sr_k = rv.load1("sr", k, ok, other=0.0)
                    reach = sr_i + sr_k + skin_val
                    ok = ok & (d2 <= reach * reach)
                if is_fill:
                    do_write = ok & (acc < end_s)
                    dst = rv.load1("perm", k, do_write, other=-1)
                    store1(nbr_out, b, has_batch, acc, dst, do_write)
                    return acc + do_write.astype(acc.dtype)
                inc = ok.astype(int_dtype)
                return (acc[0] + inc, acc[1]) if count_tiles else (acc[0] + inc,)

            return jax.lax.fori_loop(0, max_c, inner, carry)

        res = _visit_stencil_cells(
            dspec,
            r,
            cs_coords,
            dims_i,
            strides_i,
            le_vals,
            rv,
            m_s,
            init_carry,
            visit_cell,
            merge_x_slab=(not do_prune) and (is_fill or not count_tiles),
        )
        if not is_fill:
            store1(out_refs[0], b, has_batch, s, res[0], m_s)
            if count_tiles:
                store1(out_refs[1], b, has_batch, s, res[1], m_s)

    grid = (pl.cdiv(N, BLOCK), batch_size) if has_batch else (pl.cdiv(N, BLOCK),)
    out_spec = tuple(out_shapes) if len(out_shapes) > 1 else out_shapes[0]
    raw = pl.pallas_call(
        kernel,
        out_shape=out_spec,
        grid=grid,
        input_output_aliases=in_out_aliases,
        compiler_params=triton_compiler_params(num_warps=4, num_stages=2),
    )(*k_args)
    if is_fill:
        return raw
    return raw if isinstance(raw, tuple) else (raw,)


def try_pallas_build_pairs(
    pos_s: jax.Array,
    coords_s: jax.Array,
    cell_offsets: jax.Array,
    dims_i: jax.Array,
    strides_i: jax.Array,
    eff_cell_size: jax.Array,
    perm: jax.Array,
    clumps_s: jax.Array,
    bonds_s: jax.Array,
    search_radii_s: jax.Array | None,
    domain: Any,
    cutoff: jax.Array,
    skin: jax.Array,
    interact_same: Any,
    hash_overflow: jax.Array,
    r: int,
    capacity: int | None,
    *,
    do_prune: bool,
    cell_center: jax.Array | None,
    cell_half: jax.Array | None,
    tile_capacity: int | None,
    pure_fallback: Callable[[], tuple[Any, ...]],
) -> tuple[Any, ...] | None:
    """Build CSR neighbor pairs via Pallas Triton Count+Fill when supported, else ``None``."""
    if not is_pallas_enabled() or r < 1 or has_manual_axes(pos_s, cell_offsets, perm, domain):
        return None
    dim = int(pos_s.shape[-1])
    d_res = extract_domain_spec(domain, dim, need_vel_rates=False)
    if d_res is None:
        return None
    dspec, d_params = d_res

    from .._partition import _capacity_offsets

    bond_w = int(bonds_s.shape[-1])
    has_radii = search_radii_s is not None
    count_tiles = tile_capacity is not None
    dtype = pos_s.dtype

    def _pack_common(pack: ArgPack) -> None:
        pack.add("pos", pos_s.reshape(-1))
        pack.add("coords", coords_s.reshape(-1))
        pack.add("cell_off", cell_offsets)
        pack.add("dims", dims_i)
        pack.add("strides", strides_i)
        pack.add("clump", clumps_s)
        pack.add("bond", bonds_s.reshape(-1) if bond_w > 0 else jnp.zeros((1,), dtype=int))
        pack.add("perm", perm)
        pack.add("cutoff_sq", jnp.reshape(jnp.asarray(cutoff * cutoff, dtype=dtype), (1,)))
        pack.add("isame", jnp.reshape(jnp.asarray(interact_same, dtype=int), (1,)))
        if dspec.kind == "lees_edwards":
            pack.add("le", prepare_le_cell_shifts(dspec, dims_i, eff_cell_size, domain))
        if has_radii:
            assert search_radii_s is not None
            pack.add("sr", search_radii_s)
            pack.add("skin", jnp.reshape(jnp.asarray(skin, dtype=dtype), (1,)))
        if do_prune:
            assert cell_center is not None and cell_half is not None
            pack.add("cell_c", cell_center.reshape(-1))
            pack.add("cell_h", cell_half.reshape(-1))
            pack.add("cut", jnp.reshape(jnp.asarray(cutoff, dtype=dtype), (1,)))
        pack.add_group("dom", d_params)

    c_pack = ArgPack()
    _pack_common(c_pack)
    count_args, count_layout = c_pack.build()

    # Fill layout extends count_layout with (base_s, end_s, nbr_init) appended at the end.
    n_common = len(count_args)
    fill_index = dict(count_layout.index)
    fill_index["base_s"] = n_common
    fill_index["end_s"] = n_common + 1
    fill_index["nbr_init"] = n_common + 2
    fill_layout = PackLayout(fill_index, dict(count_layout.slices))

    gpu_count = batched_pallas(
        lambda kargs, batched, bsz: _run_pallas_build(
            "count", dspec, count_layout, r, bond_w, has_radii, do_prune, count_tiles, kargs, batched, bsz
        )
    )
    gpu_fill = batched_pallas(
        lambda kargs, batched, bsz: _run_pallas_build(
            "fill", dspec, fill_layout, r, bond_w, has_radii, do_prune, False, kargs, batched, bsz
        ),
        single_output=True,
    )

    perm_idx = count_layout.index["perm"]
    coff_idx = count_layout.index["cell_off"]

    def gpu_wrapped(hov: Any, *all_in: Any) -> tuple[Any, ...]:
        pm = all_in[perm_idx]
        cnt_out = gpu_count(*all_in)
        cnt_s = cnt_out[0]
        counts = jnp.empty_like(cnt_s).at[pm].set(cnt_s, unique_indices=True)
        if count_tiles:
            assert tile_capacity is not None
            tile_ov = (jnp.sum(cnt_out[1], dtype=all_in[coff_idx].dtype) > tile_capacity) | hov
        else:
            tile_ov = hov
        if capacity is None:
            return counts, tile_ov
        offsets, pair_ov = _capacity_offsets(counts, capacity)
        overflow = tile_ov | pair_ov
        neighbors = jnp.full_like(counts, -1, shape=(capacity,), dtype=int)
        if capacity == 0:
            return neighbors, offsets, overflow
        neighbors = gpu_fill(*all_in, offsets[pm], offsets[pm + 1], neighbors)
        return neighbors, offsets, overflow

    def cpu_wrapped(_hov: Any, *_all_in: Any) -> tuple[Any, ...]:
        return pure_fallback()

    return platform_dispatch(gpu_wrapped, cpu_wrapped, hash_overflow, *count_args)
