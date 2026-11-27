# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Pallas Triton CSR kernels: variable-length force evaluation & history remap."""

from __future__ import annotations

import math
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
    triton_cross,
    triton_dot,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...state import State
    from ...system import System

_REMAP_BLOCK = 128


def _run_pallas_csr_forces(
    fspec: ForceSpec,
    dspec: DomainSpec,
    layout: PackLayout,
    args: tuple[Any, ...],
    batched: tuple[bool, ...],
    batch_size: int,
) -> tuple[jax.Array, ...]:
    """Launch Triton CSR force kernel (1D unbatched or 2D batched)."""
    dim, ang_dim = fspec.dim, fspec.ang_dim
    has_batch = batch_size > 0
    pos_flat = args[layout.index["pos"]]
    N = pos_flat.shape[-1] // dim
    dtype = pos_flat.dtype
    cap = args[layout.index["nbr"]].shape[-1]
    write_hist = fspec.has_history and fspec.advance_history
    emit_torque = fspec.has_torque or (not fspec.only_spheres)

    f_shape = (batch_size, N * dim) if has_batch else (N * dim,)
    t_shape = (batch_size, N * ang_dim) if has_batch else (N * ang_dim,)
    out_shapes: list[Any] = [shape_dtype(f_shape, dtype)]
    if emit_torque:
        out_shapes.append(shape_dtype(t_shape, dtype))

    in_out_aliases: dict[int, int] = {}
    kernel_args = list(args)
    kernel_batched = list(batched)
    if write_hist:
        idx_hist = layout.index["hist"]
        h_shape = (
            (batch_size, cap * fspec.hist_width)
            if has_batch
            else (cap * fspec.hist_width,)
        )
        if has_batch and not kernel_batched[idx_hist]:
            kernel_args[idx_hist] = jnp.broadcast_to(
                kernel_args[idx_hist][None, :], h_shape
            )
            kernel_batched[idx_hist] = True
        in_out_aliases[idx_hist] = len(out_shapes)
        out_shapes.append(shape_dtype(h_shape, kernel_args[idx_hist].dtype))

    # Warp-sized blocks: each program loops to its rows' largest degree, so
    # small blocks waste fewer masked iterations (3D stateless: 38 -> 30 us).
    BLOCK = 32
    num_warps = 2 if fspec.has_history else 1

    def kernel(*refs: Any) -> None:
        n_in = len(kernel_args)
        out_refs = refs[n_in:]
        f_out_ref = out_refs[0]
        t_out_ref = out_refs[1] if emit_torque else None
        h_out_ref = out_refs[-1] if write_hist else None

        pid = pl.program_id(0)
        b = pl.program_id(1) if has_batch else 0
        rv = RefView(refs[:n_in], tuple(kernel_batched), layout, b, has_batch)

        i = pid * BLOCK + jnp.arange(BLOCK)
        m_i = i < N

        pi = rv.load_vec("pos", i, dim, m_i)
        ri = rv.load1("rad", i, m_i, other=0.0)
        start = rv.load1("off", i, m_i, other=0)
        end = rv.load1("off", i + 1, m_i, other=0)
        max_deg = jnp.max(jnp.where(m_i, end - start, 0))

        mi = rv.load1("mat", i, m_i, other=0) if rv.has("mat") else jnp.zeros_like(i)
        si = rv.load1("sp", i, m_i, other=0) if rv.has("sp") else jnp.zeros_like(i)
        vel_i = rv.load_vec("vel", i, dim, m_i) if rv.has("vel") else ()
        ang_i = rv.load_vec("ang", i, ang_dim, m_i) if rv.has("ang") else ()
        ppr_i = rv.load_vec("ppr", i, dim, m_i) if rv.has("ppr") else ()
        mass_i = (
            rv.load1("mass", i, m_i, other=1.0) if rv.has("mass") else jnp.ones_like(ri)
        )
        dt_val = rv.scalar("dt") if rv.has("dt") else jnp.asarray(0.0, dtype=dtype)
        dom = load_domain_scalars(rv, dspec, need_vel_rates=fspec.needs_vel)

        z_f = tuple(jnp.zeros((BLOCK,), dtype=dtype) for _ in range(dim))
        z_t = tuple(jnp.zeros((BLOCK,), dtype=dtype) for _ in range(ang_dim))
        init_carry = z_f + z_t if fspec.has_torque else z_f

        def body(k: jax.Array, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            slot = start + k
            in_row = m_i & (slot < end)
            nbr_j = rv.load1(
                "nbr", slot, in_row, other=-1, eviction_policy="evict_first"
            )
            has_nbr = in_row & (nbr_j >= 0)
            j = jnp.where(has_nbr, nbr_j, 0)
            # The build already dropped same-clump and bond-excluded pairs, and
            # ID rewiring requires invalidation, so no per-pair re-check here.
            valid = has_nbr
            if rv.has("pr"):
                prj = rv.load_vec("pr", j, 4, valid)
                pj, rj = prj[:dim], prj[3]
            else:
                pj = rv.load_vec("pos", j, dim, valid)
                rj = rv.load1("rad", j, valid, other=0.0)
            dr, img, shear_img = triton_displacement(dspec, pi, pj, dom)
            ctx = {
                "j": j,
                "ri": ri,
                "rj": rj,
                "R": ri + rj,
                "dr": dr,
                "n2": triton_dot(dr, dr),
                "img": img,
                "shear_img": shear_img,
                "valid": valid,
                "mi": mi,
                "mj": (
                    rv.load1("mat", j, valid, other=0)
                    if rv.has("mat")
                    else jnp.zeros_like(j)
                ),
                "si": si,
                "sj": (
                    rv.load1("sp", j, valid, other=0)
                    if rv.has("sp")
                    else jnp.zeros_like(j)
                ),
                "vel_i": vel_i,
                "ang_i": ang_i,
                "ppr_i": ppr_i,
                "mass_i": mass_i,
                "dt": dt_val,
                "slot": slot,
                "hist_out_ref": h_out_ref,
            }
            df, dt_vec = eval_law_node(
                fspec.root, rv, fspec, dspec, dom, ctx, valid & (i != j), in_row
            )
            new_f = tuple(carry[d] + df[d] for d in range(dim))
            if fspec.has_torque:
                new_t = tuple(carry[dim + d] + dt_vec[d] for d in range(ang_dim))
                return new_f + new_t
            return new_f

        res = jax.lax.fori_loop(0, max_deg, body, init_carry)
        f_res = res[:dim]
        store_vec(f_out_ref, b, has_batch, i, dim, f_res, m_i)
        if emit_torque and t_out_ref is not None:
            t_res = res[dim : dim + ang_dim] if fspec.has_torque else z_t
            if not fspec.only_spheres:
                cr = triton_cross(dim, ppr_i, f_res)
                t_res = tuple(t_res[d] + cr[d] for d in range(ang_dim))
            store_vec(t_out_ref, b, has_batch, i, ang_dim, t_res, m_i)

    grid = (pl.cdiv(N, BLOCK), batch_size) if has_batch else (pl.cdiv(N, BLOCK),)
    out_spec = tuple(out_shapes) if len(out_shapes) > 1 else out_shapes[0]
    raw = pl.pallas_call(
        kernel,
        out_shape=out_spec,
        grid=grid,
        input_output_aliases=in_out_aliases,
        compiler_params=triton_compiler_params(num_warps=num_warps, num_stages=2),
    )(*kernel_args)
    return raw if isinstance(raw, tuple) else (raw,)


def can_use_pallas_csr_forces(
    state: State,
    system: System,
    *,
    advance_history: bool,
    quasistatic: bool,
) -> bool:
    if not is_pallas_enabled() or state.N == 0 or has_manual_axes(state, system):
        return False
    f = extract_force_spec(
        system.force_model,
        state,
        system,
        advance_history=advance_history,
        quasistatic=quasistatic,
    )
    if f is None:
        return False
    return (
        extract_domain_spec(system.domain, state.dim, need_vel_rates=f[0].needs_vel)
        is not None
    )


def pallas_csr_forces(
    state: State,
    system: System,
    advance_history: bool,
    quasistatic: bool,
    pure_fallback: Callable[[], tuple[jax.Array, jax.Array, jax.Array]],
    *,
    pos: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate CSR NeighborList forces (+ torque + history) via Pallas Triton on GPU."""
    if not is_pallas_enabled() or state.N == 0 or has_manual_axes(state, system):
        return pure_fallback()
    f_res = extract_force_spec(
        system.force_model,
        state,
        system,
        advance_history=advance_history,
        quasistatic=quasistatic,
    )
    if f_res is None:
        return pure_fallback()
    fspec, f_params = f_res
    d_res = extract_domain_spec(
        system.domain, state.dim, need_vel_rates=fspec.needs_vel
    )
    if d_res is None:
        return pure_fallback()
    dspec, d_params = d_res

    col: Any = system.collider
    dim, ang_dim = fspec.dim, fspec.ang_dim
    if pos is None:
        pos = state.pos
    n = state.N
    write_hist = fspec.has_history and fspec.advance_history

    pack = ArgPack()
    pack.add("pos", pos.reshape(-1))
    pack.add("rad", state.rad)
    if dim == 3:
        # One 16-byte (x, y, z, r) record per neighbor gather: one L2 sector
        # instead of ~2.3. In 2D the extra pack pass costs what it saves.
        lane = jnp.arange(4)
        padded = jnp.pad(pos, ((0, 0), (0, 4 - dim)))
        pack.add("pr", jnp.where(lane == 3, state.rad[:, None], padded).reshape(-1))
    pack.add("off", col.row_offsets)
    pack.add("nbr", col.neighbor_list)
    if fspec.needs_mat_id:
        pack.add("mat", state.mat_id)
    if fspec.needs_species_id:
        pack.add("sp", state.species_id)
    if fspec.needs_vel:
        pack.add("vel", state.vel.reshape(-1))
        pack.add("ang", state.ang_vel.reshape(-1))
    if not fspec.only_spheres:
        pack.add("ppr", state._pos_p_rot.reshape(-1))
    if fspec.needs_mass:
        pack.add("mass", state.mass)
    if fspec.needs_dt:
        pack.add("dt", jnp.reshape(jnp.asarray(system.dt, dtype=pos.dtype), (1,)))
    if fspec.has_history:
        pack.add("hist", col.history.reshape(-1))
    pack.add_group("dom", d_params)
    pack.add_group("params", f_params)
    flat_args, layout = pack.build()

    gpu_eval = batched_pallas(
        lambda kargs, batched, bsz: _run_pallas_csr_forces(
            fspec, dspec, layout, kargs, batched, bsz
        )
    )

    def gpu_wrapped(
        st: Any, sy: Any, *kargs: Any
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        out = gpu_eval(*kargs)
        force = out[0].reshape(n, dim)
        if fspec.has_torque or (not fspec.only_spheres):
            torque = out[1].reshape(n, ang_dim)
        else:
            torque = jnp.zeros_like(st.torque)
        hist = (
            out[-1].reshape(sy.collider.history.shape)
            if write_hist
            else sy.collider.history
        )
        return force, torque, hist

    def cpu_wrapped(
        _st: Any, _sy: Any, *_kargs: Any
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return pure_fallback()

    return platform_dispatch(gpu_wrapped, cpu_wrapped, state, system, *flat_args)


# ---------------------------------------------------------------------------
# Row-local CSR history remapping (replaces global O(E log E) sort on rebuild)
# ---------------------------------------------------------------------------


def _run_pallas_remap_history(
    hist_width: int,
    layout: PackLayout,
    args: tuple[Any, ...],
    batched: tuple[bool, ...],
    batch_size: int,
) -> jax.Array:
    has_batch = batch_size > 0
    old_off = args[layout.index["old_off"]]
    new_nbr = args[layout.index["new_nbr"]]
    idx_out = layout.index["out_init"]
    N = old_off.shape[-1] - 1
    cap = new_nbr.shape[-1]
    H = hist_width

    k_args = list(args)
    k_batched = list(batched)
    out_shape = (batch_size, cap * H) if has_batch else (cap * H,)
    if has_batch and not k_batched[idx_out]:
        k_args[idx_out] = jnp.broadcast_to(k_args[idx_out][None, :], out_shape)
        k_batched[idx_out] = True

    BLOCK = _REMAP_BLOCK

    def kernel(*refs: Any) -> None:
        n_in = len(k_args)
        new_h_ref = refs[n_in]
        pid = pl.program_id(0)
        b = pl.program_id(1) if has_batch else 0
        rv = RefView(refs[:n_in], tuple(k_batched), layout, b, has_batch)

        i = pid * BLOCK + jnp.arange(BLOCK)
        m_i = i < N
        o_st = rv.load1("old_off", i, m_i, other=0)
        o_en = rv.load1("old_off", i + 1, m_i, other=0)
        n_st = rv.load1("new_off", i, m_i, other=0)
        n_en = rv.load1("new_off", i + 1, m_i, other=0)
        max_old = jnp.max(jnp.where(m_i, o_en - o_st, 0))
        max_new = jnp.max(jnp.where(m_i, n_en - n_st, 0))

        def new_loop(kn: jax.Array, _: None) -> None:
            s_new = n_st + kn
            in_new = m_i & (s_new < n_en)
            dst = rv.load1("new_nbr", s_new, in_new, other=-1)
            active = in_new & (dst >= 0)

            def scan_old(ko: jax.Array, match_slot: jax.Array) -> jax.Array:
                s_old = o_st + ko
                in_old = active & (s_old < o_en)
                od = rv.load1("old_nbr", s_old, in_old, other=-2)
                return jnp.where(in_old & (od == dst), s_old, match_slot)

            found_slot = jax.lax.fori_loop(
                0, max_old, scan_old, jnp.full((BLOCK,), -1, dtype=o_st.dtype)
            )
            found = active & (found_slot >= 0)
            o_base = jnp.where(found, found_slot, 0) * H
            n_base = s_new * H
            for c in range(H):
                val = rv.load1("old_h", o_base + c, found, other=0.0)
                store1(new_h_ref, b, has_batch, n_base + c, val, found)
            return None

        jax.lax.fori_loop(0, max_new, new_loop, None)

    grid = (pl.cdiv(N, BLOCK), batch_size) if has_batch else (pl.cdiv(N, BLOCK),)
    return pl.pallas_call(
        kernel,
        out_shape=shape_dtype(out_shape, k_args[idx_out].dtype),
        grid=grid,
        input_output_aliases={idx_out: 0},
        compiler_params=triton_compiler_params(num_warps=4, num_stages=2),
    )(*k_args)


def pallas_remap_csr_history(
    old_hist: jax.Array,
    old_offsets: jax.Array,
    old_nbr: jax.Array,
    new_offsets: jax.Array,
    new_nbr: jax.Array,
    initialized: jax.Array,
    pure_fallback: Callable[[], jax.Array],
) -> jax.Array:
    """Remap CSR pair history via row-local Triton scan (no global sort)."""
    if (
        not is_pallas_enabled()
        or old_nbr.size == 0
        or old_hist.size == 0
        or old_offsets.shape[-1] <= 1
        or has_manual_axes(old_hist, old_offsets, old_nbr, new_offsets, new_nbr)
    ):
        return pure_fallback()

    tail_shape = old_hist.shape[1:]
    H = math.prod(tail_shape)
    if H <= 0 or H > 32:
        return pure_fallback()
    cap = new_nbr.shape[0]

    layout = PackLayout(
        index={
            "old_h": 0,
            "old_off": 1,
            "old_nbr": 2,
            "new_off": 3,
            "new_nbr": 4,
            "out_init": 5,
        },
        slices={},
    )
    gpu_remap = batched_pallas(
        lambda kargs, batched, bsz: _run_pallas_remap_history(
            H, layout, kargs, batched, bsz
        ),
        single_output=True,
    )

    def gpu_wrapped(*kargs: Any) -> jax.Array:
        return gpu_remap(*kargs).reshape((cap,) + tail_shape)

    def _pure_remap(
        oh: jax.Array,
        init: jax.Array,
        o_off: jax.Array,
        o_nbr: jax.Array,
        n_off: jax.Array,
        n_nbr: jax.Array,
    ) -> jax.Array:
        from .._neighbor_cache import pair_sources, remap_history

        old_src = pair_sources(
            type("_C", (), {"neighbor_list": o_nbr, "row_offsets": o_off})()
        )
        new_src = pair_sources(
            type("_C", (), {"neighbor_list": n_nbr, "row_offsets": n_off})()
        )
        return remap_history(oh, old_src, o_nbr, new_src, n_nbr, init)

    def cpu_wrapped(
        oh_f: jax.Array,
        o_off: jax.Array,
        o_nbr: jax.Array,
        n_off: jax.Array,
        n_nbr: jax.Array,
        init_f: jax.Array,
    ) -> jax.Array:
        return _pure_remap(
            oh_f.reshape(old_hist.shape),
            init_f.reshape((cap,) + tail_shape),
            o_off,
            o_nbr,
            n_off,
            n_nbr,
        )

    @jax.custom_jvp
    def _remap_ad(
        oh: jax.Array,
        init: jax.Array,
        o_off: jax.Array,
        o_nbr: jax.Array,
        n_off: jax.Array,
        n_nbr: jax.Array,
    ) -> jax.Array:
        return platform_dispatch(
            gpu_wrapped,
            cpu_wrapped,
            oh.reshape(-1),
            o_off,
            o_nbr,
            n_off,
            n_nbr,
            init.reshape(-1),
        )

    @_remap_ad.defjvp
    def _remap_ad_jvp(primals: Any, tangents: Any) -> Any:
        p_oh, p_init, p_ooff, p_onbr, p_noff, p_nnbr = primals
        t_oh, t_init, *_ = tangents
        return (
            _remap_ad(*primals),
            jax.jvp(
                lambda a, b: _pure_remap(a, b, p_ooff, p_onbr, p_noff, p_nnbr),
                (p_oh, p_init),
                (t_oh, t_init),
            )[1],
        )

    return _remap_ad(old_hist, initialized, old_offsets, old_nbr, new_offsets, new_nbr)
