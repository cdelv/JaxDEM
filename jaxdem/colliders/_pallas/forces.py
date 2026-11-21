# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force-model AST compilation, schema validation, and Triton evaluation.

Central pair-force formulas (Spring, Hertz, WCA, WCAShifted, Lennard-Jones) as
well as ``harmonic_mean``, ``_safe_sqrt``, and ``_damping_ratio`` are imported
directly from ``jaxdem.forces`` and ``jaxdem.utils.linalg`` so Pure JAX and
Pallas Triton share a single source of truth.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ...forces import _damping_ratio, _safe_sqrt
from ...forces.hertz import _hertz_force_coeff
from ...forces.lennardjones import _lj_force_coeff
from ...forces.spring import _spring_force_coeff
from ...forces.wca import _wca_force_coeff
from ...forces.wca_shifted import _wca_shifted_force_coeff
from ...utils.linalg import harmonic_mean
from .domains import DomainSpec, triton_relative_velocity
from .refs import (
    RefView,
    load1_ref,
    scalar_at,
    store1,
    triton_cross,
    triton_cross_w_r,
    triton_dot,
    triton_scaled_norm,
    triton_scaled_unit_and_norm,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...state import State
    from ...system import System

_LAW_SCHEMA: dict[str, frozenset[str]] = {
    "SpringForce": frozenset({"laws"}),
    "DampedSpringForce": frozenset({"laws"}),
    "HertzianForce": frozenset({"laws"}),
    "WCA": frozenset({"laws"}),
    "WCAShifted": frozenset({"laws"}),
    "LennardJones": frozenset({"cutoff_ratio", "laws"}),
    "CundallStrackForce": frozenset(
        {"friction_mixing", "laws", "parameterization", "rolling_friction_mixing"}
    ),
    "LawCombiner": frozenset({"laws"}),
    "ForceRouter": frozenset({"laws", "table"}),
}


def _matches_law_schema(law: Any) -> bool:
    expected = _LAW_SCHEMA.get(type(law).__name__)
    if expected is None:
        return False
    return {f.name for f in dataclasses.fields(type(law))} == expected


@dataclass(frozen=True)
class LawNode:
    kind: str
    param_start: int
    param_count: int
    n_mat: int = 1
    hist_offset: int = 0
    hist_width: int = 0
    extra: Any = None
    children: tuple[LawNode, ...] = ()


@dataclass(frozen=True)
class ForceSpec:
    root: LawNode
    dim: int
    ang_dim: int
    only_spheres: bool
    advance_history: bool
    quasistatic: bool
    has_torque: bool
    has_history: bool
    hist_width: int
    needs_mat_id: bool
    needs_species_id: bool
    needs_vel: bool
    needs_mass: bool
    needs_dt: bool


def extract_force_spec(
    force_model: Any,
    state: State,
    system: System,
    *,
    advance_history: bool,
    quasistatic: bool,
) -> tuple[ForceSpec, tuple[jax.Array, ...]] | None:
    """Compile a supported ``ForceModel`` tree into a static AST + flat 1D params."""
    from ...forces.cundall_strack import CundallStrackForce
    from ...forces.hertz import HertzianForce
    from ...forces.law_combiner import LawCombiner
    from ...forces.lennardjones import LennardJones
    from ...forces.router import ForceRouter
    from ...forces.spring import DampedSpringForce, SpringForce
    from ...forces.wca import WCA
    from ...forces.wca_shifted import WCAShifted

    dim = int(state.dim)
    if dim not in (2, 3):
        return None
    dtype = state.pos.dtype
    mat_table = system.mat_table
    params: list[jax.Array] = []
    flags = {
        "needs_mat_id": False,
        "needs_species_id": False,
        "needs_vel": False,
        "needs_mass": False,
        "needs_dt": False,
        "has_torque": False,
    }

    def add_param(arr: jax.Array) -> None:
        params.append(jnp.reshape(jnp.asarray(arr, dtype=dtype), (-1,)))

    def mark_mat(m: int) -> int:
        if m > 1:
            flags["needs_mat_id"] = True
        return m

    def compile_node(law: Any, hist_offset: int) -> LawNode | None:
        if not _matches_law_schema(law):
            return None
        t = type(law)
        p0 = len(params)

        if t is SpringForce:
            ye = mat_table.young_eff
            add_param(ye)
            return LawNode("spring", p0, 1, n_mat=mark_mat(int(ye.shape[0])))

        if t is HertzianForce:
            E, nu = mat_table.young, mat_table.poisson
            inv_E = (1.0 - nu * nu) / E
            add_param((4.0 / 3.0) / (inv_E[:, None] + inv_E[None, :]))
            return LawNode("hertz", p0, 1, n_mat=mark_mat(int(E.shape[0])))

        if t in (WCA, WCAShifted):
            eps = mat_table.epsilon_eff
            add_param(eps)
            kind = "wca" if t is WCA else "wca_shifted"
            return LawNode(kind, p0, 1, n_mat=mark_mat(int(eps.shape[0])))

        if t is LennardJones:
            eps = mat_table.epsilon_eff
            add_param(eps)
            add_param(law.cutoff_ratio * law.cutoff_ratio)
            return LawNode("lj", p0, 2, n_mat=mark_mat(int(eps.shape[0])))

        if t is DampedSpringForce:
            ye = mat_table.young_eff
            m = mark_mat(int(ye.shape[0]))
            add_param(ye)
            if quasistatic:
                return LawNode("damped_spring", p0, 1, n_mat=m)
            flags["needs_vel"] = True
            flags["needs_mass"] = True
            e_arr = mat_table.e
            e_min = jnp.minimum(e_arr[:, None], e_arr[None, :])
            add_param(2.0 * _damping_ratio(e_min) * _safe_sqrt(ye))
            return LawNode("damped_spring", p0, 2, n_mat=m)

        if t is CundallStrackForce:
            if law.parameterization not in ("elastic", "coefficients"):
                return None
            # Quasistatic advancing history solves first-touch root per step;
            # keep that rare affine-shear/relaxation mode on the Pure-JAX path.
            if quasistatic and advance_history:
                return None
            flags["has_torque"] = True
            if not quasistatic:
                flags["needs_vel"] = True
                if advance_history:
                    flags["needs_dt"] = True
            mu_arr = mat_table.mu
            m = mark_mat(int(mu_arr.shape[0]))
            mu_mat = jnp.asarray(
                law.friction_mixing(mu_arr[:, None], mu_arr[None, :]), dtype=dtype
            )
            if not quasistatic:
                mu_r_arr = mat_table.mu_r
                mu_r_mat = jnp.asarray(
                    law.rolling_friction_mixing(mu_r_arr[:, None], mu_r_arr[None, :]),
                    dtype=dtype,
                )
            else:
                mu_r_mat = jnp.zeros((m, m), dtype=dtype)

            if law.parameterization == "elastic":
                E_arr, nu_arr = mat_table.young, mat_table.poisson
                add_param(E_arr)
                add_param(E_arr / (2.0 * (1.0 + nu_arr)))
                add_param(mu_mat)
                if not quasistatic:
                    flags["needs_mass"] = True
                    e_arr = mat_table.e
                    e_min = jnp.minimum(e_arr[:, None], e_arr[None, :])
                    add_param(2.0 * _damping_ratio(e_min))
                    add_param(mu_r_mat)
                    return LawNode(
                        "cundall_elastic",
                        p0,
                        5,
                        n_mat=m,
                        hist_offset=hist_offset,
                        hist_width=2 * dim,
                    )
                return LawNode(
                    "cundall_elastic",
                    p0,
                    3,
                    n_mat=m,
                    hist_offset=hist_offset,
                    hist_width=2 * dim,
                )

            kn_arr, kt_arr = mat_table.k_n, mat_table.k_t
            add_param(harmonic_mean(kn_arr[:, None], kn_arr[None, :]))
            add_param(harmonic_mean(kt_arr[:, None], kt_arr[None, :]))
            add_param(mu_mat)
            if not quasistatic:
                bn_arr, bt_arr = mat_table.b_n, mat_table.b_t
                add_param(harmonic_mean(bn_arr[:, None], bn_arr[None, :]))
                add_param(harmonic_mean(bt_arr[:, None], bt_arr[None, :]))
                add_param(mu_r_mat)
                return LawNode(
                    "cundall_coeff",
                    p0,
                    6,
                    n_mat=m,
                    hist_offset=hist_offset,
                    hist_width=2 * dim,
                )
            return LawNode(
                "cundall_coeff",
                p0,
                3,
                n_mat=m,
                hist_offset=hist_offset,
                hist_width=2 * dim,
            )

        if t is LawCombiner:
            if not law.laws:
                return LawNode("empty", p0, 0)
            children: list[LawNode] = []
            cur_off = hist_offset
            for sub in law.laws:
                node = compile_node(sub, cur_off)
                if node is None:
                    return None
                children.append(node)
                cur_off += math.prod(sub.history_shape(dim))
            return LawNode(
                "combiner",
                p0,
                0,
                hist_offset=hist_offset,
                hist_width=cur_off - hist_offset,
                children=tuple(children),
            )

        if t is ForceRouter:
            S = len(law.table)
            if S == 0:
                return LawNode("empty", p0, 0)
            flags["needs_species_id"] = True
            children = []
            cur_off = hist_offset
            for a in range(S):
                for b_sp in range(a, S):
                    sub = law.table[a][b_sp]
                    node = compile_node(sub, cur_off)
                    if node is None:
                        return None
                    children.append(node)
                    cur_off += math.prod(sub.history_shape(dim))
            return LawNode(
                "router",
                p0,
                0,
                hist_offset=hist_offset,
                hist_width=cur_off - hist_offset,
                extra=S,
                children=tuple(children),
            )

        return None

    root = compile_node(force_model, 0)
    if root is None:
        return None
    hist_width = math.prod(force_model.history_shape(dim))
    spec = ForceSpec(
        root=root,
        dim=dim,
        ang_dim=3 if dim == 3 else 1,
        only_spheres=bool(state.only_spheres),
        advance_history=advance_history,
        quasistatic=quasistatic,
        has_torque=flags["has_torque"],
        has_history=hist_width > 0,
        hist_width=hist_width,
        needs_mat_id=flags["needs_mat_id"],
        needs_species_id=flags["needs_species_id"],
        needs_vel=flags["needs_vel"],
        needs_mass=flags["needs_mass"],
        needs_dt=flags["needs_dt"],
    )
    return spec, tuple(params)


def _lookup_mat_pair(
    p_refs: tuple[Any, ...],
    p_batched: tuple[bool, ...],
    idx: int,
    b: Any,
    n_mat: int,
    mi: jax.Array,
    mj: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    if n_mat == 1:
        return scalar_at(p_refs[idx], b, p_batched[idx], 0)
    return load1_ref(p_refs[idx], b, p_batched[idx], mi * n_mat + mj, mask, other=0.0)


def _lookup_mat_single(
    p_refs: tuple[Any, ...],
    p_batched: tuple[bool, ...],
    idx: int,
    b: Any,
    n_mat: int,
    m_idx: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    if n_mat == 1:
        return scalar_at(p_refs[idx], b, p_batched[idx], 0)
    return load1_ref(p_refs[idx], b, p_batched[idx], m_idx, mask, other=0.0)


def _contact_rel_vel(
    rv: RefView,
    fspec: ForceSpec,
    dspec: DomainSpec,
    dom: dict[str, Any],
    ctx: dict[str, Any],
    n_vec: tuple[jax.Array, ...],
    mask: jax.Array,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Compute contact-point relative velocity, returning ``(v_rel, r_ci, wj)``."""
    dim, ang_dim = fspec.dim, fspec.ang_dim
    r_ci = tuple(-ctx["ri"] * n_vec[d] for d in range(dim))
    r_cj = tuple(ctx["rj"] * n_vec[d] for d in range(dim))
    vj = rv.load_vec("vel", ctx["j"], dim, mask)
    wj = rv.load_vec("ang", ctx["j"], ang_dim, mask)
    if fspec.only_spheres:
        arm_i, arm_j = r_ci, r_cj
    else:
        ppr_j = rv.load_vec("ppr", ctx["j"], dim, mask)
        arm_i = tuple(ctx["ppr_i"][d] + r_ci[d] for d in range(dim))
        arm_j = tuple(ppr_j[d] + r_cj[d] for d in range(dim))
    wxi = triton_cross_w_r(dim, ctx["ang_i"], arm_i)
    wxj = triton_cross_w_r(dim, wj, arm_j)
    v_ci = tuple(ctx["vel_i"][d] + wxi[d] for d in range(dim))
    v_cj = tuple(vj[d] + wxj[d] for d in range(dim))
    v_rel = triton_relative_velocity(dspec, v_ci, v_cj, ctx["img"], ctx["shear_img"], dom)
    return v_rel, r_ci, wj


def _transport_tangent(
    dim: int,
    xi: tuple[jax.Array, ...],
    pn: tuple[jax.Array, ...],
    n: tuple[jax.Array, ...],
) -> tuple[jax.Array, ...]:
    """Shortest-rotation transport of tangential spring ``xi`` from ``pn`` to ``n``."""
    if dim == 2:
        c = pn[0] * xi[1] - pn[1] * xi[0]
        rot: tuple[jax.Array, ...] = (-n[1] * c, n[0] * c)
    else:
        dpn = triton_dot(xi, pn)
        tan = tuple(xi[d] - dpn * pn[d] for d in range(3))
        bis = tuple(pn[d] + n[d] for d in range(3))
        bn2 = triton_dot(bis, bis)
        s = 2.0 * triton_dot(tan, bis) / jnp.maximum(bn2, 1e-12)
        reg = tuple(tan[d] - s * bis[d] for d in range(3))
        ax_a, ay_a, az_a = jnp.abs(pn[0]), jnp.abs(pn[1]), jnp.abs(pn[2])
        is_x = (ax_a <= ay_a) & (ax_a <= az_a)
        is_y = (~is_x) & (ay_a <= az_a)
        e_vec = (
            jnp.where(is_x, 1.0, 0.0),
            jnp.where(is_y, 1.0, 0.0),
            jnp.where((~is_x) & (~is_y), 1.0, 0.0),
        )
        axis, _ = triton_scaled_unit_and_norm(triton_cross(3, pn, e_vec))
        du = 2.0 * triton_dot(axis, xi)
        anti_v = tuple(du * axis[d] - xi[d] for d in range(3))
        anti = bn2 < 2e-6
        rot = tuple(jnp.where(anti, anti_v[d], reg[d]) for d in range(3))
    has_pn = triton_dot(pn, pn) > 0.0
    rot = tuple(jnp.where(has_pn, rot[d], xi[d]) for d in range(dim))
    dn = triton_dot(rot, n)
    return tuple(rot[d] - dn * n[d] for d in range(dim))


def _eval_cundall(
    node: LawNode,
    rv: RefView,
    fspec: ForceSpec,
    dspec: DomainSpec,
    dom: dict[str, Any],
    ctx: dict[str, Any],
    active_mask: jax.Array,
    hist_write_mask: jax.Array,
    p_refs: tuple[Any, ...],
    p_batched: tuple[bool, ...],
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Evaluate Cundall-Strack (elastic or coefficients) with fast non-contact skip."""
    dim, ang_dim = fspec.dim, fspec.ang_dim
    zero_f = tuple(jnp.zeros_like(ctx["ri"]) for _ in range(dim))
    zero_t = tuple(jnp.zeros_like(ctx["ri"]) for _ in range(ang_dim))
    is_contact = active_mask & (ctx["n2"] < ctx["R"] * ctx["R"])
    slot = ctx["slot"]
    H = fspec.hist_width
    hoff = node.hist_offset
    p0 = node.param_start
    b = rv.b
    mi, mj = ctx["mi"], ctx["mj"]

    def _do_contact(_: None) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        n_vec, dist = triton_scaled_unit_and_norm(ctx["dr"])
        overlap = jnp.maximum(0.0, ctx["R"] - dist)
        c_mask = is_contact & (overlap > 0.0)
        delta = jnp.where(c_mask, overlap, 0.0)

        if slot is not None and H > 0:
            h_base = slot * H + hoff
            xi_old = tuple(
                rv.load1("hist", h_base + d, c_mask, other=0.0) for d in range(dim)
            )
            pn_old = tuple(
                rv.load1("hist", h_base + dim + d, c_mask, other=0.0)
                for d in range(dim)
            )
        else:
            xi_old, pn_old = zero_f, zero_f

        if node.kind == "cundall_elastic":
            E_i = _lookup_mat_single(p_refs, p_batched, p0, b, node.n_mat, mi, c_mask)
            E_j = _lookup_mat_single(p_refs, p_batched, p0, b, node.n_mat, mj, c_mask)
            G_i = _lookup_mat_single(p_refs, p_batched, p0 + 1, b, node.n_mat, mi, c_mask)
            G_j = _lookup_mat_single(p_refs, p_batched, p0 + 1, b, node.n_mat, mj, c_mask)
            mu_ij = _lookup_mat_pair(p_refs, p_batched, p0 + 2, b, node.n_mat, mi, mj, c_mask)
            kn = harmonic_mean(E_i * ctx["ri"], E_j * ctx["rj"])
            kt = harmonic_mean(G_i * ctx["ri"], G_j * ctx["rj"])
            if not fspec.quasistatic:
                two_beta = _lookup_mat_pair(
                    p_refs, p_batched, p0 + 3, b, node.n_mat, mi, mj, c_mask
                )
                mu_r_ij = _lookup_mat_pair(
                    p_refs, p_batched, p0 + 4, b, node.n_mat, mi, mj, c_mask
                )
                m_j = rv.load1("mass", ctx["j"], c_mask, other=1.0)
                m_root = _safe_sqrt(0.5 * harmonic_mean(ctx["mass_i"], m_j))
                gamma_n = two_beta * _safe_sqrt(kn) * m_root
                gamma_t = two_beta * _safe_sqrt(kt) * m_root
            else:
                gamma_n = gamma_t = mu_r_ij = jnp.zeros_like(delta)
        else:
            kn = _lookup_mat_pair(p_refs, p_batched, p0, b, node.n_mat, mi, mj, c_mask)
            kt = _lookup_mat_pair(p_refs, p_batched, p0 + 1, b, node.n_mat, mi, mj, c_mask)
            mu_ij = _lookup_mat_pair(p_refs, p_batched, p0 + 2, b, node.n_mat, mi, mj, c_mask)
            if not fspec.quasistatic:
                gamma_n = _lookup_mat_pair(
                    p_refs, p_batched, p0 + 3, b, node.n_mat, mi, mj, c_mask
                )
                gamma_t = _lookup_mat_pair(
                    p_refs, p_batched, p0 + 4, b, node.n_mat, mi, mj, c_mask
                )
                mu_r_ij = _lookup_mat_pair(
                    p_refs, p_batched, p0 + 5, b, node.n_mat, mi, mj, c_mask
                )
            else:
                gamma_n = gamma_t = mu_r_ij = jnp.zeros_like(delta)

        if not fspec.quasistatic:
            v_rel, r_ci, wj = _contact_rel_vel(rv, fspec, dspec, dom, ctx, n_vec, c_mask)
            vn = triton_dot(v_rel, n_vec)
            vt = tuple(v_rel[d] - vn * n_vec[d] for d in range(dim))
            normal_raw = kn * delta - gamma_n * vn
        else:
            r_ci = tuple(-ctx["ri"] * n_vec[d] for d in range(dim))
            wj = zero_t
            vt = zero_f
            normal_raw = kn * delta

        Fn = jnp.where(c_mask, jnp.maximum(0.0, normal_raw), 0.0)
        xi = _transport_tangent(dim, xi_old, pn_old, n_vec)

        if fspec.advance_history and not fspec.quasistatic:
            dt = ctx["dt"]
            if dim == 3:
                w_sum = tuple(ctx["ang_i"][d] + wj[d] for d in range(3))
                spin = 0.5 * triton_dot(w_sum, n_vec) * dt
                cs, sn = jnp.cos(spin), jnp.sin(spin)
                nx_xi = triton_cross(3, n_vec, xi)
                xi = tuple(xi[d] * cs + nx_xi[d] * sn for d in range(3))
            xi_eval = tuple(xi[d] + vt[d] * dt for d in range(dim))
        else:
            xi_eval = xi

        if not fspec.quasistatic:
            Ft_trial = tuple(-kt * xi_eval[d] - gamma_t * vt[d] for d in range(dim))
        else:
            Ft_trial = tuple(-kt * xi_eval[d] for d in range(dim))

        Ft_norm = triton_scaled_norm(Ft_trial)
        Ft_max = mu_ij * Fn
        scale = jnp.minimum(1.0, Ft_max / jnp.where(Ft_norm > 0.0, Ft_norm, 1.0))
        Ft = tuple(jnp.where(c_mask, Ft_trial[d] * scale, 0.0) for d in range(dim))

        if fspec.advance_history and slot is not None and H > 0:
            safe_kt = jnp.where(kt > 0.0, kt, 1.0)
            kt_pos = kt > 0.0
            sliding = Ft_norm > Ft_max
            h_base = slot * H + hoff
            w_c = hist_write_mask & is_contact
            h_out = ctx["hist_out_ref"]
            for d in range(dim):
                sf_d = Ft[d] if fspec.quasistatic else Ft[d] + gamma_t * vt[d]
                xi_ret_d = jnp.where(kt_pos, -sf_d / safe_kt, 0.0)
                xi_next_d = jnp.where(sliding, xi_ret_d, xi_eval[d])
                store1(h_out, b, rv.has_batch, h_base + d, jnp.where(c_mask, xi_next_d, 0.0), w_c)
                store1(h_out, b, rv.has_batch, h_base + dim + d, jnp.where(c_mask, n_vec[d], 0.0), w_c)

        F_out = tuple(Fn * n_vec[d] + Ft[d] for d in range(dim))
        T_out = triton_cross(dim, r_ci, Ft)
        if not fspec.quasistatic:
            R_eff = 0.5 * harmonic_mean(ctx["ri"], ctx["rj"])
            dw = tuple(ctx["ang_i"][d] - wj[d] for d in range(ang_dim))
            if dim == 2:
                w_hat: tuple[jax.Array, ...] = (
                    jnp.where(dw[0] > 0.0, 1.0, jnp.where(dw[0] < 0.0, -1.0, 0.0)),
                )
            else:
                w_hat, _wn = triton_scaled_unit_and_norm(dw)
            roll_coeff = mu_r_ij * R_eff * Fn
            T_out = tuple(T_out[d] - roll_coeff * w_hat[d] for d in range(ang_dim))
        return F_out, T_out

    has_any_contact = jnp.max(is_contact.astype(jnp.int32)) != 0
    F_out, T_out = jax.lax.cond(
        has_any_contact, _do_contact, lambda _: (zero_f, zero_t), None
    )

    # Clear history only for slots that just broke contact (avoids writing
    # 24 bytes/pair on ~80-97% of already-zero Verlet skin pairs).
    if fspec.advance_history and slot is not None and H > 0:
        non_c = hist_write_mask & (~is_contact)
        h_base = slot * H + hoff
        any_nz = jnp.zeros_like(non_c)
        for d in range(2 * dim):
            hv = rv.load1("hist", h_base + d, non_c, other=0.0, eviction_policy="evict_first")
            any_nz = any_nz | (hv != 0.0)
        need_clr = non_c & any_nz

        def _do_clear(_: None) -> None:
            z_val = jnp.zeros_like(ctx["ri"])
            h_out = ctx["hist_out_ref"]
            for d in range(2 * dim):
                store1(h_out, b, rv.has_batch, h_base + d, z_val, need_clr)
            return None

        jax.lax.cond(
            jnp.max(need_clr.astype(jnp.int32)) != 0, _do_clear, lambda _: None, None
        )

    return F_out, T_out


def eval_law_node(
    node: LawNode,
    rv: RefView,
    fspec: ForceSpec,
    dspec: DomainSpec,
    dom: dict[str, Any],
    ctx: dict[str, Any],
    active_mask: jax.Array,
    hist_write_mask: jax.Array,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Evaluate a compiled ``LawNode`` in Triton registers for one neighbor lane."""
    dim, ang_dim = fspec.dim, fspec.ang_dim
    zero_f = tuple(jnp.zeros_like(ctx["ri"]) for _ in range(dim))
    zero_t = tuple(jnp.zeros_like(ctx["ri"]) for _ in range(ang_dim))

    if node.kind == "empty":
        return zero_f, zero_t

    if node.kind == "combiner":
        fx, tx = zero_f, zero_t
        for ch in node.children:
            cf, ct = eval_law_node(ch, rv, fspec, dspec, dom, ctx, active_mask, hist_write_mask)
            fx = tuple(fx[d] + cf[d] for d in range(dim))
            if fspec.has_torque:
                tx = tuple(tx[d] + ct[d] for d in range(ang_dim))
        return fx, tx

    if node.kind == "router":
        S = int(node.extra)
        si, sj = ctx["si"], ctx["sj"]
        lo, hi = jnp.minimum(si, sj), jnp.maximum(si, sj)
        fx, tx = zero_f, zero_t
        idx_c = 0
        for a in range(S):
            for b_sp in range(a, S):
                ch = node.children[idx_c]
                idx_c += 1
                sel = (lo == a) & (hi == b_sp)
                ch_write = hist_write_mask & ((~ctx["valid"]) | sel)
                cf, ct = eval_law_node(
                    ch, rv, fspec, dspec, dom, ctx, active_mask & sel, ch_write
                )
                fx = tuple(jnp.where(sel, cf[d], fx[d]) for d in range(dim))
                if fspec.has_torque:
                    tx = tuple(jnp.where(sel, ct[d], tx[d]) for d in range(ang_dim))
        return fx, tx

    p_refs, p_batched = rv.group("params")
    p0 = node.param_start
    b = rv.b
    dr, n2, R = ctx["dr"], ctx["n2"], ctx["R"]
    mi, mj = ctx["mi"], ctx["mj"]

    if node.kind == "spring":
        k = _lookup_mat_pair(p_refs, p_batched, p0, b, node.n_mat, mi, mj, active_mask)
        coeff = jnp.where(active_mask, _spring_force_coeff(n2, R, k), 0.0)
        return tuple(coeff * dr[d] for d in range(dim)), zero_t

    if node.kind == "hertz":
        ft_E = _lookup_mat_pair(p_refs, p_batched, p0, b, node.n_mat, mi, mj, active_mask)
        coeff = jnp.where(active_mask, _hertz_force_coeff(n2, ctx["ri"], ctx["rj"], ft_E), 0.0)
        return tuple(coeff * dr[d] for d in range(dim)), zero_t

    if node.kind == "wca":
        eps = _lookup_mat_pair(p_refs, p_batched, p0, b, node.n_mat, mi, mj, active_mask)
        coeff = jnp.where(active_mask, _wca_force_coeff(n2, R, eps), 0.0)
        return tuple(coeff * dr[d] for d in range(dim)), zero_t

    if node.kind == "wca_shifted":
        eps = _lookup_mat_pair(p_refs, p_batched, p0, b, node.n_mat, mi, mj, active_mask)
        coeff = jnp.where(active_mask, _wca_shifted_force_coeff(n2, R, eps), 0.0)
        return tuple(coeff * dr[d] for d in range(dim)), zero_t

    if node.kind == "lj":
        eps = _lookup_mat_pair(p_refs, p_batched, p0, b, node.n_mat, mi, mj, active_mask)
        cr2 = scalar_at(p_refs[p0 + 1], b, p_batched[p0 + 1], 0)
        coeff = jnp.where(active_mask, _lj_force_coeff(n2, R, eps, cr2), 0.0)
        return tuple(coeff * dr[d] for d in range(dim)), zero_t

    if node.kind == "damped_spring":
        n_vec, dist = triton_scaled_unit_and_norm(dr)
        overlap = jnp.maximum(0.0, R - dist)
        contact = active_mask & (overlap > 0.0)
        k = _lookup_mat_pair(p_refs, p_batched, p0, b, node.n_mat, mi, mj, contact)
        mag = k * overlap
        if not fspec.quasistatic:
            tb_sk = _lookup_mat_pair(p_refs, p_batched, p0 + 1, b, node.n_mat, mi, mj, contact)
            m_j = rv.load1("mass", ctx["j"], contact, other=1.0)
            damping = tb_sk * _safe_sqrt(0.5 * harmonic_mean(ctx["mass_i"], m_j))
            v_rel, _, _ = _contact_rel_vel(rv, fspec, dspec, dom, ctx, n_vec, contact)
            mag = mag - damping * triton_dot(v_rel, n_vec)
        fn = jnp.where(contact, jnp.maximum(0.0, mag), 0.0)
        return tuple(fn * n_vec[d] for d in range(dim)), zero_t

    if node.kind in ("cundall_elastic", "cundall_coeff"):
        return _eval_cundall(
            node, rv, fspec, dspec, dom, ctx, active_mask, hist_write_mask, p_refs, p_batched
        )

    return zero_f, zero_t
