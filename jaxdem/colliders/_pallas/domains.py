# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Domain schema validation and Triton displacement / relative-velocity helpers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .refs import RefView, scalar_at, triton_rint

_DOMAIN_SCHEMA: dict[str, frozenset[str]] = {
    "FreeDomain": frozenset({"anchor", "box_size", "inv_box_size"}),
    "ReflectDomain": frozenset(
        {"anchor", "box_size", "inv_box_size", "restitution_coefficient"}
    ),
    "ReflectSphereDomain": frozenset(
        {"anchor", "box_size", "inv_box_size", "restitution_coefficient"}
    ),
    "PeriodicDomain": frozenset({"anchor", "box_rate", "box_size", "inv_box_size"}),
    "LeesEdwardsDomain": frozenset(
        {
            "alpha",
            "alpha_axis",
            "anchor",
            "beta",
            "beta_axis",
            "box_rate",
            "box_size",
            "gamma",
            "gamma_dot",
            "inv_box_size",
        }
    ),
}


def _matches_schema(obj: Any, expected: frozenset[str]) -> bool:
    """Reject classes whose dataclass fields changed without updating Pallas."""
    return {f.name for f in dataclasses.fields(type(obj))} == expected


@dataclass(frozen=True)
class DomainSpec:
    kind: str  # "nonperiodic" | "periodic" | "lees_edwards"
    dim: int
    alpha: int = 0
    beta: int = 1


def extract_domain_spec(
    domain: Any, dim: int, *, need_vel_rates: bool
) -> tuple[DomainSpec, tuple[jax.Array, ...]] | None:
    """Extract a static ``DomainSpec`` and 1D parameter arrays, or ``None``."""
    from ...domains.free import FreeDomain
    from ...domains.lees_edwards import LeesEdwardsDomain
    from ...domains.periodic import PeriodicDomain
    from ...domains.reflect import ReflectDomain
    from ...domains.reflect_sphere import ReflectSphereDomain

    if dim not in (2, 3):
        return None
    dtype = domain.box_size.dtype
    t = type(domain)
    expected = _DOMAIN_SCHEMA.get(t.__name__)
    if expected is None or not _matches_schema(domain, expected):
        return None

    if t in (FreeDomain, ReflectDomain, ReflectSphereDomain):
        return DomainSpec("nonperiodic", dim), ()

    box = domain.box_size
    inv_box = 1.0 / box
    if t is PeriodicDomain:
        params = (box, inv_box, domain.box_rate) if need_vel_rates else (box, inv_box)
        return DomainSpec("periodic", dim), params

    if t is LeesEdwardsDomain:
        gamma = jnp.reshape(domain.gamma.astype(dtype), (1,))
        spec = DomainSpec("lees_edwards", dim, int(domain.alpha), int(domain.beta))
        if need_vel_rates:
            gamma_dot = jnp.reshape(domain.gamma_dot.astype(dtype), (1,))
            return spec, (box, inv_box, gamma, domain.box_rate, gamma_dot)
        return spec, (box, inv_box, gamma)

    return None


def load_domain_scalars(
    rv: RefView, spec: DomainSpec, *, need_vel_rates: bool
) -> dict[str, Any]:
    """Load small domain scalars once per Triton block into registers."""
    if spec.kind == "nonperiodic":
        return {}
    dom_refs, dom_batched = rv.group("dom")
    b = rv.b
    out: dict[str, Any] = {
        "box": tuple(
            scalar_at(dom_refs[0], b, dom_batched[0], d) for d in range(spec.dim)
        ),
        "inv": tuple(
            scalar_at(dom_refs[1], b, dom_batched[1], d) for d in range(spec.dim)
        ),
    }
    if spec.kind == "periodic":
        if need_vel_rates:
            out["box_rate"] = tuple(
                scalar_at(dom_refs[2], b, dom_batched[2], d) for d in range(spec.dim)
            )
        return out
    out["gamma"] = scalar_at(dom_refs[2], b, dom_batched[2], 0)
    if need_vel_rates:
        out["box_rate"] = tuple(
            scalar_at(dom_refs[3], b, dom_batched[3], d) for d in range(spec.dim)
        )
        out["gamma_dot"] = scalar_at(dom_refs[4], b, dom_batched[4], 0)
    return out


def triton_displacement(
    spec: DomainSpec,
    pi: tuple[jax.Array, ...],
    pj: tuple[jax.Array, ...],
    dom: dict[str, Any],
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...] | None, jax.Array | None]:
    """Compute minimum-image displacement and periodic image counts in registers."""
    raw = tuple(pi[d] - pj[d] for d in range(spec.dim))
    if spec.kind == "nonperiodic":
        return raw, None, None
    box, inv = dom["box"], dom["inv"]
    if spec.kind == "periodic":
        img = tuple(triton_rint(raw[d] * inv[d]) for d in range(spec.dim))
        dr = tuple(raw[d] - box[d] * img[d] for d in range(spec.dim))
        return dr, img, None

    # Lees-Edwards
    a, b_ax = spec.alpha, spec.beta
    gamma, L_b = dom["gamma"], box[b_ax]
    sb = triton_rint(raw[b_ax] * inv[b_ax])
    db = raw[b_ax] - L_b * sb
    sa = raw[a] - sb * (L_b * gamma)
    ia = triton_rint(sa * inv[a])
    da = sa - box[a] * ia
    if spec.dim == 2:
        if a == 0:
            return (da, db), (ia, sb), sb
        return (db, da), (sb, ia), sb
    o = 3 - a - b_ax
    io = triton_rint(raw[o] * inv[o])
    do = raw[o] - box[o] * io
    dr_list = [da, db, do]
    img_list = [ia, sb, io]
    dr_list[a], dr_list[b_ax], dr_list[o] = da, db, do
    img_list[a], img_list[b_ax], img_list[o] = ia, sb, io
    return (
        (dr_list[0], dr_list[1], dr_list[2]),
        (img_list[0], img_list[1], img_list[2]),
        sb,
    )


def triton_relative_velocity(
    spec: DomainSpec,
    v_ci: tuple[jax.Array, ...],
    v_cj: tuple[jax.Array, ...],
    img: tuple[jax.Array, ...] | None,
    shear_img: jax.Array | None,
    dom: dict[str, Any],
) -> tuple[jax.Array, ...]:
    """Compute contact-point relative velocity reusing image counts from displacement."""
    dv = tuple(v_ci[d] - v_cj[d] for d in range(spec.dim))
    if spec.kind == "nonperiodic":
        return dv
    assert img is not None
    box_rate = dom["box_rate"]
    if spec.kind == "periodic":
        return tuple(dv[d] - img[d] * box_rate[d] for d in range(spec.dim))
    assert shear_img is not None
    a, b_ax = spec.alpha, spec.beta
    shear_vel = shear_img * (
        dom["gamma_dot"] * dom["box"][b_ax] + dom["gamma"] * box_rate[b_ax]
    )
    out = [dv[d] - img[d] * box_rate[d] for d in range(spec.dim)]
    out[a] = out[a] - shear_vel
    return tuple(out)
