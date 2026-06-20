# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Facet contact force model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..utils.linalg import dot, norm

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
def point_triangle_distance(
    p: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Distance from point p to triangle a, b, c."""
    ab = system.domain._displacement(b, a, system)
    ac = system.domain._displacement(c, a, system)
    ap = system.domain._displacement(p, a, system)

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)

    is_a = (d1 <= 0.0) * (d2 <= 0.0)

    bp = ap - ab
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)

    is_b = (d3 >= 0.0) * (d4 <= d3)

    vc = d1 * d4 - d3 * d2

    v_ab = (vc <= 0.0) * (d1 >= 0.0) * (d3 <= 0.0)
    denom_ab = d1 - d3
    v_ab_val = d1 / jnp.where(denom_ab != 0.0, denom_ab, 1.0)

    cp = ap - ac
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    is_c = (d6 >= 0.0) * (d5 <= d6)

    vb = d5 * d2 - d1 * d6

    v_ac = (vb <= 0.0) * (d2 >= 0.0) * (d6 <= 0.0)
    denom_ac = d2 - d6
    v_ac_val = d2 / jnp.where(denom_ac != 0.0, denom_ac, 1.0)

    va = d3 * d6 - d5 * d4

    v_bc = (va <= 0.0) * ((d4 - d3) >= 0.0) * ((d5 - d6) >= 0.0)
    bc = ac - ab
    denom_bc = (d4 - d3) + (d5 - d6)
    v_bc_val = (d4 - d3) / jnp.where(denom_bc != 0.0, denom_bc, 1.0)

    denom_tot = va + vb + vc
    denom = 1.0 / jnp.where(denom_tot != 0.0, denom_tot, 1.0)
    v_val = vb * denom
    w_val = vc * denom

    closest = jnp.where(
        is_a[..., None],
        a,
        jnp.where(
            is_b[..., None],
            b,
            jnp.where(
                is_c[..., None],
                c,
                jnp.where(
                    v_ab[..., None],
                    a + v_ab_val[..., None] * ab,
                    jnp.where(
                        v_ac[..., None],
                        a + v_ac_val[..., None] * ac,
                        jnp.where(
                            v_bc[..., None],
                            b + v_bc_val[..., None] * bc,
                            a + v_val[..., None] * ab + w_val[..., None] * ac,
                        ),
                    ),
                ),
            ),
        ),
    )

    zeros = jnp.zeros_like(d1)
    ones = jnp.ones_like(d1)

    c_a = jnp.stack([ones, zeros, zeros], axis=-1)
    c_b = jnp.stack([zeros, ones, zeros], axis=-1)
    c_c = jnp.stack([zeros, zeros, ones], axis=-1)
    c_ab = jnp.stack([ones - v_ab_val, v_ab_val, zeros], axis=-1)
    c_ac = jnp.stack([ones - v_ac_val, zeros, v_ac_val], axis=-1)
    c_bc = jnp.stack([zeros, ones - v_bc_val, v_bc_val], axis=-1)
    c_in = jnp.stack([ones - v_val - w_val, v_val, w_val], axis=-1)

    coords = jnp.where(
        is_a[..., None],
        c_a,
        jnp.where(
            is_b[..., None],
            c_b,
            jnp.where(
                is_c[..., None],
                c_c,
                jnp.where(
                    v_ab[..., None],
                    c_ab,
                    jnp.where(
                        v_ac[..., None],
                        c_ac,
                        jnp.where(
                            v_bc[..., None],
                            c_bc,
                            c_in,
                        ),
                    ),
                ),
            ),
        ),
    )

    return norm(system.domain._displacement(p, closest, system)), closest, coords


@jax.jit
def segment_segment_distance(
    p1: jax.Array, q1: jax.Array, p2: jax.Array, q2: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Distance between two line segments p1q1 and p2q2."""
    d1 = system.domain._displacement(q1, p1, system)
    d2 = system.domain._displacement(q2, p2, system)
    r = system.domain._displacement(p1, p2, system)
    a = dot(d1, d1)
    e = dot(d2, d2)
    f = dot(d2, r)

    b = dot(d1, d2)
    c = dot(d1, r)

    denom = a * e - b * b
    safe_denom = jnp.where(denom != 0.0, denom, 1.0)
    s_val = (b * f - c * e) / safe_denom

    s = jnp.where(denom != 0.0, jnp.clip(s_val, 0.0, 1.0), 0.0)

    safe_e = jnp.where(e != 0.0, e, 1.0)
    t = (b * s + f) / safe_e

    safe_a = jnp.where(a != 0.0, a, 1.0)
    s = jnp.where(
        t < 0.0,
        jnp.clip(-c / safe_a, 0.0, 1.0),
        jnp.where(t > 1.0, jnp.clip((b - c) / safe_a, 0.0, 1.0), s),
    )

    t = jnp.clip(t, 0.0, 1.0)

    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]

    coords1 = jnp.stack([1.0 - s, s], axis=-1)
    coords2 = jnp.stack([1.0 - t, t], axis=-1)

    return norm(system.domain._displacement(c1, c2, system)), c1, c2, coords1, coords2


@jax.jit
def triangle_triangle_distance(
    t1_a: jax.Array,
    t1_b: jax.Array,
    t1_c: jax.Array,
    t2_a: jax.Array,
    t2_b: jax.Array,
    t2_c: jax.Array,
    system: "System",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shortest distance and closest points between two triangles."""

    # 3 points of T1 to T2
    d1, c1, coords_c1 = point_triangle_distance(t1_a, t2_a, t2_b, t2_c, system)
    d2, c2, coords_c2 = point_triangle_distance(t1_b, t2_a, t2_b, t2_c, system)
    d3, c3, coords_c3 = point_triangle_distance(t1_c, t2_a, t2_b, t2_c, system)

    # 3 points of T2 to T1
    d4, c4, coords_c4 = point_triangle_distance(t2_a, t1_a, t1_b, t1_c, system)
    d5, c5, coords_c5 = point_triangle_distance(t2_b, t1_a, t1_b, t1_c, system)
    d6, c6, coords_c6 = point_triangle_distance(t2_c, t1_a, t1_b, t1_c, system)

    # 9 segment-segment distances
    d7, c7_1, c7_2, coords_c7_1, coords_c7_2 = segment_segment_distance(
        t1_a, t1_b, t2_a, t2_b, system
    )
    d8, c8_1, c8_2, coords_c8_1, coords_c8_2 = segment_segment_distance(
        t1_a, t1_b, t2_b, t2_c, system
    )
    d9, c9_1, c9_2, coords_c9_1, coords_c9_2 = segment_segment_distance(
        t1_a, t1_b, t2_c, t2_a, system
    )

    d10, c10_1, c10_2, coords_c10_1, coords_c10_2 = segment_segment_distance(
        t1_b, t1_c, t2_a, t2_b, system
    )
    d11, c11_1, c11_2, coords_c11_1, coords_c11_2 = segment_segment_distance(
        t1_b, t1_c, t2_b, t2_c, system
    )
    d12, c12_1, c12_2, coords_c12_1, coords_c12_2 = segment_segment_distance(
        t1_b, t1_c, t2_c, t2_a, system
    )

    d13, c13_1, c13_2, coords_c13_1, coords_c13_2 = segment_segment_distance(
        t1_c, t1_a, t2_a, t2_b, system
    )
    d14, c14_1, c14_2, coords_c14_1, coords_c14_2 = segment_segment_distance(
        t1_c, t1_a, t2_b, t2_c, system
    )
    d15, c15_1, c15_2, coords_c15_1, coords_c15_2 = segment_segment_distance(
        t1_c, t1_a, t2_c, t2_a, system
    )

    zeros = jnp.zeros_like(d1)
    ones = jnp.ones_like(d1)

    c1_t1 = jnp.stack([ones, zeros, zeros], axis=-1)
    c2_t1 = jnp.stack([zeros, ones, zeros], axis=-1)
    c3_t1 = jnp.stack([zeros, zeros, ones], axis=-1)
    c4_t1 = coords_c4
    c5_t1 = coords_c5
    c6_t1 = coords_c6
    c7_t1 = jnp.stack([coords_c7_1[..., 0], coords_c7_1[..., 1], zeros], axis=-1)
    c8_t1 = jnp.stack([coords_c8_1[..., 0], coords_c8_1[..., 1], zeros], axis=-1)
    c9_t1 = jnp.stack([coords_c9_1[..., 0], coords_c9_1[..., 1], zeros], axis=-1)
    c10_t1 = jnp.stack([zeros, coords_c10_1[..., 0], coords_c10_1[..., 1]], axis=-1)
    c11_t1 = jnp.stack([zeros, coords_c11_1[..., 0], coords_c11_1[..., 1]], axis=-1)
    c12_t1 = jnp.stack([zeros, coords_c12_1[..., 0], coords_c12_1[..., 1]], axis=-1)
    c13_t1 = jnp.stack([coords_c13_1[..., 1], zeros, coords_c13_1[..., 0]], axis=-1)
    c14_t1 = jnp.stack([coords_c14_1[..., 1], zeros, coords_c14_1[..., 0]], axis=-1)
    c15_t1 = jnp.stack([coords_c15_1[..., 1], zeros, coords_c15_1[..., 0]], axis=-1)

    c1_t2 = coords_c1
    c2_t2 = coords_c2
    c3_t2 = coords_c3
    c4_t2 = jnp.stack([ones, zeros, zeros], axis=-1)
    c5_t2 = jnp.stack([zeros, ones, zeros], axis=-1)
    c6_t2 = jnp.stack([zeros, zeros, ones], axis=-1)
    c7_t2 = jnp.stack([coords_c7_2[..., 0], coords_c7_2[..., 1], zeros], axis=-1)
    c8_t2 = jnp.stack([zeros, coords_c8_2[..., 0], coords_c8_2[..., 1]], axis=-1)
    c9_t2 = jnp.stack([coords_c9_2[..., 1], zeros, coords_c9_2[..., 0]], axis=-1)
    c10_t2 = jnp.stack([coords_c10_2[..., 0], coords_c10_2[..., 1], zeros], axis=-1)
    c11_t2 = jnp.stack([zeros, coords_c11_2[..., 0], coords_c11_2[..., 1]], axis=-1)
    c12_t2 = jnp.stack([coords_c12_2[..., 1], zeros, coords_c12_2[..., 0]], axis=-1)
    c13_t2 = jnp.stack([coords_c13_2[..., 0], coords_c13_2[..., 1], zeros], axis=-1)
    c14_t2 = jnp.stack([zeros, coords_c14_2[..., 0], coords_c14_2[..., 1]], axis=-1)
    c15_t2 = jnp.stack([coords_c15_2[..., 1], zeros, coords_c15_2[..., 0]], axis=-1)

    distances = jnp.stack(
        [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15], axis=-1
    )

    min_idx = jnp.argmin(distances, axis=-1)
    min_dist = jnp.min(distances, axis=-1)

    min_idx_3 = jnp.broadcast_to(min_idx[..., None], t1_a.shape)

    closest_1 = jax.lax.select_n(
        min_idx_3,
        t1_a,
        t1_b,
        t1_c,
        c4,
        c5,
        c6,
        c7_1,
        c8_1,
        c9_1,
        c10_1,
        c11_1,
        c12_1,
        c13_1,
        c14_1,
        c15_1,
    )
    closest_2 = jax.lax.select_n(
        min_idx_3,
        c1,
        c2,
        c3,
        t2_a,
        t2_b,
        t2_c,
        c7_2,
        c8_2,
        c9_2,
        c10_2,
        c11_2,
        c12_2,
        c13_2,
        c14_2,
        c15_2,
    )
    closest_coords1 = jax.lax.select_n(
        min_idx_3,
        c1_t1,
        c2_t1,
        c3_t1,
        c4_t1,
        c5_t1,
        c6_t1,
        c7_t1,
        c8_t1,
        c9_t1,
        c10_t1,
        c11_t1,
        c12_t1,
        c13_t1,
        c14_t1,
        c15_t1,
    )
    closest_coords2 = jax.lax.select_n(
        min_idx_3,
        c1_t2,
        c2_t2,
        c3_t2,
        c4_t2,
        c5_t2,
        c6_t2,
        c7_t2,
        c8_t2,
        c9_t2,
        c10_t2,
        c11_t2,
        c12_t2,
        c13_t2,
        c14_t2,
        c15_t2,
    )

    return min_dist, closest_1, closest_2, closest_coords1, closest_coords2


@jax.jit
def point_segment_distance(
    p: jax.Array, a: jax.Array, b: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Distance from point p to segment a-b."""
    ab = system.domain._displacement(b, a, system)
    ap = system.domain._displacement(p, a, system)
    denom = dot(ab, ab)
    t = dot(ap, ab) / jnp.where(denom != 0.0, denom, 1.0)
    t = jnp.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    coords = jnp.stack([1.0 - t, t], axis=-1)
    return norm(system.domain._displacement(p, closest, system)), closest, coords


@jax.jit
def get_facet_indices(
    idx: jax.Array, state: "State"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Find the indices of the spheres belonging to the same facet clump.
    Returns (is_facet, indices, is_primary)."""

    def single(idx_val: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        is_facet = state.facet_id[idx_val] != -1

        indices = state.facet_vertices[idx_val]
        indices = jnp.where(is_facet, indices, idx_val)

        is_primary = idx_val == jnp.min(indices)

        return is_facet, indices, is_primary

    if jnp.ndim(idx) == 0:
        return single(idx)
    else:
        return jax.vmap(single)(idx)
