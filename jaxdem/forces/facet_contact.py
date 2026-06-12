# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Facet contact force model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..utils.linalg import cross, dot, norm, unit, unit_and_norm
from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
def point_triangle_distance(
    p: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Distance from point p to triangle a, b, c."""
    ab = system.domain.displacement(b, a, system)
    ac = system.domain.displacement(c, a, system)
    ap = system.domain.displacement(p, a, system)

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)

    is_a = (d1 <= 0.0) & (d2 <= 0.0)

    bp = system.domain.displacement(p, b, system)
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)

    is_b = (d3 >= 0.0) & (d4 <= d3)

    vc = d1 * d4 - d3 * d2

    v_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denom_ab = d1 - d3
    v_ab_val = d1 / jnp.where(denom_ab != 0.0, denom_ab, 1.0)

    cp = system.domain.displacement(p, c, system)
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    is_c = (d6 >= 0.0) & (d5 <= d6)

    vb = d5 * d2 - d1 * d6

    v_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denom_ac = d2 - d6
    v_ac_val = d2 / jnp.where(denom_ac != 0.0, denom_ac, 1.0)

    va = d3 * d6 - d5 * d4

    v_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    bc = system.domain.displacement(c, b, system)
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

    return norm(system.domain.displacement(p, closest, system)), closest, coords


@jax.jit
def segment_segment_distance(
    p1: jax.Array, q1: jax.Array, p2: jax.Array, q2: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Distance between two line segments p1q1 and p2q2."""
    d1 = system.domain.displacement(q1, p1, system)
    d2 = system.domain.displacement(q2, p2, system)
    r = system.domain.displacement(p1, p2, system)
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

    return norm(system.domain.displacement(c1, c2, system)), c1, c2, coords1, coords2


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

    distances = jnp.stack(
        [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15], axis=-1
    )

    # Collect closest points on T1
    p1 = jnp.stack(
        [
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
        ],
        axis=-2,
    )
    # Collect closest points on T2
    p2 = jnp.stack(
        [
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
        ],
        axis=-2,
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

    coords1_stack = jnp.stack(
        [
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
        ],
        axis=-2,
    )
    coords2_stack = jnp.stack(
        [
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
        ],
        axis=-2,
    )

    min_idx = jnp.argmin(distances, axis=-1)
    min_dist = jnp.min(distances, axis=-1)

    mask = jax.nn.one_hot(min_idx, 15, dtype=p1.dtype)
    closest_1 = jnp.sum(p1 * mask[..., None], axis=-2)
    closest_2 = jnp.sum(p2 * mask[..., None], axis=-2)

    closest_coords1 = jnp.sum(coords1_stack * mask[..., None], axis=-2)
    closest_coords2 = jnp.sum(coords2_stack * mask[..., None], axis=-2)

    return min_dist, closest_1, closest_2, closest_coords1, closest_coords2


@jax.jit
def point_segment_distance(
    p: jax.Array, a: jax.Array, b: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Distance from point p to segment a-b."""
    ab = system.domain.displacement(b, a, system)
    ap = system.domain.displacement(p, a, system)
    denom = dot(ab, ab)
    t = dot(ap, ab) / jnp.where(denom != 0.0, denom, 1.0)
    t = jnp.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    coords = jnp.stack([1.0 - t, t], axis=-1)
    return norm(system.domain.displacement(p, closest, system)), closest, coords


@jax.jit
def get_facet_indices(
    idx: jax.Array, state: "State"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Find the indices of the spheres belonging to the same facet clump.
    Returns (is_facet, indices, is_primary)."""
    # O(N) scatter (unique_id is a permutation), much cheaper than argsort
    # since this runs inside the collider's per-pair loop.
    inv_perm = state.unique_id.at[state.unique_id].set(
        jax.lax.iota(size=state.unique_id.shape[-1], dtype=state.unique_id.dtype)
    )

    def single(idx_val: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        is_facet = state.facet_id[idx_val] != -1
        uid = state.unique_id[idx_val]

        # original indices (unique_ids) of the facet vertices
        orig_indices = state.facet_vertices[idx_val]
        orig_indices = jnp.where(is_facet, orig_indices, uid)

        # map original indices to current sorted indices
        sorted_indices = inv_perm[orig_indices]

        is_primary = uid == jnp.min(orig_indices)

        return is_facet, sorted_indices, is_primary

    if jnp.ndim(idx) == 0:
        return single(idx)
    else:
        return jax.vmap(single)(idx)


@partial(jax.jit, inline=True)
def _sphere_facet_pair(
    i: jax.Array, j: jax.Array, pos: jax.Array, state: "State", system: "System"
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Shared sphere-facet pair kernel used by both ``force`` and ``energy``.

    Returns ``(k, delta, is_contact, w, n, c_1, thick_i, is_rigid)`` where
    ``delta`` is the (masked) overlap, ``w`` the partition-of-unity weight,
    ``n`` the contact normal and ``c_1`` the contact point on body ``i``.
    Quantities unused by a caller are removed by XLA dead-code elimination.
    """
    dim = pos.shape[-1]

    is_facet_i, idxs_i, is_primary_i = get_facet_indices(i, state)
    is_facet_j, idxs_j, is_primary_j = get_facet_indices(j, state)

    idx0_i = jnp.where(is_facet_i, idxs_i[..., 0], i)
    idx0_j = jnp.where(is_facet_j, idxs_j[..., 0], j)

    is_rigid_i = jnp.where(
        is_facet_i,
        jnp.all(state.clump_id[idxs_i] == state.clump_id[idxs_i[..., 0:1]], axis=-1),
        True,
    )
    is_rigid_j = jnp.where(
        is_facet_j,
        jnp.all(state.clump_id[idxs_j] == state.clump_id[idxs_j[..., 0:1]], axis=-1),
        True,
    )
    is_rigid = is_rigid_i & is_rigid_j

    compute_interaction = jnp.where(
        is_rigid,
        jnp.where(is_facet_i, is_primary_i, True)
        & jnp.where(is_facet_j, is_primary_j, True),
        True,
    ) & (is_facet_i ^ is_facet_j)

    mi, mj = state.mat_id[idx0_i], state.mat_id[idx0_j]
    k = system.mat_table.young_eff[mi, mj]

    v_a = jnp.where(is_facet_i[..., None], pos[idxs_i[..., 0]], pos[idxs_j[..., 0]])
    v_b = jnp.where(is_facet_i[..., None], pos[idxs_i[..., 1]], pos[idxs_j[..., 1]])
    if dim == 3:
        v_c = jnp.where(is_facet_i[..., None], pos[idxs_i[..., 2]], pos[idxs_j[..., 2]])
    else:
        v_c = None

    p = jnp.where(is_facet_i[..., None], pos[idx0_j], pos[idx0_i])

    if dim == 3:
        d_sf, c_f, coords_f = point_triangle_distance(p, v_a, v_b, v_c, system)
    else:
        d_sf, c_f, coords_f = point_segment_distance(p, v_a, v_b, system)

    c_1 = jnp.where(is_facet_i[..., None], c_f, p)
    c_2 = jnp.where(is_facet_j[..., None], c_f, p)

    thick_i = state.rad[idx0_i]
    thick_j = state.rad[idx0_j]

    rij = system.domain.displacement(c_1, c_2, system)
    n, r = unit_and_norm(rij)

    fallback_rij = system.domain.displacement(
        state.pos_c[idx0_i], state.pos_c[idx0_j], system
    )
    n = jnp.where(r[..., 0:1] > 1e-7, n, unit(fallback_rij))

    delta = thick_i + thick_j - d_sf
    is_contact = (delta > 0) & compute_interaction
    delta *= is_contact

    idxs_facet = jnp.where(is_facet_i[..., None], idxs_i, idxs_j)
    particle_facet = jnp.where(is_facet_i, i, j)
    v_idx = jnp.argmax(idxs_facet == particle_facet[..., None], axis=-1)
    w_vertex = jnp.sum(
        coords_f * jax.nn.one_hot(v_idx, dim, dtype=coords_f.dtype), axis=-1
    )

    w = jnp.where(is_rigid, 1.0, w_vertex)

    return k, delta, is_contact, w, n, c_1, thick_i, is_rigid


@ForceModel.register("sphere_facet_spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SphereFacetSpringForce(ForceModel):
    r"""Linear spring contact between spheres and facets.

    .. warning::
        Facet contacts are detected through the facet's *vertex spheres*
        (in particular the facet's primary vertex). The collider's neighbor
        cutoff must therefore be at least the facet circumradius (the largest
        vertex-to-contact-point distance) plus the contact thickness
        (``state.rad``). If the primary vertex lies outside the cutoff while
        the contact point is in range, the contact is silently missed or
        applied asymmetrically. Cell-list based colliders must be configured
        with ``cutoff >= max facet circumradius + thickness``.

    The contact thickness is taken from the facet vertices' ``state.rad``
    (set via ``State.add_facet(thickness=...)``); the force model itself
    has no thickness parameter.
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SphereFacetSpringForce.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        k, delta, is_contact, w, n, c_1, thick_i, is_rigid = _sphere_facet_pair(
            i, j, pos, state, system
        )

        fn_mag = jnp.maximum(0.0, k * delta)
        f_total = fn_mag[..., None] * n * is_contact[..., None] * w[..., None]

        r_ci = (
            system.domain.displacement(c_1, state.pos[i], system)
            - thick_i[..., None] * n
        )

        t_total = cross(r_ci, f_total)

        is_rigid_mask = is_rigid
        while is_rigid_mask.ndim < t_total.ndim:
            is_rigid_mask = is_rigid_mask[..., None]
        t_total = t_total * is_rigid_mask

        return f_total, t_total

    @staticmethod
    @partial(jax.jit, inline=True)
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        k, delta, is_contact, w, n, c_1, thick_i, is_rigid = _sphere_facet_pair(
            i, j, pos, state, system
        )
        return 0.5 * k * delta * delta * w


@partial(jax.jit, inline=True)
def _facet_facet_pair(
    i: jax.Array, j: jax.Array, pos: jax.Array, state: "State", system: "System"
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Shared facet-facet pair kernel used by both ``force`` and ``energy``.

    Returns ``(k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid)`` where
    ``delta`` is the (masked) overlap, ``w`` the partition-of-unity weight,
    ``n`` the contact normal and ``c_ff_1`` the contact point on facet ``i``.
    Quantities unused by a caller are removed by XLA dead-code elimination.
    """
    dim = pos.shape[-1]
    i_arr = jnp.asarray(i)
    j_arr = jnp.asarray(j)

    is_facet_i, idxs_i, is_primary_i = get_facet_indices(i, state)
    is_facet_j, idxs_j, is_primary_j = get_facet_indices(j, state)

    idx0_i = jnp.where(is_facet_i, idxs_i[..., 0], i)
    idx0_j = jnp.where(is_facet_j, idxs_j[..., 0], j)

    is_rigid_i = jnp.where(
        is_facet_i,
        jnp.all(state.clump_id[idxs_i] == state.clump_id[idxs_i[..., 0:1]], axis=-1),
        True,
    )
    is_rigid_j = jnp.where(
        is_facet_j,
        jnp.all(state.clump_id[idxs_j] == state.clump_id[idxs_j[..., 0:1]], axis=-1),
        True,
    )
    is_rigid = is_rigid_i & is_rigid_j

    compute_interaction = (
        jnp.where(
            is_rigid,
            is_primary_i & is_primary_j,
            True,
        )
        & is_facet_i
        & is_facet_j
        & (idx0_i != idx0_j)
    )

    mi, mj = state.mat_id[idx0_i], state.mat_id[idx0_j]
    k = system.mat_table.young_eff[mi, mj]

    t1_a = pos[idxs_i[..., 0]]
    t1_b = pos[idxs_i[..., 1]]
    t2_a = pos[idxs_j[..., 0]]
    t2_b = pos[idxs_j[..., 1]]

    swap = idx0_i > idx0_j

    tA_a = jnp.where(swap[..., None], t2_a, t1_a)
    tA_b = jnp.where(swap[..., None], t2_b, t1_b)
    tB_a = jnp.where(swap[..., None], t1_a, t2_a)
    tB_b = jnp.where(swap[..., None], t1_b, t2_b)

    if dim == 3:
        t1_c = pos[idxs_i[..., 2]]
        t2_c = pos[idxs_j[..., 2]]
        tA_c = jnp.where(swap[..., None], t2_c, t1_c)
        tB_c = jnp.where(swap[..., None], t1_c, t2_c)
        tA_a, tA_b, tA_c, tB_a, tB_b, tB_c = jnp.broadcast_arrays(
            tA_a, tA_b, tA_c, tB_a, tB_b, tB_c
        )
        d_ff, c_A, c_B, coords_A, coords_B = triangle_triangle_distance(
            tA_a, tA_b, tA_c, tB_a, tB_b, tB_c, system
        )
    else:
        tA_a, tA_b, tB_a, tB_b = jnp.broadcast_arrays(tA_a, tA_b, tB_a, tB_b)
        d_ff, c_A, c_B, coords_A, coords_B = segment_segment_distance(
            tA_a, tA_b, tB_a, tB_b, system
        )

    c_ff_1 = jnp.where(swap[..., None], c_B, c_A)
    c_ff_2 = jnp.where(swap[..., None], c_A, c_B)
    coords_1 = jnp.where(swap[..., None], coords_B, coords_A)
    coords_2 = jnp.where(swap[..., None], coords_A, coords_B)

    thick_i = state.rad[idx0_i]
    thick_j = state.rad[idx0_j]

    delta = thick_i + thick_j - d_ff
    is_contact = (delta > 0) & compute_interaction
    delta *= is_contact

    rij = system.domain.displacement(c_ff_1, c_ff_2, system)
    n, r = unit_and_norm(rij)

    fallback_rij = system.domain.displacement(
        state.pos_c[idx0_i], state.pos_c[idx0_j], system
    )
    n = jnp.where(r[..., 0:1] > 1e-7, n, unit(fallback_rij))

    v_idx_i = jnp.argmax(idxs_i == i_arr[..., None], axis=-1)
    w_vertex_i = jnp.sum(
        coords_1 * jax.nn.one_hot(v_idx_i, dim, dtype=coords_1.dtype), axis=-1
    )

    v_idx_j = jnp.argmax(idxs_j == j_arr[..., None], axis=-1)
    w_vertex_j = jnp.sum(
        coords_2 * jax.nn.one_hot(v_idx_j, dim, dtype=coords_2.dtype), axis=-1
    )

    w = jnp.where(is_rigid, 1.0, w_vertex_i * w_vertex_j)

    return k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid


@ForceModel.register("facet_facet_spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FacetFacetSpringForce(ForceModel):
    r"""Linear spring contact between facets.

    .. warning::
        Facet contacts are detected through the facets' *vertex spheres*
        (in particular each facet's primary vertex). The collider's neighbor
        cutoff must therefore be at least the facet circumradius (the largest
        vertex-to-contact-point distance) plus the contact thickness
        (``state.rad``). If a primary vertex lies outside the cutoff while
        the contact point is in range, the contact is silently missed or
        applied asymmetrically. Cell-list based colliders must be configured
        with ``cutoff >= max facet circumradius + thickness``.

    The contact thickness is taken from the facet vertices' ``state.rad``
    (set via ``State.add_facet(thickness=...)``); the force model itself
    has no thickness parameter.
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="FacetFacetSpringForce.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid = _facet_facet_pair(
            i, j, pos, state, system
        )

        fn_mag = jnp.maximum(0.0, k * delta)
        f_total = fn_mag[..., None] * n * is_contact[..., None] * w[..., None]

        r_ci = (
            system.domain.displacement(c_ff_1, state.pos[i], system)
            - thick_i[..., None] * n
        )

        t_total = cross(r_ci, f_total)

        is_rigid_mask = is_rigid
        while is_rigid_mask.ndim < t_total.ndim:
            is_rigid_mask = is_rigid_mask[..., None]
        t_total = t_total * is_rigid_mask

        return f_total, t_total

    @staticmethod
    @partial(jax.jit, inline=True)
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid = _facet_facet_pair(
            i, j, pos, state, system
        )
        return 0.5 * k * delta * delta * w
