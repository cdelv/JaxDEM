# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Facet contact force model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..utils.linalg import dot, norm, unit

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
def point_triangle_distance(
    p: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the distance from point ``p`` to the triangle with vertices ``a``, ``b``, ``c``.

    Also return the closest point on the triangle and its barycentric
    coordinates.
    """
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

    v_ab = (vc <= 0.0) * (d1 >= 0.0) * (d3 <= 0.0) * (d1 - d3 > 0.0)
    denom_ab = d1 - d3
    v_ab_val = d1 / jnp.where(denom_ab != 0.0, denom_ab, 1.0)

    cp = ap - ac
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    is_c = (d6 >= 0.0) * (d5 <= d6)

    vb = d5 * d2 - d1 * d6

    v_ac = (vb <= 0.0) * (d2 >= 0.0) * (d6 <= 0.0) * (d2 - d6 > 0.0)
    denom_ac = d2 - d6
    v_ac_val = d2 / jnp.where(denom_ac != 0.0, denom_ac, 1.0)

    va = d3 * d6 - d5 * d4

    v_bc = (
        (va <= 0.0)
        * ((d4 - d3) >= 0.0)
        * ((d5 - d6) >= 0.0)
        * ((d4 - d3) + (d5 - d6) > 0.0)
    )
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
    """Compute the shortest distance between the segments ``p1``-``q1`` and ``p2``-``q2``.

    Also return the closest point on each segment and its interpolation
    weights. Collapsed segments reduce to point-segment or point-point queries.
    """
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
    # A collapsed second segment is a point projected onto the first.
    s = jnp.where(e == 0.0, jnp.clip(-c / safe_a, 0.0, 1.0), s)

    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]

    coords1 = jnp.stack([1.0 - s, s], axis=-1)
    coords2 = jnp.stack([1.0 - t, t], axis=-1)

    return norm(system.domain._displacement(c1, c2, system)), c1, c2, coords1, coords2


@jax.jit
def _segment_triangle_intersection(
    p: jax.Array, q: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, system: System
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return an edge-face piercing and its segment/triangle coordinates."""
    direction = system.domain._displacement(q, p, system)
    ab = system.domain._displacement(b, a, system)
    ac = system.domain._displacement(c, a, system)
    offset = system.domain._displacement(p, a, system)
    cross_direction = jnp.cross(direction, ac)
    determinant = dot(ab, cross_direction)
    inverse = 1.0 / jnp.where(determinant != 0.0, determinant, 1.0)
    u = dot(offset, cross_direction) * inverse
    cross_offset = jnp.cross(offset, ab)
    v = dot(direction, cross_offset) * inverse
    t = dot(ac, cross_offset) * inverse
    hit = (
        (determinant != 0.0)
        & (t >= 0.0)
        & (t <= 1.0)
        & (u >= 0.0)
        & (v >= 0.0)
        & (u + v <= 1.0)
    )
    point = p + jnp.clip(t, 0.0, 1.0)[..., None] * direction
    return (
        hit,
        point,
        jnp.stack([1.0 - t, t], axis=-1),
        jnp.stack([1.0 - u - v, u, v], axis=-1),
    )


@jax.jit
def triangle_triangle_distance(
    t1_a: jax.Array,
    t1_b: jax.Array,
    t1_c: jax.Array,
    t2_a: jax.Array,
    t2_b: jax.Array,
    t2_c: jax.Array,
    system: System,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return triangle separation, closest points and barycentric coordinates.

    Vertex-face and edge-edge candidates cover disjoint and coplanar triangles;
    edge-face piercings also detect interior crossings. Zero-area triangles
    reduce to their edges or vertices. At nonunique contacts, vertex order
    deterministically selects one valid closest point.
    """
    dtype = jnp.result_type(t1_a, t1_b, t1_c, t2_a, t2_b, t2_c, 1.0)
    vertices = jnp.broadcast_arrays(
        *[jnp.asarray(x, dtype=dtype) for x in (t1_a, t1_b, t1_c, t2_a, t2_b, t2_c)]
    )
    first, second = vertices[:3], vertices[3:]
    basis = jnp.eye(3, dtype=dtype)
    candidates = []

    for i in range(3):
        distance, point, coords = point_triangle_distance(first[i], *second, system)
        candidates.append(
            (distance, first[i], point, jnp.broadcast_to(basis[i], point.shape), coords)
        )
        distance, point, coords = point_triangle_distance(second[i], *first, system)
        candidates.append(
            (
                distance,
                point,
                second[i],
                coords,
                jnp.broadcast_to(basis[i], point.shape),
            )
        )

    for i in range(3):
        for j in range(3):
            distance, point_a, point_b, wa, wb = segment_segment_distance(
                first[i], first[(i + 1) % 3], second[j], second[(j + 1) % 3], system
            )
            coords_a = wa[..., :1] * basis[i] + wa[..., 1:] * basis[(i + 1) % 3]
            coords_b = wb[..., :1] * basis[j] + wb[..., 1:] * basis[(j + 1) % 3]
            candidates.append((distance, point_a, point_b, coords_a, coords_b))

    for i in range(3):
        hit, point, edge, face = _segment_triangle_intersection(
            first[i], first[(i + 1) % 3], *second, system
        )
        coords = edge[..., :1] * basis[i] + edge[..., 1:] * basis[(i + 1) % 3]
        candidates.append((jnp.where(hit, 0.0, jnp.inf), point, point, coords, face))
        hit, point, edge, face = _segment_triangle_intersection(
            second[i], second[(i + 1) % 3], *first, system
        )
        coords = edge[..., :1] * basis[i] + edge[..., 1:] * basis[(i + 1) % 3]
        candidates.append((jnp.where(hit, 0.0, jnp.inf), point, point, face, coords))

    distances = jnp.stack([candidate[0] for candidate in candidates], axis=-1)
    index = jnp.argmin(distances, axis=-1)
    distance = jnp.take_along_axis(distances, index[..., None], axis=-1)[..., 0]
    vector_index = jnp.broadcast_to(index[..., None], first[0].shape)
    points_and_coords = tuple(
        jax.lax.select_n(vector_index, *[candidate[i] for candidate in candidates])
        for i in range(1, 5)
    )
    point_a, point_b, coords_a, coords_b = points_and_coords
    return distance, point_a, point_b, coords_a, coords_b


@jax.jit(inline=True)
def facet_normal(
    a: jax.Array, b: jax.Array, c: jax.Array | None, system: System
) -> jax.Array:
    """Return a facet's unit normal, or zero for collapsed geometry."""
    edge = system.domain._displacement(b, a, system)
    if c is None:
        return unit(jnp.stack([-edge[..., 1], edge[..., 0]], axis=-1))
    return unit(jnp.cross(edge, system.domain._displacement(c, a, system)))


@jax.jit
def point_segment_distance(
    p: jax.Array, a: jax.Array, b: jax.Array, system: "System"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the distance from point ``p`` to the segment ``a``-``b``.

    Also return the closest point on the segment and its interpolation
    weights.
    """
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
    """Get the sphere indices of the facet clump that contains ``idx``.

    Returns ``(is_facet, indices, is_primary)``.
    """

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


@jax.jit(inline=True)
def facet_search_radii(state: State, system: System) -> jax.Array:
    """Bound each contact primitive from its current search-key position.

    Recompute flexible facet extents from their vertices; ``_rad`` remains a
    conservative user-supplied lower bound on search extent, not thickness.
    """
    pos = state.pos
    indices = jnp.maximum(state.facet_vertices, 0)
    delta = system.domain._displacement(pos[indices], pos[:, None, :], system)
    extent = jnp.max(norm(delta), axis=-1)
    thickness = jnp.max(state.rad[indices], axis=-1)
    bound = jnp.where(state.facet_id >= 0, extent + thickness, state.rad)
    return jnp.maximum(state._rad, bound)
