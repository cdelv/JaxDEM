# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Facet contact force model."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..utils.linalg import dot, norm, unit

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.custom_jvp
def _safe_ratio(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Divide without squaring a dimensional tiny denominator in reverse AD."""
    safe = jnp.where(denominator != 0.0, denominator, 1.0)
    return numerator / safe


@_safe_ratio.defjvp
def _safe_ratio_jvp(
    primals: tuple[jax.Array, jax.Array], tangents: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    numerator, denominator = primals
    numerator_dot, denominator_dot = tangents
    value = jax.lax.optimization_barrier(  # type: ignore[no-untyped-call]
        _safe_ratio(numerator, denominator)
    )
    safe = jnp.where(denominator != 0.0, denominator, 1.0)
    denominator_dot = jnp.where(denominator != 0.0, denominator_dot, 0.0)
    return value, (numerator_dot - value * denominator_dot) / safe


def _consistent_triangle_bc(ab: jax.Array, ac: jax.Array, bc: jax.Array) -> jax.Array:
    """Retain short-edge digits within the triangle's A-anchored image."""
    anchored_bc = ac - ab
    epsilon = jnp.finfo(ab.dtype).eps  # type: ignore[no-untyped-call]
    tolerance = 8.0 * epsilon * jnp.maximum(jnp.abs(ab), jnp.abs(ac))
    same_image = jnp.all(jnp.abs(bc - anchored_bc) <= tolerance, axis=-1)
    return jnp.where(same_image[..., None], bc, anchored_bc)


def _triangle_cross(ab: jax.Array, ac: jax.Array, bc: jax.Array) -> jax.Array:
    """Compute the oriented area using a short boundary when possible."""
    short_bc = jnp.max(jnp.abs(bc), axis=-1) < jnp.minimum(
        jnp.max(jnp.abs(ab), axis=-1), jnp.max(jnp.abs(ac), axis=-1)
    )
    return jnp.where(short_bc[..., None], jnp.cross(ab, bc), jnp.cross(ab, ac))


@partial(jax.jit, static_argnames=("face", "edge", "vertex"))
def point_triangle_distance(
    p: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    system: "System",
    *,
    face: bool = True,
    edge: bool = True,
    vertex: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return point-triangle separation, closest point and barycentric weights.

    Static feature switches omit contacts whose closest point lies in a
    disabled Voronoi region; they do not select a farther feature. Face and
    edge interiors exclude their boundaries. Collapsed triangles reduce to
    their segments or vertices. Rejected queries return infinite distance
    and finite placeholder points and coordinates.
    """
    return _point_triangle_feature_distance(
        p, a, b, c, system, face=face, edge=edge, vertex=vertex
    )


def _triangle_face_coordinates(
    ab: jax.Array, ac: jax.Array, bc: jax.Array, ap: jax.Array, *, project: bool
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project onto the plane without subtracting nearly equal Gram products."""
    if ab.shape[-1] == 3:
        normal = _triangle_cross(ab, ac, bc)
        scale = jax.lax.stop_gradient(jnp.max(jnp.abs(normal), axis=-1))
        safe_scale = jnp.where(scale > 0.0, scale, 1.0)[..., None]
        scaled_normal = normal / safe_scale
        determinant = dot(scaled_normal, scaled_normal)
        denominator = jnp.where(determinant != 0.0, determinant, 1.0)
        v = dot(jnp.cross(ap, ac) / safe_scale, scaled_normal) / denominator
        w = dot(jnp.cross(ab, ap) / safe_scale, scaled_normal) / denominator
        # Resolve the long direction with a dot product. Independent cross
        # ratios can otherwise disagree along that direction on thin faces.
        ab2, ac2, ab_ac = dot(ab, ab), dot(ac, ac), dot(ab, ac)
        v_long = _safe_ratio(dot(ap, ab) - w * ab_ac, ab2)
        w_long = _safe_ratio(dot(ap, ac) - v * ab_ac, ac2)
        long_ab = ab2 >= ac2
        v, w = jnp.where(long_ab, v_long, v), jnp.where(long_ab, w, w_long)
        projection = (
            ap - scaled_normal * (dot(ap, scaled_normal) / denominator)[..., None]
            if project
            else ap
        )
    else:
        determinant = ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]
        v = _safe_ratio(ap[..., 0] * ac[..., 1] - ap[..., 1] * ac[..., 0], determinant)
        w = _safe_ratio(ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0], determinant)
        projection = ap
    return 1.0 - v - w, v, w, determinant != 0.0, projection


def _point_triangle_feature_distance(
    p: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    system: System,
    *,
    face: bool,
    edge: bool,
    vertex: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Specialize stable Voronoi-region queries to the enabled features."""
    dtype = jnp.result_type(p, a, b, c, 1.0)
    p, a, b, c = jnp.broadcast_arrays(
        *[jnp.asarray(x, dtype=dtype) for x in (p, a, b, c)]
    )
    placeholder = jnp.broadcast_to(
        jnp.array([1.0, 0.0, 0.0], dtype=dtype), a.shape[:-1] + (3,)
    )
    if not (face or edge or vertex):
        return jnp.full(a.shape[:-1], jnp.inf, dtype=dtype), a, placeholder

    ab = system.domain._displacement(b, a, system)
    ac = system.domain._displacement(c, a, system)
    ap = system.domain._displacement(p, a, system)
    bc = system.domain._displacement(c, b, system)
    bc = _consistent_triangle_bc(ab, ac, bc)
    zeros = jnp.zeros(a.shape[:-1], dtype=dtype)
    ones = jnp.ones_like(zeros)
    closest, coords = a, placeholder
    enabled = jnp.zeros(a.shape[:-1], dtype=bool)

    if face or edge:
        u, v, w, has_face, projection = _triangle_face_coordinates(
            ab, ac, bc, ap, project=face
        )
        if face:
            is_face = has_face & (u > 0.0) & (v > 0.0) & (w > 0.0)
            vf, wf = jnp.where(is_face, v, 0.0), jnp.where(is_face, w, 0.0)
            closest = a + jnp.where(is_face[..., None], projection, 0.0)
            coords = jnp.stack([1.0 - vf - wf, vf, wf], axis=-1)
            enabled = is_face

    if edge or vertex:
        bp, cp = ap - ab, ap - ac
        ab_ap, ac_ap = dot(ab, ap), dot(ac, ap)
        ab_bp, bc_bp = dot(ab, bp), dot(bc, bp)
        ac_cp, bc_cp = dot(ac, cp), dot(bc, cp)
        is_a = (ab_ap <= 0.0) & (ac_ap <= 0.0)
        is_b = (ab_bp >= 0.0) & (bc_bp <= 0.0)
        is_c = (ac_cp >= 0.0) & (bc_cp >= 0.0)
        is_vertex = is_a | is_b | is_c

        if edge:
            # Barycentric signs identify the outside of each supporting edge.
            # A zero-area triangle has only boundary regions. Direct BC dot
            # products avoid cancellation when the two A-edges nearly agree.
            u = jnp.where(has_face, u, 0.0)
            v = jnp.where(has_face, v, 0.0)
            w = jnp.where(has_face, w, 0.0)
            is_ab = ~is_vertex & (w <= 0.0) & (ab_ap > 0.0) & (ab_bp < 0.0)
            is_ac = ~is_vertex & (v <= 0.0) & (ac_ap > 0.0) & (ac_cp < 0.0)
            is_bc = ~is_vertex & (u <= 0.0) & (bc_bp > 0.0) & (bc_cp < 0.0)
            denom_ab, denom_ac, denom_bc = dot(ab, ab), dot(ac, ac), dot(bc, bc)
            tab = _safe_ratio(jnp.where(is_ab, ab_ap, 0.0), denom_ab)
            tac = _safe_ratio(jnp.where(is_ac, ac_ap, 0.0), denom_ac)
            tbc = _safe_ratio(jnp.where(is_bc, bc_bp, 0.0), denom_bc)
            closest = jnp.where(
                is_ab[..., None],
                a + tab[..., None] * ab,
                jnp.where(
                    is_ac[..., None],
                    a + tac[..., None] * ac,
                    jnp.where(is_bc[..., None], b + tbc[..., None] * bc, closest),
                ),
            )
            coords = jnp.where(
                is_ab[..., None],
                jnp.stack([ones - tab, tab, zeros], axis=-1),
                jnp.where(
                    is_ac[..., None],
                    jnp.stack([ones - tac, zeros, tac], axis=-1),
                    jnp.where(
                        is_bc[..., None],
                        jnp.stack([zeros, ones - tbc, tbc], axis=-1),
                        coords,
                    ),
                ),
            )
            enabled = enabled | is_ab | is_ac | is_bc

        if vertex:
            closest = jnp.where(
                is_a[..., None],
                a,
                jnp.where(is_b[..., None], b, jnp.where(is_c[..., None], c, closest)),
            )
            coords = jnp.where(
                is_a[..., None],
                placeholder,
                jnp.where(
                    is_b[..., None],
                    jnp.stack([zeros, ones, zeros], axis=-1),
                    jnp.where(
                        is_c[..., None],
                        jnp.stack([zeros, zeros, ones], axis=-1),
                        coords,
                    ),
                ),
            )
            enabled = enabled | is_a | is_b | is_c

    closest = jnp.where(enabled[..., None], closest, a)
    coords = jnp.where(enabled[..., None], coords, placeholder)
    distance = norm(system.domain._displacement(p, closest, system))
    return jnp.where(enabled, distance, jnp.inf), closest, coords


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

    # Gram products cancel for nearly parallel edges, even when their closest
    # points are well inside both segments. Cross products retain the small
    # angle. Scaling avoids squaring a tiny or very large cross product.
    if d1.shape[-1] == 3:
        normal = jnp.cross(d1, d2)
        scale = jax.lax.stop_gradient(jnp.max(jnp.abs(normal), axis=-1))
        safe_scale = jnp.where(scale > 0.0, scale, 1.0)[..., None]
        scaled_normal = normal / safe_scale
        denom = dot(scaled_normal, scaled_normal)
        numerator = dot(jnp.cross(d2, r) / safe_scale, scaled_normal)
    else:
        denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        scale = jax.lax.stop_gradient(jnp.abs(denom))
        safe_scale = jnp.where(scale > 0.0, scale, 1.0)
        denom = denom / safe_scale
        numerator = (d2[..., 0] * r[..., 1] - d2[..., 1] * r[..., 0]) / safe_scale
    safe_denom = jnp.where(denom != 0.0, denom, 1.0)
    s_val = numerator / safe_denom

    s = jnp.where(denom != 0.0, jnp.clip(s_val, 0.0, 1.0), 0.0)

    t = _safe_ratio(b * s + f, e)

    s = jnp.where(
        t < 0.0,
        jnp.clip(_safe_ratio(-c, a), 0.0, 1.0),
        jnp.where(t > 1.0, jnp.clip(_safe_ratio(b - c, a), 0.0, 1.0), s),
    )

    t = jnp.clip(t, 0.0, 1.0)
    # A collapsed second segment is a point projected onto the first.
    s = jnp.where(e == 0.0, jnp.clip(_safe_ratio(-c, a), 0.0, 1.0), s)

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
    u = _safe_ratio(dot(offset, cross_direction), determinant)
    cross_offset = jnp.cross(offset, ab)
    v = _safe_ratio(dot(direction, cross_offset), determinant)
    t = _safe_ratio(dot(ac, cross_offset), determinant)
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


@partial(jax.jit, static_argnames=("vertex_triangle", "edge_edge", "edge_triangle"))
def triangle_triangle_distance(
    t1_a: jax.Array,
    t1_b: jax.Array,
    t1_c: jax.Array,
    t2_a: jax.Array,
    t2_b: jax.Array,
    t2_c: jax.Array,
    system: System,
    *,
    vertex_triangle: bool = True,
    edge_edge: bool = True,
    edge_triangle: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return triangle separation, closest points and barycentric coordinates.

    Vertex-face and edge-edge candidates cover disjoint and coplanar triangles;
    edge-face piercings also detect interior crossings. Zero-area triangles
    reduce to their edges or vertices. At nonunique contacts, vertex order
    deterministically selects one valid closest point.

    The static switches select computational query families, rather than
    disjoint geometric contact types. ``vertex_triangle`` includes a
    triangle's boundary, ``edge_edge`` includes segment endpoints, and
    ``edge_triangle`` tests piercings only. Disabled families are omitted
    during tracing. With all families disabled, the distance is infinite and
    the other outputs are finite placeholders.
    """
    dtype = jnp.result_type(t1_a, t1_b, t1_c, t2_a, t2_b, t2_c, 1.0)
    vertices = jnp.broadcast_arrays(
        *[jnp.asarray(x, dtype=dtype) for x in (t1_a, t1_b, t1_c, t2_a, t2_b, t2_c)]
    )
    first, second = vertices[:3], vertices[3:]
    basis = jnp.eye(3, dtype=dtype)
    candidates = []

    if vertex_triangle:
        for i in range(3):
            distance, point, coords = point_triangle_distance(first[i], *second, system)
            candidates.append(
                (
                    distance,
                    first[i],
                    point,
                    jnp.broadcast_to(basis[i], point.shape),
                    coords,
                )
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

    if edge_edge:
        for i in range(3):
            for j in range(3):
                distance, point_a, point_b, wa, wb = segment_segment_distance(
                    first[i], first[(i + 1) % 3], second[j], second[(j + 1) % 3], system
                )
                coords_a = wa[..., :1] * basis[i] + wa[..., 1:] * basis[(i + 1) % 3]
                coords_b = wb[..., :1] * basis[j] + wb[..., 1:] * basis[(j + 1) % 3]
                candidates.append((distance, point_a, point_b, coords_a, coords_b))

    if edge_triangle:
        for i in range(3):
            hit, point, edge, face = _segment_triangle_intersection(
                first[i], first[(i + 1) % 3], *second, system
            )
            coords = edge[..., :1] * basis[i] + edge[..., 1:] * basis[(i + 1) % 3]
            candidates.append(
                (jnp.where(hit, 0.0, jnp.inf), point, point, coords, face)
            )
            hit, point, edge, face = _segment_triangle_intersection(
                second[i], second[(i + 1) % 3], *first, system
            )
            coords = edge[..., :1] * basis[i] + edge[..., 1:] * basis[(i + 1) % 3]
            candidates.append(
                (jnp.where(hit, 0.0, jnp.inf), point, point, face, coords)
            )

    if not candidates:
        coords = jnp.broadcast_to(basis[0], first[0].shape)
        return (
            jnp.full(first[0].shape[:-1], jnp.inf, dtype=dtype),
            first[0],
            second[0],
            coords,
            coords,
        )

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
        normal = jnp.stack([-edge[..., 1], edge[..., 0]], axis=-1)
    else:
        ac = system.domain._displacement(c, a, system)
        bc = _consistent_triangle_bc(
            edge, ac, system.domain._displacement(c, b, system)
        )
        normal = _triangle_cross(edge, ac, bc)
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(normal), axis=-1))
    return unit(normal / jnp.where(scale > 0.0, scale, 1.0)[..., None])


@partial(jax.jit, static_argnames=("edge", "vertex"))
def point_segment_distance(
    p: jax.Array,
    a: jax.Array,
    b: jax.Array,
    system: "System",
    *,
    edge: bool = True,
    vertex: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the distance from point ``p`` to the segment ``a``-``b``.

    Also return the closest point on the segment and its interpolation
    weights. ``edge`` controls the open segment interior and ``vertex`` its
    endpoints, including collapsed segments. A disabled closest feature
    returns infinite distance and finite placeholders without substituting
    another feature. The switches are static and specialize the query.
    """
    if not (edge and vertex):
        dtype = jnp.result_type(p, a, b, 1.0)
        p, a, b = jnp.broadcast_arrays(
            *[jnp.asarray(x, dtype=dtype) for x in (p, a, b)]
        )
        if not (edge or vertex):
            coords = jnp.broadcast_to(
                jnp.array([1.0, 0.0], dtype=dtype), a.shape[:-1] + (2,)
            )
            return jnp.full(a.shape[:-1], jnp.inf, dtype=dtype), a, coords
        ab = system.domain._displacement(b, a, system)
        ap = system.domain._displacement(p, a, system)
        denominator, numerator = dot(ab, ab), dot(ap, ab)
        if edge:
            t = _safe_ratio(numerator, denominator)
            enabled = (denominator > 0.0) & (t > 0.0) & (t < 1.0)
            t = jnp.where(enabled, t, 0.0)
        else:
            enabled = (numerator <= 0.0) | (numerator >= denominator)
            t = jnp.where((numerator >= denominator) & (denominator > 0.0), 1.0, 0.0)
        closest = a + t[..., None] * ab
        coords = jnp.stack([1.0 - t, t], axis=-1)
        distance = norm(system.domain._displacement(p, closest, system))
        return jnp.where(enabled, distance, jnp.inf), closest, coords

    ab = system.domain._displacement(b, a, system)
    ap = system.domain._displacement(p, a, system)
    denom = dot(ab, ab)
    t = _safe_ratio(dot(ap, ab), denom)
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

    Recompute the union of all incident flexible facets from their vertices.
    ``_rad`` remains a conservative user-supplied lower bound on search extent,
    not thickness. Shared nodes can have facets only in the extra slots.
    """
    pos = state.pos
    indices = jnp.maximum(state.facet_vertices, 0)
    delta = system.domain._displacement(pos[indices], pos[:, None, :], system)
    extent = jnp.max(norm(delta), axis=-1)
    thickness = jnp.max(state.rad[indices], axis=-1)
    bound = jnp.where(state.facet_id >= 0, extent + thickness, state.rad)
    if state.facet_ids_extra.shape[-1]:
        extra_indices = jnp.maximum(state.facet_vertices_extra, 0)
        extra_delta = system.domain._displacement(
            pos[extra_indices], pos[:, None, None, :], system
        )
        extra_extent = jnp.max(norm(extra_delta), axis=-1)
        extra_thickness = jnp.max(state.rad[extra_indices], axis=-1)
        extra_bound = jnp.where(
            state.facet_ids_extra >= 0, extra_extent + extra_thickness, 0.0
        )
        bound = jnp.maximum(bound, jnp.max(extra_bound, axis=-1))
    return jnp.maximum(state._rad, bound)
