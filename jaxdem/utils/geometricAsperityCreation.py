# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Utility functions for creating Geometric Asperity particle states in 2D and 3D.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from typing import Any, Optional, Sequence, Tuple, Union, cast

from .quaternion import Quaternion
from .randomSphereConfiguration import random_sphere_configuration
from .randomizeOrientations import randomize_orientations
from ..state import State
from ..forces.deformable_particle import (
    DeformableParticleContainer,
    angle_between_normals,
)


def duplicate_clump_template(template: State, com_positions: jnp.ndarray) -> State:
    """
    template: a single clump with Ns spheres (template.pos_c same for all spheres, template.clump_ID same for all spheres)
    com_positions: (M, dim) desired clump COM positions
    returns: State with M clumps, total N = M*Ns spheres
    """
    com_positions = jnp.asarray(com_positions, dtype=float)
    M, dim = com_positions.shape
    Ns = template.N
    assert dim == template.dim

    # repeat template leaf shaped (Ns, ...) -> (M*Ns, ...)
    def tile0(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.broadcast_to(x, (M,) + x.shape).reshape(
            (M * x.shape[0],) + x.shape[1:]
        )

    # clump COM per sphere
    pos_c = jnp.repeat(com_positions, repeats=Ns, axis=0)  # (M*Ns, dim)

    # unique clump ids 0..M-1, repeated for each sphere in the clump
    ID = jnp.repeat(jnp.arange(M, dtype=int), repeats=Ns)  # (M*Ns,)

    q = Quaternion(tile0(template.q.w), tile0(template.q.xyz))

    return State(
        pos_c=pos_c,
        pos_p=tile0(template.pos_p),
        vel=tile0(template.vel),
        force=tile0(template.force),
        q=q,
        angVel=tile0(template.angVel),
        torque=tile0(template.torque),
        rad=tile0(template.rad),
        volume=tile0(template.volume),
        mass=tile0(template.mass),
        inertia=tile0(template.inertia),
        clump_ID=ID,
        deformable_ID=jnp.arange(ID.size, dtype=int),
        unique_ID=jnp.arange(ID.size),
        mat_id=tile0(template.mat_id),
        species_id=tile0(template.species_id),
        fixed=tile0(template.fixed),
    )


def _ensure_per_body_params(
    x: float | jax.Array | None, n_bodies: int, name: str
) -> jax.Array | None:
    if x is None:
        return None
    arr = jnp.asarray(x, dtype=float)
    if arr.ndim == 0:
        return jnp.ones((n_bodies,), dtype=float) * arr
    if arr.shape == (n_bodies,):
        return arr
    raise ValueError(f"{name} must be a scalar or shape ({n_bodies},), got {arr.shape}")


def _ensure_single_body_coeff(
    x: float | jax.Array | None, name: str
) -> Optional[jnp.ndarray]:
    """
    DeformableParticleContainer.create expects coefficient arrays of shape (num_bodies,).
    For the single-body builders, accept scalars and coerce to shape (1,).
    """
    if x is None:
        return None
    arr = jnp.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr[None]
    if arr.ndim == 1 and arr.shape == (1,):
        return arr
    raise ValueError(f"{name} must be a scalar or shape (1,), got {arr.shape}")


def _pick_core_index(pts: jnp.ndarray) -> int:
    """Heuristic: point closest to centroid is treated as interior/core node."""
    c = jnp.mean(pts, axis=0)
    d2 = jnp.sum((pts - c) ** 2, axis=1)
    return int(jnp.argmin(d2))


def _order_boundary_2d(pts: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Order boundary indices CCW by polar angle around centroid."""
    bpts = pts[idx]
    c = jnp.mean(bpts, axis=0)
    angles = jnp.arctan2(bpts[:, 1] - c[1], bpts[:, 0] - c[0])
    order = jnp.argsort(angles)
    ordered = idx[order]

    # enforce CCW orientation (positive signed area)
    poly = pts[ordered]
    x, y = poly[:, 0], poly[:, 1]
    area2 = jnp.sum(x * jnp.roll(y, -1) - y * jnp.roll(x, -1))
    ordered = jnp.where(area2 < 0, jnp.flip(ordered, axis=0), ordered)
    return ordered


def _rotate_points_2d(
    pts: jnp.ndarray, theta: jnp.ndarray, center: jnp.ndarray
) -> jnp.ndarray:
    """Rotate 2D points around center by angle theta (radians)."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    r = pts - center
    x = c * r[:, 0] - s * r[:, 1]
    y = s * r[:, 0] + c * r[:, 1]
    return jnp.stack([x, y], axis=1) + center


def _rotate_points_3d_quat(
    pts: jnp.ndarray, q4: jnp.ndarray, center: jnp.ndarray
) -> jnp.ndarray:
    """
    Rotate 3D points around center by quaternion q4 = [w, x, y, z].
    Uses the same formula as `Quaternion.rotate` (vectorized for points).
    """
    r = pts - center
    w = q4[0]
    xyz = q4[1:4]
    # T = xyz x v
    T = jnp.cross(jnp.broadcast_to(xyz, r.shape), r)
    # B = xyz x T
    B = jnp.cross(jnp.broadcast_to(xyz, r.shape), T)
    r_rot = r + 2.0 * (w * T + B)
    return r_rot + center


def _randomize_orientation(pts: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
    """Randomly rotate a single deformable body's node positions about its centroid."""
    dim = pts.shape[-1]
    center = jnp.mean(pts, axis=0)
    if dim == 2:
        theta = jax.random.uniform(key, (), minval=0.0, maxval=2.0 * jnp.pi)
        return _rotate_points_2d(pts, theta, center)
    if dim == 3:
        q4 = jax.random.normal(key, (4,))
        q4 = q4 / jnp.linalg.norm(q4)
        return _rotate_points_3d_quat(pts, q4, center)
    return pts


def _polygon_elements_from_order(order: jnp.ndarray) -> jnp.ndarray:
    """Build (M,2) segments from ordered vertex indices."""
    return jnp.stack([order, jnp.roll(order, -1)], axis=1)


def _bending_adjacency_for_ring(n_elements: int) -> jnp.ndarray:
    """Consecutive element pairs (m, (m+1)%M). Shape (M,2)."""
    m = jnp.arange(n_elements, dtype=int)
    return jnp.stack([m, (m + 1) % n_elements], axis=1)


def _initial_bending_2d(
    vertices: jnp.ndarray, elements: jnp.ndarray, element_adjacency: jnp.ndarray
) -> jnp.ndarray:
    """Compute rest bending angles for 2D segments using segment normals."""
    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    edge = p1 - p0
    length = jnp.linalg.norm(edge, axis=-1)
    normal = jnp.stack([edge[:, 1], -edge[:, 0]], axis=1)
    unit_normal = normal / jnp.where(length[:, None] == 0, 1.0, length[:, None])
    n1 = unit_normal[element_adjacency[:, 0]]
    n2 = unit_normal[element_adjacency[:, 1]]
    return angle_between_normals(n1, n2)


def _initial_bending_3d(
    vertices: jnp.ndarray, faces: jnp.ndarray, face_adjacency: jnp.ndarray
) -> jnp.ndarray:
    """Compute rest bending angles for 3D triangles using face normals."""
    tri = vertices[faces]  # (F,3,3)
    r2 = tri[:, 1] - tri[:, 0]
    r3 = tri[:, 2] - tri[:, 0]
    face_normal = jnp.cross(r2, r3)
    nrm = jnp.linalg.norm(face_normal, axis=-1)
    unit = face_normal / jnp.where(nrm[:, None] == 0, 1.0, nrm[:, None])
    n1 = unit[face_adjacency[:, 0]]
    n2 = unit[face_adjacency[:, 1]]
    return angle_between_normals(n1, n2)


def generate_asperities_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    aspect_ratio: float = 1.0,
    add_core: Optional[bool] = False,
    use_uniform_mesh: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipse)
    num_vertices: int - number of asperities
    aspect_ratio: float - optional aspect ratio of the ellipse
    add_core: bool - optional.  Adds a central core particle if True, otherwise does nothing
    calculations.  "true" physical core added.
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipses
    ____
    returns:
    asperity_positions: jnp.ndarray - (num_vertices + add_core, 2) array of positions of the asperities
    asperity_radii: jnp.ndarray - (num_vertices + add_core,) array of radii of the asperities
    ____
    notes:
    creates a particle composed of a set of surface asperities
    places asperities along either a circle or an ellipse in 2d
    ensures that the outer-most length of the particle is equal to 2 * particle_radius
    adds a core which is useful for covering up large gaps between adjacent asperities
    """
    from shapely.geometry import Point
    from shapely import affinity

    core_radius = particle_radius - asperity_radius
    if asperity_radius > particle_radius:
        print(
            f"Warning: asperity radius exceeds particle radius.  {asperity_radius} > {particle_radius}"
        )
    if aspect_ratio < 1:
        aspect_ratio = 1 / aspect_ratio
    a = core_radius
    b = core_radius / aspect_ratio
    circle = Point(0.0, 0.0).buffer(1.0, quad_segs=1000 * int(num_vertices))
    if use_uniform_mesh and aspect_ratio != 1.0:
        # when making an ellipse, select the points evenly along the outer perimeter
        # this avoids asperities bunching up at the major axis
        ellipse = affinity.scale(circle, xfact=a, yfact=b)
        # distances = jnp.sort(jnp.random.uniform(0, ellipse.length, num_vertices))  # for random case
        distances = jnp.arange(int(num_vertices)) * ellipse.length / num_vertices
        points = [ellipse.boundary.interpolate(d) for d in distances]
        asperity_positions = jnp.array([[p.x, p.y] for p in points])
    else:
        distances = jnp.arange(int(num_vertices)) * circle.length / num_vertices
        points = [circle.boundary.interpolate(d) for d in distances]
        asperity_positions = jnp.array([[p.x * a, p.y * b] for p in points])
    asperity_radii = jnp.ones(int(num_vertices)) * asperity_radius

    if add_core:
        if aspect_ratio == 1.0:
            asperity_positions = jnp.concatenate(
                (asperity_positions, jnp.zeros((1, 2))), axis=0
            )
            asperity_radii = jnp.concatenate(
                (asperity_radii, jnp.array([core_radius])), axis=0
            )
        else:
            print("Warning: ellipse core not yet supported")
    return asperity_positions, asperity_radii


def compute_polygon_properties(
    shape: Any, mass: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Green's theorem: COM, polar inertia, and principal-axis quaternion for a 2D solid."""
    from shapely.geometry.polygon import orient
    import numpy as np

    shape_ccw = orient(shape, sign=1.0)
    coords = np.array(shape_ccw.exterior.coords[:-1])
    x, y = coords[:, 0], coords[:, 1]

    x1, y1 = x, y
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x1 * y2 - x2 * y1

    area = 0.5 * np.sum(cross)
    cx = np.sum((x1 + x2) * cross) / (6.0 * area)
    cy = np.sum((y1 + y2) * cross) / (6.0 * area)

    Ixx_o = np.sum(cross * (y1**2 + y1 * y2 + y2**2)) / 12.0
    Iyy_o = np.sum(cross * (x1**2 + x1 * x2 + x2**2)) / 12.0
    Ixy_o = np.sum(cross * (x1 * y2 + 2 * x1 * y1 + 2 * x2 * y2 + x2 * y1)) / 24.0

    A = abs(area)
    Ixx = Ixx_o - area * cy**2
    Iyy = Iyy_o - area * cx**2
    Ixy = Ixy_o - area * cx * cy

    density = mass / A
    I_polar = (Ixx + Iyy) * density

    C = np.array([[Iyy, Ixy], [Ixy, Ixx]])
    _, eigvecs = np.linalg.eigh(C)
    theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    half = theta / 2.0
    q = jnp.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    pos_c = jnp.array([cx, cy])

    return pos_c, q, jnp.asarray(I_polar, dtype=float), jnp.asarray(A, dtype=float)


def compute_mesh_properties(
    mesh: Any, mass: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact mesh-based: COM, principal inertia (3-vector), and quaternion for a 3D solid."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    density = mass / mesh.volume
    com = jnp.array(mesh.center_mass)

    # mesh.moment_inertia: 3x3 inertia tensor at unit density about COM
    I_tensor = np.array(mesh.moment_inertia) * density
    I_tensor = 0.5 * (I_tensor + I_tensor.T)

    eigvals, eigvecs = np.linalg.eigh(I_tensor)

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1

    rot = Rotation.from_matrix(eigvecs)
    q_xyzw = rot.as_quat()  # scipy: [x, y, z, w]
    q = jnp.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # JaxDEM: [w, x, y, z]

    return com, q, jnp.array(eigvals), jnp.asarray(mesh.volume, dtype=float)


def make_single_particle_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    aspect_ratio: float = 1.0,
    body_type: Optional[str] = "solid",
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = (0.0, 0.0),
    mass: float = 1.0,
    quad_segs: int = 10_000,
) -> State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: float - optional aspect ratios of the ellipsoid
    body_type: str - optional. 'true-solid' (physical core), 'solid' (core used for area and inertia), 'point' (point masses)
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    particle_center: Sequence[float] - optional particle center location
    mass: float - optional mass of the entire particle
    quad_segs: int - optional number of segments used to define the mass
    ____
    returns:
    single_clump_state: State - jaxdem state object containing the single clump particle in 2d
    """

    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    if body_type not in ["true-solid", "solid", "point"]:
        raise ValueError(f"body_type {body_type} not understood")

    if aspect_ratio < 1.0:
        raise ValueError(f"aspect_ratio cannot be less than 1.0")

    if body_type in ["true-solid", "solid"]:
        add_core = True
    else:
        add_core = False

    asperity_positions, asperity_radii = generate_asperities_2d(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        num_vertices=num_vertices,
        aspect_ratio=aspect_ratio,
        add_core=add_core,
        use_uniform_mesh=use_uniform_mesh,
    )

    if body_type == "point":
        n = asperity_positions.shape[0]
        m_i = mass / n
        pos_c = jnp.mean(asperity_positions, axis=0)
        r_sq = jnp.sum((asperity_positions - pos_c) ** 2, axis=-1)
        I_polar = jnp.sum(m_i * r_sq)
        r_prime = asperity_positions - pos_c
        Cov = jnp.einsum("ni,nj->ij", r_prime, r_prime) * m_i
        _, eigvecs = jnp.linalg.eigh(Cov)
        theta = jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        half = theta / 2.0
        q = jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)])
        vol = jnp.sum(jnp.pi * asperity_radii**2)
    else:
        shapes = []
        if body_type == "true-solid":
            if aspect_ratio > 1.0:
                raise ValueError(
                    "Warning: true-solid particle not implemented for 2D ellipses"
                )
            else:
                shapes = [
                    Point(p).buffer(r, quad_segs=quad_segs)
                    for p, r in zip(asperity_positions, asperity_radii)
                ]
        if body_type == "solid":
            if aspect_ratio == 1.0:
                shapes = [
                    Point(p).buffer(r, quad_segs=quad_segs)
                    for p, r in zip(asperity_positions, asperity_radii)
                ]
                asperity_positions = asperity_positions[:-1]
                asperity_radii = asperity_radii[:-1]
            else:
                shapes = [
                    Point(p).buffer(r, quad_segs=quad_segs)
                    for p, r in zip(asperity_positions, asperity_radii)
                ] + [Polygon(asperity_positions)]
        shape = unary_union(shapes)
        if shape.geom_type == "MultiPolygon":
            raise ValueError(
                "Shape is not simply connected — asperities may not overlap. "
                "Try increasing asperity_radius or decreasing num_vertices."
            )

        pos_c, q, I_polar, A = compute_polygon_properties(shape, mass)
        vol = A

    n = asperity_positions.shape[0]
    Q = Quaternion.create(
        w=jnp.full((n, 1), q[0]),
        xyz=jnp.tile(q[1:], (n, 1)),
    )
    particle_center_arr = jnp.asarray(particle_center, dtype=float)
    sphere_pos = asperity_positions + particle_center_arr
    pos_c_tiled = jnp.tile(pos_c + particle_center_arr, (n, 1))

    state = State.create(
        pos=sphere_pos,
        rad=asperity_radii,
        clump_ID=jnp.zeros(n),
        volume=jnp.ones(n) * vol,
        mass=jnp.ones(n) * mass,
        inertia=jnp.full((n, 1), I_polar),
        q=Q,
    )

    state.pos_c = pos_c_tiled
    state.pos_p = Quaternion.rotate_back(Q, sphere_pos - pos_c_tiled)

    return state


def make_single_deformable_ga_particle_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    *,
    aspect_ratio: float = 1.0,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = (0.0, 0.0),
    mass: float = 1.0,
    # Energy coefficients (per body; scalars accepted)
    em: Optional[float | jnp.ndarray] = None,
    ec: Optional[float | jnp.ndarray] = None,
    eb: Optional[float | jnp.ndarray] = None,
    el: Optional[float | jnp.ndarray] = None,
    gamma: Optional[float | jnp.ndarray] = None,
    random_orientation: bool = True,
    seed: Optional[int] = None,
) -> tuple[State, DeformableParticleContainer]:
    """
    Build a single 2D GA particle as a deformable particle.

    Nodes are asperity centers (plus optional interior core). Boundary elements are a closed polygon
    through boundary nodes; core is excluded from elements/edges.
    """
    # 1) Generate GA nodes
    pts, rads = generate_asperities_2d(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        num_vertices=num_vertices,
        aspect_ratio=aspect_ratio,
        add_core=False,
        use_uniform_mesh=use_uniform_mesh,
    )
    pts = jnp.asarray(pts, dtype=float) + jnp.asarray(particle_center, dtype=float)
    rads = jnp.asarray(rads, dtype=float)

    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    shape = unary_union(
        [Point(p).buffer(r, quad_segs=1e4) for p, r in zip(pts, rads)] + [Polygon(pts)]
    )

    if random_orientation:
        import numpy as np

        if seed is None:
            seed = int(np.random.randint(0, 1_000_000_000))
        pts = _randomize_orientation(pts, key=jax.random.PRNGKey(seed))

    # 2) Build boundary ordering (exclude core if present)
    n_nodes = pts.shape[0]

    boundary_idx = jnp.arange(n_nodes, dtype=int)

    boundary_order = _order_boundary_2d(pts, boundary_idx)
    elements = _polygon_elements_from_order(boundary_order)  # (M,2)
    elements_ID = jnp.zeros((elements.shape[0],), dtype=int)

    # 3) Optional edges / bending topology
    edges = elements if el is not None else None
    edges_ID = jnp.zeros((elements.shape[0],), dtype=int) if el is not None else None

    element_adjacency = None
    element_adjacency_ID = None
    initial_bending = None
    if eb is not None:
        element_adjacency = _bending_adjacency_for_ring(elements.shape[0])
        element_adjacency_ID = jnp.zeros((element_adjacency.shape[0],), dtype=int)
        initial_bending = _initial_bending_2d(pts, elements, element_adjacency)

    # 4) State (single deformable body => deformable_ID=0)
    state = State.create(
        pos=pts,
        rad=rads,
        mass=(mass / n_nodes)
        * jnp.ones((n_nodes,), dtype=float),  # total mass constant for all particles
        # mass=(mass) * jnp.ones((n_nodes,), dtype=float),
        deformable_ID=jnp.zeros((n_nodes,), dtype=int),
        volume=jnp.ones(pts.shape[0])
        * (shape.area / n_nodes),  # dp vertices share the volume evenly
    )

    # 5) Container (single body => coefficient arrays length 1)
    em = _ensure_single_body_coeff(em, "em")
    ec = _ensure_single_body_coeff(ec, "ec")
    eb = _ensure_single_body_coeff(eb, "eb")
    el = _ensure_single_body_coeff(el, "el")
    gamma = _ensure_single_body_coeff(gamma, "gamma")
    container = DeformableParticleContainer.create(
        vertices=state.pos,
        elements=elements,
        elements_ID=elements_ID,
        element_adjacency=element_adjacency,
        element_adjacency_ID=element_adjacency_ID,
        initial_bending=initial_bending,
        edges=edges,
        edges_ID=edges_ID,
        em=em,
        ec=ec,
        eb=eb,
        el=el,
        gamma=gamma,
    )

    return state, container


def make_single_deformable_ga_particle_3d(
    asperity_radius: float,
    particle_radius: float,
    target_num_vertices: int,
    *,
    aspect_ratio: Sequence[float] = (1.0, 1.0, 1.0),
    use_uniform_mesh: bool = False,
    mesh_type: str = "ico",
    particle_center: Sequence[float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    # Energy coefficients (per body; scalars accepted)
    em: Optional[float | jnp.ndarray] = None,
    ec: Optional[float | jnp.ndarray] = None,
    eb: Optional[float | jnp.ndarray] = None,
    el: Optional[float | jnp.ndarray] = None,
    gamma: Optional[float | jnp.ndarray] = None,
    random_orientation: bool = True,
    seed: Optional[int] = None,
) -> tuple[State, DeformableParticleContainer]:
    """
    Build a single 3D GA particle as a deformable particle.

    Nodes are asperity centers (plus optional interior core). Boundary elements are the convex hull triangles
    through boundary nodes; core is excluded from elements/edges.
    """
    import numpy as np

    # 1) Generate GA nodes
    pts, rads, mesh = cast(
        tuple[jnp.ndarray, jnp.ndarray, Any],
        generate_asperities_3d(
            asperity_radius=asperity_radius,
            particle_radius=particle_radius,
            target_num_vertices=target_num_vertices,
            aspect_ratio=aspect_ratio,
            add_core=False,
            use_uniform_mesh=use_uniform_mesh,
            mesh_type=mesh_type,
            return_mesh=True,
        ),
    )
    pts = jnp.asarray(pts, dtype=float) + jnp.asarray(particle_center, dtype=float)
    rads = jnp.asarray(rads, dtype=float)

    # Compute the actual union volume from the boolean union of sphere meshes
    union_mesh = generate_mesh(
        asperity_positions=pts,
        asperity_radii=rads,
        subdivisions=4,
    )

    if random_orientation:
        if seed is None:
            seed = int(np.random.randint(0, 1_000_000_000))
        pts = _randomize_orientation(pts, key=jax.random.PRNGKey(seed))

    # 2) Determine boundary nodes (exclude core if present)
    n_nodes = pts.shape[0]

    faces = np.asarray(mesh.faces, dtype=int)
    edges = np.asarray(mesh.edges_unique, dtype=int)
    adjacency = np.asarray(mesh.face_adjacency, dtype=int)

    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    initial_bending = angle_between_normals(n[adjacency[:, 0]], n[adjacency[:, 1]])

    # 5) State (single deformable body => deformable_ID=0)
    state = State.create(
        pos=pts,
        rad=rads,
        mass=(mass / n_nodes) * jnp.ones((n_nodes,), dtype=float),
        deformable_ID=jnp.zeros((n_nodes,), dtype=int),
        volume=jnp.ones(pts.shape[0]) * (union_mesh.volume / n_nodes),
    )

    # 6) Container (single body => coefficient arrays length 1)
    em = _ensure_single_body_coeff(em, "em")
    ec = _ensure_single_body_coeff(ec, "ec")
    eb = _ensure_single_body_coeff(eb, "eb")
    el = _ensure_single_body_coeff(el, "el")
    gamma = _ensure_single_body_coeff(gamma, "gamma")
    container = DeformableParticleContainer.create(
        vertices=state.pos,
        elements=faces,
        element_adjacency=adjacency,
        initial_bending=initial_bending,
        edges=edges,
        em=em,
        ec=ec,
        eb=eb,
        el=el,
        gamma=gamma,
    )

    return state, container


def generate_asperities_3d(
    asperity_radius: float,
    particle_radius: float,
    target_num_vertices: int,
    aspect_ratio: Sequence[float] = (1.0, 1.0, 1.0),
    add_core: Optional[bool] = False,
    use_uniform_mesh: bool = False,
    mesh_type: str = "ico",
    return_mesh: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[jnp.ndarray, jnp.ndarray, Any]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid
    add_core: bool - optional.  Adds a central core particle if True, otherwise does nothing
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    mesh_type: str - one of 'ico', 'octa', or 'tetra' (icosphere, octasphere, tetrasphere).
    icosphere has the most, but smallest defects.  tetrasphere has the fewest, but largest defects.
    tetrasphere has the greatest granularity.
    ____
    returns:
    asperity_positions: jnp.ndarray - (num_vertices + add_core, 3) array of positions of the asperities
    asperity_radii: jnp.ndarray - (num_vertices + add_core,) array of radii of the asperities
    ____
    notes:
    creates a particle composed of a set of surface asperities
    places asperities along either a sphere or an ellipsoid in 3d
    ensures that the outer-most length of the particle is equal to 2 * particle_radius
    adds a core which is useful for covering up large gaps between adjacent asperities
    the number of subdivisions for the icosphere mesh is suggested from target_num_vertices
    """

    import trimesh
    import meshzoo

    if len(aspect_ratio) != 3:
        raise ValueError(
            f"Error: aspect ratio must be a 3-length list-like.  Expected 3, got {len(aspect_ratio)}"
        )
    aspect_ratio_arr = jnp.asarray(aspect_ratio)
    aspect_ratio_arr /= jnp.min(aspect_ratio_arr)
    if asperity_radius > particle_radius:
        print(
            f"Warning: asperity radius exceeds particle radius.  {asperity_radius} > {particle_radius}"
        )
    core_radius = particle_radius - asperity_radius
    if mesh_type == "tetra":
        n_tetra = jnp.maximum(jnp.round(jnp.sqrt((target_num_vertices - 2) / 2)), 1)
        pts, tri = meshzoo.tetra_sphere(int(n_tetra))
    elif mesh_type == "octa":
        n_octa = jnp.maximum(jnp.round(jnp.sqrt((target_num_vertices - 2) / 4)), 1)
        pts, tri = meshzoo.octa_sphere(int(n_octa))
    elif mesh_type == "ico":
        n_ico = jnp.maximum(jnp.round(jnp.sqrt((target_num_vertices - 2) / 10)), 1)
        pts, tri = meshzoo.icosa_sphere(int(n_ico))
    else:
        raise ValueError(
            f'Error: mesh_type {mesh_type} not supported.  Must be one of "tetra", "octa", "ico"'
        )
    pts = jnp.asarray(pts, dtype=float) * core_radius
    tri = jnp.asarray(tri, dtype=int)
    m = trimesh.Trimesh(vertices=pts, faces=tri, process=False)
    m.apply_scale(np.asarray(aspect_ratio_arr, dtype=float))  # type: ignore[no-untyped-call]
    if use_uniform_mesh and jnp.sum(aspect_ratio_arr) > 3:
        # when using an ellipsoid, re-mesh to ensure the vertices are evenly spaced
        # this avoids asperities bunching up at the major axes
        raise ValueError("Using uniform mesh isn't supported yet")
    asperity_positions = jnp.asarray(np.asarray(m.vertices, dtype=float))
    asperity_radii = jnp.ones(m.vertices.shape[0]) * asperity_radius
    if add_core:
        if jnp.all(aspect_ratio_arr == 1.0):
            asperity_positions = jnp.concatenate(
                (asperity_positions, jnp.zeros((1, 3))), axis=0
            )
            asperity_radii = jnp.concatenate(
                (asperity_radii, jnp.array([core_radius])), axis=0
            )
        else:
            print("Warning: ellipsoid core not yet supported")
    if return_mesh:
        return asperity_positions, asperity_radii, m
    return asperity_positions, asperity_radii


def generate_mesh(
    asperity_positions: jnp.ndarray, asperity_radii: jnp.ndarray, subdivisions: int
) -> Any:

    import trimesh

    meshes = []
    for a, r in zip(asperity_positions, asperity_radii):
        m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(r))
        m.apply_translation(a)
        meshes.append(m)
    engines: set[str | None] = getattr(trimesh.boolean, "engines_available", set())
    if "manifold" in engines:
        mesh = trimesh.boolean.union(meshes, engine="manifold")
    elif None in engines:
        mesh = trimesh.boolean.union(meshes, engine=None)
    else:
        raise RuntimeError(
            "No trimesh boolean backend is available; can't union sphere meshes. "
            "Install one (recommended: `pip install manifold3d`)."
        )

    assert mesh.is_volume
    return mesh


def make_single_particle_3d(
    asperity_radius: float,
    particle_radius: float,
    target_num_vertices: int,
    aspect_ratio: Sequence[float] = (1.0, 1.0, 1.0),
    body_type: Optional[str] = "solid",
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    mesh_subdivisions: int = 4,
    mesh_type: str = "ico",
) -> State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid (length 3)
    body_type: str - 'true-solid' (physical core), 'solid' (core used for volume and inertia), 'point' (point masses)
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    particle_center: Sequence[float] - optional particle center location
    mass: float - optional mass of the entire particle
    mesh_subdivisions: int - number of subdivisions for the icosphere mesh used to define the volume
    mesh_type: str - one of 'ico', 'octa', or 'tetra'
    ____
    returns:
    state: State - jaxdem state object containing the single clump particle in 3d
    """
    import numpy as np

    if body_type not in ["true-solid", "solid", "point"]:
        raise ValueError(f"body_type {body_type} not understood")

    add_core = body_type in ["true-solid", "solid"]

    asperity_positions, asperity_radii = cast(
        Tuple[jnp.ndarray, jnp.ndarray],
        generate_asperities_3d(
            asperity_radius=asperity_radius,
            particle_radius=particle_radius,
            target_num_vertices=target_num_vertices,
            aspect_ratio=aspect_ratio,
            add_core=add_core,
            use_uniform_mesh=use_uniform_mesh,
            mesh_type=mesh_type,
            return_mesh=False,
        ),
    )
    if body_type == "point":
        n = asperity_positions.shape[0]
        m_i = mass / n
        pos_c = jnp.mean(asperity_positions, axis=0)
        r_prime = asperity_positions - pos_c
        r_sq = jnp.sum(r_prime**2, axis=-1)

        # Point-mass inertia tensor: I_ij = Sigma m_k (|r_k|^2 delta_ij - r_ki r_kj)
        term1 = jnp.sum(m_i * r_sq[:, None, None] * jnp.eye(3)[None, :, :], axis=0)
        term2 = m_i * jnp.einsum("ni,nj->ij", r_prime, r_prime)
        I_tensor = term1 - term2
        I_tensor = 0.5 * (I_tensor + I_tensor.T)

        eigvals, eigvecs = jnp.linalg.eigh(I_tensor)

        eigvecs_np = np.array(eigvecs)
        if np.linalg.det(eigvecs_np) < 0:
            eigvecs_np[:, -1] *= -1

        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(eigvecs_np)
        q_xyzw = rot.as_quat()  # [x, y, z, w]
        q = jnp.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        inertia = eigvals  # (3,) principal moments
        vol = jnp.sum(4.0 / 3.0 * jnp.pi * asperity_radii**3)

    else:
        mesh = generate_mesh(
            asperity_positions=asperity_positions,
            asperity_radii=asperity_radii,
            subdivisions=mesh_subdivisions,
        )

        if body_type == "solid":
            # Remove core from physical spheres (it was only used for mesh volume/inertia).
            # generate_asperities_3d only adds a core for isotropic aspect ratios,
            # so only trim if one was actually appended.
            aspect_ratio_arr = jnp.asarray(aspect_ratio) / jnp.min(
                jnp.asarray(aspect_ratio)
            )
            if jnp.all(aspect_ratio_arr == 1.0):
                asperity_positions = asperity_positions[:-1]
                asperity_radii = asperity_radii[:-1]

        pos_c, q, inertia, vol = compute_mesh_properties(mesh, mass)

    # ---- Build State (common to all body types) ----
    n = asperity_positions.shape[0]
    Q = Quaternion.create(
        w=jnp.full((n, 1), q[0]),
        xyz=jnp.tile(q[1:], (n, 1)),
    )
    particle_center_arr = jnp.asarray(particle_center, dtype=float)
    sphere_pos = asperity_positions + particle_center_arr
    pos_c_tiled = jnp.tile(pos_c + particle_center_arr, (n, 1))

    state = State.create(
        pos=sphere_pos,
        rad=asperity_radii,
        clump_ID=jnp.zeros(n),
        volume=jnp.ones(n) * vol,
        mass=jnp.ones(n) * mass,
        inertia=jnp.tile(inertia, (n, 1)),  # (n, 3) for 3D
        q=Q,
    )

    state.pos_c = pos_c_tiled
    state.pos_p = Quaternion.rotate_back(Q, sphere_pos - pos_c_tiled)

    return state


def generate_ga_clump_state(
    particle_radii: jnp.ndarray,
    vertex_counts: jnp.ndarray,
    phi: float,
    dim: int,
    asperity_radius: float,
    *,
    seed: Optional[int] = None,
    body_type: Optional[str] = "solid",
    use_uniform_mesh: bool = False,
    mass: float = 1.0,
    aspect_ratio: Optional[Union[float, Sequence[float]]] = None,
    quad_segs: int = 10_000,
    mesh_subdivisions: int = 4,
    mesh_type: str = "ico",
    use_random_orientations: bool = True,
) -> Tuple[State, jnp.ndarray]:
    """
    Build a `jaxdem.State` containing a system of Geometric Asperity model particles as clumps in either 2D or 3D.
    """

    if particle_radii.size != vertex_counts.size:
        raise ValueError(
            f"particle_radii and vertex_counts must be the same size!  sizes do not match: {particle_radii.size} and {vertex_counts.size}"
        )

    if aspect_ratio is None:
        if dim == 2:
            aspect_ratio = 1.0
        else:
            aspect_ratio = [1.0, 1.0, 1.0]

    import numpy as np
    from tqdm import tqdm

    # create initial positions
    if seed is None:
        seed = int(np.random.randint(0, int(1e9)))
    sphere_pos, box_size = random_sphere_configuration(
        np.asarray(particle_radii, dtype=float).tolist(), phi, dim, int(seed)
    )

    rad_nv = jnp.column_stack((particle_radii, vertex_counts))
    unique_rad_nv, ids = jnp.unique(rad_nv, axis=0, return_inverse=True)
    state = None

    # loop over unique particle types
    for idx, (rad, nv) in tqdm(
        enumerate(unique_rad_nv), desc="Generating Clumps", total=unique_rad_nv.shape[0]
    ):

        # create a template state for each particle type
        nv = int(nv)
        if dim == 2:
            if not isinstance(aspect_ratio, (int, float, np.floating)):
                raise TypeError(
                    f"For dim=2, expected aspect_ratio to be a float; got {type(aspect_ratio)}"
                )
            template_state = make_single_particle_2d(
                particle_radius=rad,
                num_vertices=nv,
                asperity_radius=asperity_radius,
                body_type=body_type,
                use_uniform_mesh=use_uniform_mesh,
                mass=mass,
                aspect_ratio=float(aspect_ratio),
                quad_segs=quad_segs,
            )
        elif dim == 3:
            if isinstance(aspect_ratio, (int, float)) and aspect_ratio == 1.0:
                aspect_ratio = [1.0, 1.0, 1.0]
            aspect_ratio_3d = jnp.asarray(aspect_ratio)
            if aspect_ratio_3d.shape != (3,):
                raise TypeError(
                    f"For dim=3, expected aspect_ratio to be a length-3 sequence; got shape {aspect_ratio_3d.shape}"
                )
            template_state = make_single_particle_3d(
                particle_radius=rad,
                target_num_vertices=nv,
                asperity_radius=asperity_radius,
                body_type=body_type,
                use_uniform_mesh=use_uniform_mesh,
                mass=mass,
                aspect_ratio=tuple(float(x) for x in np.asarray(aspect_ratio_3d)),
                mesh_subdivisions=mesh_subdivisions,
                mesh_type=mesh_type,
            )
        else:
            raise ValueError(f"dim: {dim} not supported")

        # duplicate the template state for each instance of the particle type
        # set the duplicated particle positions to be at the sphere positions
        duplicated_state = duplicate_clump_template(
            template_state, sphere_pos[ids == idx]
        )

        # merge with the prior duplicated states
        if state is None:
            state = duplicated_state
        else:
            state = State.merge(state, duplicated_state)

    # randomize orientations
    if use_random_orientations:
        key = jax.random.PRNGKey(seed)
        state = randomize_orientations(state, key)
    assert state is not None
    return state, box_size


def generate_ga_deformable_state(
    particle_radii: jnp.ndarray,
    vertex_counts: jnp.ndarray,
    phi: float,
    dim: int,
    asperity_radius: float,
    *,
    seed: Optional[int] = None,
    use_uniform_mesh: bool = False,
    mass: float = 1.0,
    aspect_ratio: Optional[Union[float, Sequence[float]]] = None,
    mesh_type: str = "ico",
    # Energy coefficients: scalar or per-body arrays of shape (num_bodies,)
    em: Optional[Union[float, jnp.ndarray]] = None,
    ec: Optional[Union[float, jnp.ndarray]] = None,
    eb: Optional[Union[float, jnp.ndarray]] = None,
    el: Optional[Union[float, jnp.ndarray]] = None,
    gamma: Optional[Union[float, jnp.ndarray]] = None,
    random_orientations: bool = True,
) -> Tuple[State, DeformableParticleContainer, jnp.ndarray]:
    """
    Build a `jaxdem.State` and matching `DeformableParticleContainer` containing a system of
    Geometric Asperity model particles as deformable particles in either 2D or 3D.

    Nodes are asperity centers (plus optional core). Topology is auto-generated to support any
    subset of {em, ec, eb, el, gamma}.
    """
    if particle_radii.size != vertex_counts.size:
        raise ValueError(
            f"particle_radii and vertex_counts must be the same size! sizes do not match: {particle_radii.size} and {vertex_counts.size}"
        )

    if aspect_ratio is None:
        if dim == 2:
            aspect_ratio = 1.0
        else:
            aspect_ratio = [1.0, 1.0, 1.0]

    import numpy as np

    n_bodies = int(particle_radii.size)

    # create initial positions for body centers
    if seed is None:
        seed = int(np.random.randint(0, int(1e9)))
    sphere_pos, box_size = random_sphere_configuration(
        np.asarray(particle_radii, dtype=float).tolist(), phi, dim, int(seed)
    )
    sphere_pos = jnp.asarray(sphere_pos, dtype=float)

    # Per-body parameter arrays
    em_b = _ensure_per_body_params(em, n_bodies, "em")
    ec_b = _ensure_per_body_params(ec, n_bodies, "ec")
    eb_b = _ensure_per_body_params(eb, n_bodies, "eb")
    el_b = _ensure_per_body_params(el, n_bodies, "el")
    gamma_b = _ensure_per_body_params(gamma, n_bodies, "gamma")

    # Group by unique (radius, nv) for template reuse
    rad_nv = jnp.column_stack((particle_radii, vertex_counts))
    unique_rad_nv, ids = jnp.unique(rad_nv, axis=0, return_inverse=True)

    # Accumulators for global State
    pos_all = []
    rad_all = []
    mass_all = []
    volume_all = []
    deformable_id_all = []

    # Accumulators for global Container topology
    elements_all = []
    elements_id_all = []
    edges_all = []
    edges_id_all = []
    adjacency_all = []
    adjacency_id_all = []
    initial_bending_all = []

    # Track offsets
    node_offset = 0
    elem_offset = 0

    # Precompute templates per unique type (no random orientation here; we randomize per body below)
    templates: dict[
        int,
        tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            DeformableParticleContainer,
        ],
    ] = {}
    for type_idx, (rad, nv) in enumerate(unique_rad_nv):
        rad_f = float(rad)
        nv_i = int(nv)
        # Build template with all topology possibly needed (coefficients set to 1.0 if requested)
        if dim == 2:
            if not isinstance(aspect_ratio, (int, float, np.floating)):
                raise TypeError(
                    f"For dim=2, expected aspect_ratio to be a float; got {type(aspect_ratio)}"
                )
            t_state, t_container = make_single_deformable_ga_particle_2d(
                asperity_radius=float(asperity_radius),
                particle_radius=rad_f,
                num_vertices=nv_i,
                aspect_ratio=float(aspect_ratio),
                use_uniform_mesh=use_uniform_mesh,
                particle_center=(0.0, 0.0),
                mass=mass,
                em=1.0 if em_b is not None else None,
                ec=1.0 if ec_b is not None else None,
                eb=1.0 if eb_b is not None else None,
                el=1.0 if el_b is not None else None,
                gamma=1.0 if gamma_b is not None else None,
                random_orientation=random_orientations,
            )
        elif dim == 3:
            aspect_ratio_3d = jnp.asarray(aspect_ratio, dtype=float)
            if aspect_ratio_3d.shape != (3,):
                raise TypeError(
                    f"For dim=3, expected aspect_ratio to be a length-3 sequence; got shape {aspect_ratio_3d.shape}"
                )
            t_state, t_container = make_single_deformable_ga_particle_3d(
                asperity_radius=float(asperity_radius),
                particle_radius=rad_f,
                target_num_vertices=nv_i,
                aspect_ratio=tuple(float(x) for x in np.asarray(aspect_ratio_3d)),
                use_uniform_mesh=use_uniform_mesh,
                mesh_type=mesh_type,
                particle_center=(0.0, 0.0, 0.0),
                mass=mass,
                em=1.0 if em_b is not None else None,
                ec=1.0 if ec_b is not None else None,
                eb=1.0 if eb_b is not None else None,
                el=1.0 if el_b is not None else None,
                gamma=1.0 if gamma_b is not None else None,
                random_orientation=random_orientations,
            )
        else:
            raise ValueError(f"dim: {dim} not supported")

        templates[type_idx] = (
            t_state.pos,
            t_state.rad,
            t_state.mass,
            t_state.volume,
            t_container,
        )

    # Instantiate each body from its template
    key = jax.random.PRNGKey(int(seed))
    body_keys = jax.random.split(key, n_bodies) if random_orientations else None
    for body_idx in range(n_bodies):
        type_idx = int(ids[body_idx])
        t_pos, t_rad, t_mass, t_volume, t_container = templates[type_idx]

        n_nodes = int(t_pos.shape[0])
        if random_orientations:
            assert body_keys is not None
            pos_local = _randomize_orientation(t_pos, key=body_keys[body_idx])
        else:
            pos_local = t_pos
        pos_i = pos_local + sphere_pos[body_idx]

        pos_all.append(pos_i)
        rad_all.append(t_rad)
        mass_all.append(t_mass)
        volume_all.append(t_volume)
        deformable_id_all.append(jnp.ones((n_nodes,), dtype=int) * body_idx)

        # Elements / IDs (required for em/ec/gamma/eb)
        if (
            (em_b is not None)
            or (ec_b is not None)
            or (gamma_b is not None)
            or (eb_b is not None)
        ):
            assert (
                t_container.elements is not None and t_container.elements_ID is not None
            )
            elems = t_container.elements + node_offset
            elements_all.append(elems)
            elements_id_all.append(jnp.ones((elems.shape[0],), dtype=int) * body_idx)

        # Edges / IDs (required for el)
        if el_b is not None:
            assert t_container.edges is not None
            e = t_container.edges + node_offset
            edges_all.append(e)
            edges_id_all.append(jnp.ones((e.shape[0],), dtype=int) * body_idx)

        # Adjacency / IDs / rest bending (required for eb)
        if eb_b is not None:
            assert (
                t_container.element_adjacency is not None
                and t_container.initial_bending is not None
            )
            adj = t_container.element_adjacency + elem_offset
            adjacency_all.append(adj)
            adjacency_id_all.append(jnp.ones((adj.shape[0],), dtype=int) * body_idx)
            initial_bending_all.append(t_container.initial_bending)

        # Update offsets
        node_offset += n_nodes
        if (
            (em_b is not None)
            or (ec_b is not None)
            or (gamma_b is not None)
            or (eb_b is not None)
        ):
            elem_offset += (
                int(t_container.elements.shape[0])
                if t_container.elements is not None
                else 0
            )

    # Concatenate State arrays
    pos = jnp.concatenate(pos_all, axis=0)
    rad = jnp.concatenate(rad_all, axis=0)
    mass_arr = jnp.concatenate(mass_all, axis=0)
    volume = jnp.concatenate(volume_all, axis=0)
    deformable_ID = jnp.concatenate(deformable_id_all, axis=0)
    state = State.create(
        pos=pos,
        rad=rad,
        mass=mass_arr,
        volume=volume,
        deformable_ID=deformable_ID,
    )

    # Concatenate container arrays
    elements = jnp.concatenate(elements_all, axis=0) if elements_all else None
    elements_ID = jnp.concatenate(elements_id_all, axis=0) if elements_id_all else None
    edges = jnp.concatenate(edges_all, axis=0) if edges_all else None
    edges_ID = jnp.concatenate(edges_id_all, axis=0) if edges_id_all else None
    element_adjacency = (
        jnp.concatenate(adjacency_all, axis=0) if adjacency_all else None
    )
    element_adjacency_ID = (
        jnp.concatenate(adjacency_id_all, axis=0) if adjacency_id_all else None
    )
    initial_bending = (
        jnp.concatenate(initial_bending_all, axis=0) if initial_bending_all else None
    )

    container = DeformableParticleContainer.create(
        vertices=state.pos,
        elements=elements,
        elements_ID=elements_ID,
        edges=edges,
        edges_ID=edges_ID,
        element_adjacency=element_adjacency,
        element_adjacency_ID=element_adjacency_ID,
        initial_bending=initial_bending,
        em=em_b,
        ec=ec_b,
        eb=eb_b,
        el=el_b,
        gamma=gamma_b,
    )

    return state, container, box_size
