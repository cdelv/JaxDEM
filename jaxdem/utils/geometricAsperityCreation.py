# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions for creating Geometric Asperity particle states in 2D and 3D.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Tuple, Optional, Sequence, Union

from .quaternion import Quaternion
from .clumps import compute_clump_properties
from .randomSphereConfiguration import random_sphere_configuration
from .randomizeOrientations import randomize_orientations
from ..materials import Material, MaterialTable
from ..material_matchmakers import MaterialMatchmaker
from ..state import State
from ..forces.deformable_particle import DeformableParticleContainer, angle_between_normals


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
    def tile0(x):
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


def _ensure_per_body_params(x, n_bodies: int, name: str):
    if x is None:
        return None
    arr = jnp.asarray(x, dtype=float)
    if arr.ndim == 0:
        return jnp.ones((n_bodies,), dtype=float) * arr
    if arr.shape == (n_bodies,):
        return arr
    raise ValueError(f"{name} must be a scalar or shape ({n_bodies},), got {arr.shape}")


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


def _rotate_points_2d(pts: jnp.ndarray, theta: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
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


def _randomize_orientation(
    pts: jnp.ndarray, *, key: jax.random.KeyArray
) -> jnp.ndarray:
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


def _initial_bending_2d(vertices: jnp.ndarray, elements: jnp.ndarray, element_adjacency: jnp.ndarray) -> jnp.ndarray:
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


def _initial_bending_3d(vertices: jnp.ndarray, faces: jnp.ndarray, face_adjacency: jnp.ndarray) -> jnp.ndarray:
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
    core_type: Optional[str] = None,
    use_uniform_mesh: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipse)
    num_vertices: int - number of asperities
    aspect_ratio: float - optional aspect ratio of the ellipse
    core_type: str - optional.  None: no core added.  "false" core added only for inertia and volume
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

    if core_type is not None:
        if core_type not in ['true', 'false']:
            raise ValueError(f'Unknown value for core_type: {core_type}')
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


def make_single_particle_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    aspect_ratio: float = 1.0,
    core_type: Optional[str] = None,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = jnp.zeros(2),
    mass: float = 1.0,
    quad_segs: int = 10_000,
    use_point_inertia: bool = False
) -> State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: float - optional aspect ratios of the ellipsoid
    core_type: str - optional.  None: no core added.  "false" core added only for inertia and volume
    calculations.  "true" physical core added.
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    particle_center: Sequence[float] - optional particle center location
    mass: float - optional mass of the entire particle
    quad_segs: int - optional number of segments used to define the mass
    ____
    returns:
    single_clump_state: State - jaxdem state object containing the single clump particle in 2d
    """

    from shapely.geometry import Point
    from shapely.ops import unary_union

    asperity_positions, asperity_radii = generate_asperities_2d(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        num_vertices=num_vertices,
        aspect_ratio=aspect_ratio,
        core_type=core_type,
        use_uniform_mesh=use_uniform_mesh,
    )

    shape = unary_union(
        [
            Point(p).buffer(r, quad_segs=quad_segs)
            for p, r in zip(asperity_positions, asperity_radii)
        ]
    )

    # if only using the core for the inertia and volume calculations,
    # remove the physical core vertex before constructing the state
    if core_type == "false":
        asperity_positions = asperity_positions[:-1]
        asperity_radii = asperity_radii[:-1]

    single_clump_state = State.create(
        pos=asperity_positions + particle_center,
        rad=asperity_radii,
        clump_ID=jnp.zeros(asperity_positions.shape[0]),
        volume=jnp.ones(asperity_positions.shape[0]) * shape.area,
    )

    mats = [Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = MaterialMatchmaker.create("harmonic")
    mat_table = MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = compute_clump_properties(
        single_clump_state, mat_table, n_samples=50_000
    )

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
    if use_point_inertia:
        sphere_mass = mass / asperity_radii.size
        r = jnp.linalg.norm(single_clump_state.pos - single_clump_state.pos_c, axis=-1) ** 2
        single_clump_state.inertia = jnp.sum(sphere_mass * r) * jnp.ones_like(single_clump_state.mass)[..., None]
    else:
        single_clump_state.inertia *= (true_mass / single_clump_state.mass)[..., None]
    single_clump_state.mass = true_mass

    return single_clump_state


def make_single_deformable_ga_particle_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    *,
    aspect_ratio: float = 1.0,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = jnp.zeros(2),
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
        core_type=None,
        use_uniform_mesh=use_uniform_mesh,
    )
    pts = jnp.asarray(pts, dtype=float) + jnp.asarray(particle_center, dtype=float)
    rads = jnp.asarray(rads, dtype=float)

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
        # mass=(mass / n_nodes) * jnp.ones((n_nodes,), dtype=float),
        mass=(mass) * jnp.ones((n_nodes,), dtype=float),
        deformable_ID=jnp.zeros((n_nodes,), dtype=int),
    )

    # 5) Container (single body => coefficient arrays length 1)
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
    particle_center: Sequence[float] = jnp.zeros(3),
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
    import trimesh

    # 1) Generate GA nodes
    pts, rads, mesh = generate_asperities_3d(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        target_num_vertices=target_num_vertices,
        aspect_ratio=aspect_ratio,
        core_type=None,
        use_uniform_mesh=use_uniform_mesh,
        mesh_type=mesh_type,
        return_mesh=True,
    )
    pts = jnp.asarray(pts, dtype=float) + jnp.asarray(particle_center, dtype=float)
    rads = jnp.asarray(rads, dtype=float)

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
        # mass=(mass / n_nodes) * jnp.ones((n_nodes,), dtype=float),
        mass=(mass) * jnp.ones((n_nodes,), dtype=float),
        deformable_ID=jnp.zeros((n_nodes,), dtype=int),
    )

    # 6) Container (single body => coefficient arrays length 1)
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
    core_type: Optional[str] = None,
    use_uniform_mesh: bool = False,
    mesh_type: str = "ico",
    return_mesh: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid
    core_type: str - optional.  None: no core added.  "false" core added only for inertia and volume
    calculations.  "true" physical core added.
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
    aspect_ratio = jnp.asarray(aspect_ratio)
    aspect_ratio /= jnp.min(aspect_ratio)
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
    m.apply_scale(aspect_ratio)
    if use_uniform_mesh and jnp.sum(aspect_ratio) > 3:
        # when using an ellipsoid, re-mesh to ensure the vertices are evenly spaced
        # this avoids asperities bunching up at the major axes
        raise ValueError("Using uniform mesh isnt supported yet")
    asperity_positions = m.vertices
    asperity_radii = jnp.ones(m.vertices.shape[0]) * asperity_radius
    if core_type is not None:
        if core_type not in ['true', 'false']:
            raise ValueError(f'Unknown value for core_type: {core_type}')
        if jnp.all(aspect_ratio == 1.0):
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
):

    import trimesh

    meshes = []
    for a, r in zip(asperity_positions, asperity_radii):
        m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(r))
        m.apply_translation(a)
        meshes.append(m)
    engines = getattr(trimesh.boolean, "engines_available", set())
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
    aspect_ratio: Sequence[float] = jnp.ones(3),
    core_type: Optional[str] = None,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = jnp.zeros(3),
    mass: float = 1.0,
    mesh_subdivisions: int = 4,
    mesh_type: str = "ico",
    use_point_inertia: bool = False
) -> State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid
    core_type: str - optional.  None: no core added.  "false" core added only for inertia and volume
    calculations.  "true" physical core added.
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    particle_center: Sequence[float] - optional particle center location
    mass: float - optional mass of the entire particle
    mesh_subdivisions: int - optional number of subdivisions when making the icosphere mesh to define the mass
    ____
    returns:
    single_clump_state: State - jaxdem state object containing the single clump particle in 3d
    """
    asperity_positions, asperity_radii = generate_asperities_3d(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        target_num_vertices=target_num_vertices,
        aspect_ratio=aspect_ratio,
        core_type=core_type,
        use_uniform_mesh=use_uniform_mesh,
        mesh_type=mesh_type,
        return_mesh=False,
    )
    mesh = generate_mesh(
        asperity_positions=asperity_positions,
        asperity_radii=asperity_radii,
        subdivisions=mesh_subdivisions,
    )
    # if only using the core for the inertia and volume calculations,
    # remove the physical core vertex before constructing the state
    if core_type == "false":
        asperity_positions = asperity_positions[:-1]
        asperity_radii = asperity_radii[:-1]
    single_clump_state = State.create(
        pos=asperity_positions + particle_center,
        rad=asperity_radii,
        clump_ID=jnp.zeros(asperity_positions.shape[0]),
        volume=jnp.ones(asperity_positions.shape[0]) * mesh.volume,
    )

    mats = [Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = MaterialMatchmaker.create("harmonic")
    mat_table = MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = compute_clump_properties(
        single_clump_state, mat_table, n_samples=50_000
    )

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
    if use_point_inertia:
        raise NotImplementedError('Point-mass inertia not implemented for 3D yet!')
    else:
        single_clump_state.inertia *= (true_mass / single_clump_state.mass)[..., None]
    single_clump_state.mass = true_mass

    return single_clump_state


def generate_ga_clump_state(
    particle_radii: jnp.ndarray,
    vertex_counts: jnp.ndarray,
    phi: float,
    dim: int,
    asperity_radius: float,
    *,
    seed: Optional[float] = None,
    core_type: Optional[str] = None,
    use_uniform_mesh: bool = False,
    mass: float = 1.0,
    aspect_ratio: Optional[Union[float, Sequence[float]]] = None,
    quad_segs: int = 10_000,
    mesh_subdivisions: int = 4,
    mesh_type: str = "ico",
    use_point_inertia: bool = False
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
        seed = np.random.randint(0, 1e9)
    sphere_pos, box_size = random_sphere_configuration(particle_radii, phi, dim, seed)

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
                core_type=core_type,
                use_uniform_mesh=use_uniform_mesh,
                mass=mass,
                aspect_ratio=float(aspect_ratio),
                quad_segs=quad_segs,
                use_point_inertia=use_point_inertia
            )
        elif dim == 3:
            if aspect_ratio == 1.0:
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
                core_type=core_type,
                use_uniform_mesh=use_uniform_mesh,
                mass=mass,
                aspect_ratio=aspect_ratio_3d,
                mesh_subdivisions=mesh_subdivisions,
                mesh_type=mesh_type,
                use_point_inertia=use_point_inertia
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
    key = jax.random.PRNGKey(seed)
    state = randomize_orientations(state, key)
    return state, box_size


def generate_ga_deformable_state(
    particle_radii: jnp.ndarray,
    vertex_counts: jnp.ndarray,
    phi: float,
    dim: int,
    asperity_radius: float,
    *,
    seed: Optional[float] = None,
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
        seed = np.random.randint(0, 1e9)
    sphere_pos, box_size = random_sphere_configuration(particle_radii, phi, dim, seed)
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
    templates: dict[int, tuple[jnp.ndarray, jnp.ndarray, DeformableParticleContainer]] = {}
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
                particle_center=jnp.zeros(2),
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
                particle_center=jnp.zeros(3),
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

        templates[type_idx] = (t_state.pos, t_state.rad, t_state.mass, t_container)

    # Instantiate each body from its template
    key = jax.random.PRNGKey(int(seed))
    body_keys = jax.random.split(key, n_bodies) if random_orientations else None
    for body_idx in range(n_bodies):
        type_idx = int(ids[body_idx])
        t_pos, t_rad, t_mass, t_container = templates[type_idx]

        n_nodes = int(t_pos.shape[0])
        if random_orientations:
            pos_local = _randomize_orientation(
                t_pos, key=body_keys[body_idx]
            )
        else:
            pos_local = t_pos
        pos_i = pos_local + sphere_pos[body_idx]

        pos_all.append(pos_i)
        rad_all.append(t_rad)
        mass_all.append(t_mass)
        deformable_id_all.append(jnp.ones((n_nodes,), dtype=int) * body_idx)

        # Elements / IDs (required for em/ec/gamma/eb)
        if (em_b is not None) or (ec_b is not None) or (gamma_b is not None) or (eb_b is not None):
            assert t_container.elements is not None and t_container.elements_ID is not None
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
        if (em_b is not None) or (ec_b is not None) or (gamma_b is not None) or (eb_b is not None):
            elem_offset += int(t_container.elements.shape[0]) if t_container.elements is not None else 0

    # Concatenate State arrays
    pos = jnp.concatenate(pos_all, axis=0)
    rad = jnp.concatenate(rad_all, axis=0)
    mass_arr = jnp.concatenate(mass_all, axis=0)
    deformable_ID = jnp.concatenate(deformable_id_all, axis=0)
    state = State.create(
        pos=pos,
        rad=rad,
        mass=mass_arr,
        # mass=mass * jnp.ones((pos.shape[0],), dtype=float),
        # mass=0.01 * jnp.ones((pos.shape[0],), dtype=float),
        deformable_ID=deformable_ID,
    )

    # Concatenate container arrays
    elements = jnp.concatenate(elements_all, axis=0) if elements_all else None
    elements_ID = jnp.concatenate(elements_id_all, axis=0) if elements_id_all else None
    edges = jnp.concatenate(edges_all, axis=0) if edges_all else None
    edges_ID = jnp.concatenate(edges_id_all, axis=0) if edges_id_all else None
    element_adjacency = jnp.concatenate(adjacency_all, axis=0) if adjacency_all else None
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
