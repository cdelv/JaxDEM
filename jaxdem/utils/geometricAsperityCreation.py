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


def duplicate_clump_template(template: State, com_positions: jnp.ndarray) -> State:
    """
    template: a single clump with Ns spheres (template.pos_c same for all spheres, template.ID same for all spheres)
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
        ID=ID,
        mat_id=tile0(template.mat_id),
        species_id=tile0(template.species_id),
        fixed=tile0(template.fixed),
    )


def generate_asperities_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    aspect_ratio: float = 1.0,
    add_core: bool = False,
    use_uniform_mesh: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipse)
    num_vertices: int - number of asperities
    aspect_ratio: float - optional aspect ratio of the ellipse
    add_core: bool - whether to construct the particles with a solid core
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


def make_single_particle_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    aspect_ratio: float = 1.0,
    add_core: bool = True,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = jnp.zeros(2),
    mass: float = 1.0,
    quad_segs: int = 10_000,
) -> State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: float - optional aspect ratios of the ellipsoid
    add_core: bool - whether to construct the particles with a solid core
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
        add_core=add_core,
        use_uniform_mesh=use_uniform_mesh,
    )

    shape = unary_union(
        [
            Point(p).buffer(r, quad_segs=quad_segs)
            for p, r in zip(asperity_positions, asperity_radii)
        ]
    )

    single_clump_state = State.create(
        pos=asperity_positions + particle_center,
        rad=asperity_radii,
        ID=jnp.zeros(asperity_positions.shape[0]),
        volume=jnp.ones(asperity_positions.shape[0])
        * shape.area
        / asperity_positions.shape[0],
    )

    mats = [Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = MaterialMatchmaker.create("harmonic")
    mat_table = MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = compute_clump_properties(
        single_clump_state, mat_table, n_samples=50_000
    )

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
    single_clump_state.inertia *= (true_mass / single_clump_state.mass)[..., None]
    single_clump_state.mass = true_mass

    return single_clump_state


def generate_asperities_3d(
    asperity_radius: float,
    particle_radius: float,
    target_num_vertices: int,
    aspect_ratio: Sequence[float] = [1.0, 1.0, 1.0],
    add_core: bool = False,
    use_uniform_mesh: bool = False,
    mesh_type: str = "ico",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid
    add_core: bool - whether to construct the particles with a solid core
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
        pts, tri = meshzoo.tetra_sphere(n_tetra)
    elif mesh_type == "octa":
        n_octa = jnp.maximum(jnp.round(jnp.sqrt((target_num_vertices - 2) / 4)), 1)
        pts, tri = meshzoo.octa_sphere(n_octa)
    elif mesh_type == "ico":
        n_ico = jnp.maximum(jnp.round(jnp.sqrt((target_num_vertices - 2) / 10)), 1)
        pts, tri = meshzoo.icosa_sphere(n_ico)
    else:
        raise ValueError(
            f'Error: mesh_type {mesh_type} not supported.  Must be one of "tetra", "octa", "ico"'
        )
    pts = jnp.asarray(pts, dtype=float) * particle_radius
    tri = jnp.asarray(tri, dtype=int)
    m = trimesh.Trimesh(vertices=pts, faces=tri, process=False)
    m.apply_scale(aspect_ratio)
    if use_uniform_mesh and jnp.sum(aspect_ratio) > 3:
        # when using an ellipsoid, re-mesh to ensure the vertices are evenly spaced
        # this avoids asperities bunching up at the major axes
        raise ValueError("Using uniform mesh isnt supported yet")
    asperity_positions = m.vertices
    asperity_radii = jnp.ones(m.vertices.shape[0]) * asperity_radius
    if add_core:
        if jnp.all(aspect_ratio == 1.0):
            asperity_positions = jnp.concatenate(
                (asperity_positions, jnp.zeros((1, 3))), axis=0
            )
            asperity_radii = jnp.concatenate(
                (asperity_radii, jnp.array([core_radius])), axis=0
            )
        else:
            print("Warning: ellipsoid core not yet supported")
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
    add_core: bool = True,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = jnp.zeros(3),
    mass: float = 1.0,
    mesh_subdivisions: int = 4,
    mesh_type: str = "ico",
) -> State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid
    add_core: bool - whether to construct the particles with a solid core
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
        add_core=add_core,
        use_uniform_mesh=use_uniform_mesh,
        mesh_type=mesh_type,
    )
    mesh = generate_mesh(
        asperity_positions=asperity_positions,
        asperity_radii=asperity_radii,
        subdivisions=mesh_subdivisions,
    )
    single_clump_state = State.create(
        pos=asperity_positions + particle_center,
        rad=asperity_radii,
        ID=jnp.zeros(asperity_positions.shape[0]),
        volume=jnp.ones(asperity_positions.shape[0])
        * mesh.volume
        / asperity_positions.shape[0],
    )

    mats = [Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = MaterialMatchmaker.create("harmonic")
    mat_table = MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = compute_clump_properties(
        single_clump_state, mat_table, n_samples=50_000
    )

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
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
    add_core: bool = True,
    use_uniform_mesh: bool = False,
    mass: float = 1.0,
    aspect_ratio: Union[float, Sequence[float]] = 1.0,
    quad_segs: int = 10_000,
    mesh_subdivisions: int = 4,
    mesh_type: str = "ico",
) -> Tuple[State, jnp.ndarray]:
    """
    Build a `jaxdem.State` containing a system of Geometric Asperity model particles as clumps in either 2D or 3D.
    """

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
                add_core=add_core,
                use_uniform_mesh=use_uniform_mesh,
                mass=mass,
                aspect_ratio=float(aspect_ratio),
                quad_segs=quad_segs,
            )
        elif dim == 3:
            aspect_ratio_3d = jnp.asarray(aspect_ratio)
            if aspect_ratio_3d.shape != (3,):
                raise TypeError(
                    f"For dim=3, expected aspect_ratio to be a length-3 sequence; got shape {aspect_ratio_3d.shape}"
                )
            template_state = make_single_particle_3d(
                particle_radius=rad,
                target_num_vertices=nv,
                asperity_radius=asperity_radius,
                add_core=add_core,
                use_uniform_mesh=use_uniform_mesh,
                mass=mass,
                aspect_ratio=aspect_ratio_3d,
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
    key = jax.random.PRNGKey(seed)
    state = randomize_orientations(state, key)
    return state, box_size
