# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for mesh and facets creation functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem

jax.config.update("jax_enable_x64", True)


def test_add_facet_with_safety_factor():
    state = jdem.State.create()
    vertices = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # 1. safety_factor = 1.0 (default)
    state1 = jdem.State.add_facet(
        state, vertices, thickness=0.1, rigid=True, safety_factor=1.0
    )

    # 2. safety_factor = 1.5
    state2 = jdem.State.add_facet(
        state, vertices, thickness=0.1, rigid=True, safety_factor=1.5
    )

    # Compare search radii _rad
    assert np.allclose(state2._rad, state1._rad * 1.5)

    # 3. Flexible facet with safety_factor = 2.0
    state3 = jdem.State.add_facet(
        state, vertices, thickness=0.1, rigid=False, safety_factor=2.0
    )
    state4 = jdem.State.add_facet(
        state, vertices, thickness=0.1, rigid=False, safety_factor=1.0
    )
    assert np.allclose(state3._rad, state4._rad * 2.0)


def test_add_mesh_rigid_and_flexible():
    state = jdem.State.create()

    # A simple tetrahedron mesh (4 vertices, 4 faces)
    vertices = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = jnp.array(
        [
            [0, 2, 1],  # bottom face
            [0, 1, 3],  # front face
            [0, 3, 2],  # side face
            [1, 2, 3],  # diagonal face
        ]
    )

    # Rigid filled mesh
    state_rigid = jdem.State.add_mesh(
        state, vertices, faces, thickness=0.1, rigid=True, filled=True, mass=4.0
    )

    # Rigid hollow mesh
    state_hollow = jdem.State.add_mesh(
        state, vertices, faces, thickness=0.1, rigid=True, filled=False, mass=4.0
    )

    # Check that they have the correct number of particles (4 faces * 3 vertices = 12 particles)
    assert state_rigid.N == 12
    assert state_hollow.N == 12

    # Check that all particles share the same clump_id
    assert np.unique(state_rigid.clump_id).size == 1
    assert np.unique(state_hollow.clump_id).size == 1

    # Check that COM is computed using triangles and differs between filled and hollow
    com_rigid = state_rigid.pos_c[0]
    com_hollow = state_hollow.pos_c[0]
    assert not np.allclose(com_rigid, com_hollow)

    # Check that moments of inertia differ
    inertia_rigid = state_rigid.inertia[0]
    inertia_hollow = state_hollow.inertia[0]
    assert not np.allclose(inertia_rigid, inertia_hollow)

    # Flexible mesh
    state_flex = jdem.State.add_mesh(
        state, vertices, faces, thickness=0.1, rigid=False, mass=4.0
    )
    assert state_flex.N == 12
    # Check that flexible mesh vertices do not share clump_ids
    assert np.unique(state_flex.clump_id).size == 12


def test_rigid_icosahedron():
    from jaxdem.utils.meshes import _icosahedron_faces, _icosahedron_vertices

    dim = 3
    # 1. Create a rigid icosahedron
    state = jdem.State.create()
    vertices = _icosahedron_vertices() * 2.0  # radius 2
    faces = _icosahedron_faces()

    key = jax.random.PRNGKey(42)
    key_v, key_w = jax.random.split(key)
    vel = jax.random.uniform(key_v, (3,), minval=-5.0, maxval=5.0)
    ang_vel = jax.random.uniform(key_w, (3,), minval=-2.0, maxval=2.0)

    # 1.1 First check inertia
    state = jdem.State.add_mesh(
        state,
        vertices,
        faces,
        rigid=True,
        filled=True,
        mass=10.0,
        vel=vel,
        ang_vel=ang_vel,
    )

    # From Monte Carlo and volume/density scaling:
    # V ~ 20.289, unit_density_inertia ~ 23.491
    # Expected I = 10.0 * 23.491 / 20.289 = ~11.5787
    expected_inertia = 11.578708763999668

    computed_inertia = state.inertia[0, 0]  # principal inertia
    np.testing.assert_allclose(computed_inertia, expected_inertia, rtol=1e-3)

    # 1.2 Add reflective box and random velocity
    system = jdem.System.create(
        state.shape,
        domain_type="reflect",
        collider_type="naive",
        dt=0.001,
        domain_kw={
            "box_size": jnp.array([10.0, 10.0, 10.0]),
            "anchor": jnp.array([-2.0, -2.0, -2.0]),
        },
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
    )

    # Let it bounce around
    state, system = system.step(state, system, n=50000)

    # 1.3 Check that vertices still represent the icosahedron correctly
    num_mesh_vertices = len(faces) * 3

    # In JaxDEM, rigid clump slave particles store their relative positions in pos_p.
    # The distance from the center of mass is simply the norm of pos_p.
    distances = jnp.linalg.norm(state.pos_p[:num_mesh_vertices], axis=-1)

    np.testing.assert_allclose(distances, 2.0, atol=1e-5)


def test_flexible_mesh():
    from jaxdem.utils.meshes import _icosahedron_faces, _icosahedron_vertices

    dim = 3
    # 1. Create a flexible icosahedron (rigid=False)
    state = jdem.State.create()
    vertices = _icosahedron_vertices() * 2.0  # radius 2
    faces = _icosahedron_faces()

    state = jdem.State.add_mesh(
        state,
        vertices,
        faces,
        rigid=False,
        filled=False,
        mass=10.0,
        thickness=1e-2,
    )

    # Note: Flexible particles do not have a uniform 'center of mass' constraint.
    # They are just individual facet particles.

    # 2. Add normal spheres falling on top
    N_spheres = 10
    sphere_rad = 0.2

    grid_x, grid_y = jnp.meshgrid(
        jnp.linspace(-1.5, 1.5, int(jnp.sqrt(N_spheres))),
        jnp.linspace(-1.5, 1.5, int(jnp.sqrt(N_spheres))),
    )

    sphere_pos = jnp.stack(
        [grid_x.flatten(), grid_y.flatten(), jnp.full(grid_x.size, 5.0)], axis=-1
    )
    sphere_rads = jnp.full(grid_x.size, sphere_rad)
    sphere_mass = jnp.full(grid_x.size, 0.1)

    # We add spheres with an initial downward velocity
    state = jdem.State.add_clump(
        state,
        pos=sphere_pos,
        rad=sphere_rads,
        mass=sphere_mass,
        vel=jnp.array([0.0, 0.0, -2.0]),
    )

    system = jdem.System.create(
        state.shape,
        domain_type="reflect",
        collider_type="naive",
        dt=0.001,
        domain_kw={
            "box_size": jnp.array([10.0, 10.0, 10.0]),
            "anchor": jnp.array([-5.0, -5.0, -2.0]),
        },
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
    )

    state, system, _ = system.trajectory_rollout(state, system, n=1, stride=3000)

    # 3. Check that no spheres penetrated into the icosahedron.
    # A point is inside the icosahedron if it is on the negative half-space of ALL facet planes.

    sphere_mask = state.facet_id == -1
    sph_pos = state.pos_c[sphere_mask]

    facet_mask = state.facet_id != -1
    fac_pos = state.pos_c[facet_mask]
    fac_q = jax.tree.map(lambda x: x[facet_mask], state.q)

    # Reconstruct facet normals
    # Facet vertices are initially arranged to form a triangle in xy plane if pos_p was reconstructed
    # Or we can just use the quaternion since facets point their normal in the Z direction locally.

    # In JaxDEM, a facet's normal is the rotated z-axis (0, 0, 1)
    normals = jax.vmap(lambda q: q.rotate(q, jnp.array([0.0, 0.0, 1.0])))(fac_q)

    # Distance from each sphere to each plane
    # If a sphere is inside, it's behind ALL planes.
    # We check if there's any sphere that is behind all planes by more than its radius.

    # For each sphere, compute signed distance to all facet planes
    # dot(pos - plane_center, normal)
    # fac_pos is the center of the facet.

    def check_sphere_inside(s_pos, s_rad):
        # distance = (s_pos - fac_pos) \cdot normals
        distances = jnp.sum((s_pos[None, :] - fac_pos) * normals, axis=-1)
        # It's strictly inside if distance < -s_rad for all facets
        is_inside = jnp.all(distances < -s_rad)
        return is_inside

    any_inside = jax.vmap(check_sphere_inside)(sph_pos, state.rad[sphere_mask])

    assert not jnp.any(any_inside), "A sphere penetrated the flexible mesh!"
