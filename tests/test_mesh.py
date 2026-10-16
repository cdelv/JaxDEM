# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for mesh and facet creation functions."""

import jax.numpy as jnp
import numpy as np

import jaxdem as jdem


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

    # Uniform solid right tetrahedron: V=1/6, COM=(1/4,1/4,1/4),
    # covariance diag=3/80, offdiag=-1/80. Reconstruct the world tensor
    # independently of the production quaternion helpers/eigenvector ordering.
    np.testing.assert_allclose(com_rigid, [0.25] * 3, atol=2e-7)
    np.testing.assert_allclose(state_rigid.volume, 1 / 6, rtol=2e-6)
    xyz = np.asarray(state_rigid.q.xyz[0])
    w = float(state_rigid.q.w[0, 0])
    x, y, z = xyz
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + 2 * w * skew + 2 * skew @ skew
    from jaxdem.state import INERTIA_REGULARIZATION

    moments = np.asarray(inertia_rigid) - 4 * INERTIA_REGULARIZATION
    expected_tensor = 4 * (np.eye(3) / 16 + np.ones((3, 3)) / 80)
    np.testing.assert_allclose(
        rotation @ np.diag(moments) @ rotation.T, expected_tensor, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        state_rigid.pos,
        np.asarray(vertices)[np.asarray(faces)].reshape(-1, 3),
        atol=2e-6,
    )

    # Flexible mesh
    state_flex = jdem.State.add_mesh(
        state, vertices, faces, thickness=0.1, rigid=False, mass=4.0
    )
    assert state_flex.N == 12
    # Check that flexible mesh vertices do not share clump_ids
    assert np.unique(state_flex.clump_id).size == 12


def test_rigid_icosahedron():
    from jaxdem.utils.meshes import _icosahedron_faces, _icosahedron_vertices
    from jaxdem.state import INERTIA_REGULARIZATION

    vertices = np.asarray(_icosahedron_vertices()) * 2
    velocity = np.array([0.2, -0.3, 0.1])
    state = jdem.State.add_mesh(
        jdem.State.create(),
        vertices,
        _icosahedron_faces(),
        rigid=True,
        filled=True,
        mass=10.0,
        vel=jnp.asarray(velocity),
        ang_vel=jnp.array([0.0, 0.0, 1.0]),
    )
    # Solid regular icosahedron: I=M*a^2*(3+sqrt(5))/20, R=2.
    phi = (1 + np.sqrt(5)) / 2
    edge_squared = 16 / (1 + phi**2)
    expected_inertia = 10 * edge_squared * (3 + np.sqrt(5)) / 20
    np.testing.assert_allclose(
        state.inertia, expected_inertia + 10 * INERTIA_REGULARIZATION, rtol=2e-6
    )
    before = np.asarray(state.pos)
    center = np.asarray(state.pos_c[0])
    system = jdem.System.create(
        state=state,
        dt=0.01,
        collider_type="",
        domain_type="free",
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
    )
    state, system = jdem.System.initialize(state, system)
    final, _ = jdem.System.step(state, system, n=4)
    theta = 0.04
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    expected = center + theta * velocity + (before - center) @ rotation.T
    np.testing.assert_allclose(final.pos, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        final.vel, np.broadcast_to(velocity, final.vel.shape), atol=2e-7
    )
    np.testing.assert_allclose(
        final.q.w**2 + jnp.sum(final.q.xyz**2, axis=-1, keepdims=True), 1.0, atol=2e-6
    )


def test_flexible_mesh():
    # Interior triangle contact: barycentric weights (1/2,1/4,1/4).
    # A compliant spring permits overlap; force and vertex reactions are exact.
    triangle = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    state = jdem.State.add_mesh(
        jdem.State.create(),
        triangle,
        jnp.array([[0, 1, 2]]),
        rigid=False,
        filled=False,
        thickness=0.1,
        mass=3.0,
        species_id=1,
    )
    sphere = jdem.State.create(
        pos=jnp.array([[0.5, 0.5, 0.2]]),
        rad=jnp.array([0.2]),
        species_id=jnp.array([0]),
    )
    state = jdem.State.merge(state, sphere)
    table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=1.0, density=1.0, poisson=0.25)]
    )
    law = jdem.ForceRouter.from_dict(
        2, {(0, 1): jdem.ForceModel.create("sphere_facet_spring")}
    )
    system = jdem.System.create(
        state=state, force_model=law, mat_table=table, collider_type="naive", dt=0.01
    )
    evaluated, _ = jdem.System.evaluate_forces(state, system)
    expected = np.array([[0, 0, -0.05], [0, 0, -0.025], [0, 0, -0.025], [0, 0, 0.1]])
    np.testing.assert_allclose(evaluated.force, expected, rtol=2e-6, atol=1e-7)
    state, system = jdem.System.initialize(state, system)
    advanced, _ = jdem.System.step(state, system)
    expected_pos = (
        np.asarray(state.pos)
        + 0.5 * system.dt**2 * expected / np.asarray(state.mass)[:, None]
    )
    np.testing.assert_allclose(advanced.pos, expected_pos, rtol=2e-6, atol=1e-8)
