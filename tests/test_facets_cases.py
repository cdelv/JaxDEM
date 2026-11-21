# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for rigid and deformable facet use cases."""

import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jdem


@pytest.mark.parametrize("dim", [2, 3])
def test_facet_rigid_vs_deformable(dim):
    state = jdem.State.create()
    if dim == 2:
        vertices = jnp.array([[0.0, 0.0], [2.0, 0.0]])
        sphere_pos = jnp.array([[1.5, 0.1]])  # barycentric [0.25, 0.75]
        nearest = np.array([1.5, 0.0])
        expected_weights = jnp.array([0.25, 0.75])
    else:
        vertices = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        sphere_pos = jnp.array([[0.5, 1.0, 0.1]])  # barycentric [0.25, 0.25, 0.5]
        nearest = np.array([0.5, 1.0, 0.0])
        expected_weights = jnp.array([0.25, 0.25, 0.5])

    state = jdem.State.add_facet(
        state, vertices, thickness=0.1, species_id=jnp.array([1])
    )
    sphere_state = jdem.State.create(
        pos=sphere_pos,
        rad=jnp.array([0.2]),
        mass=jnp.array([1.0]),
        species_id=jnp.array([0]),
    )
    state = jdem.State.merge(state, sphere_state)

    mat = jdem.Material.create("elastic", young=1e4, poisson=0.3, density=1.0)
    mat_table = jdem.MaterialTable.from_materials([mat])
    router = jdem.ForceRouter.from_dict(
        S=2,
        mapping={
            (0, 0): jdem.ForceModel.create("spring"),
            (1, 0): jdem.ForceModel.create("sphere_facet_spring"),
            (1, 1): jdem.ForceModel.create("facet_facet_spring"),
        },
    )
    system = jdem.System.create(
        state.shape,
        domain_type="free",
        collider_type="Naive",
        dt=0.001,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )
    tol = 1e-11 if state.pos.dtype == jnp.float64 else 1e-4

    # Analytical overlap = (rad + thickness) - dist = (0.2 + 0.1) - 0.1 = 0.2
    stiffness = float(system.mat_table.young_eff[0, 0])
    expected_sphere_force = np.zeros(dim)
    expected_sphere_force[-1] = stiffness * 0.2
    expected_energy = 0.5 * stiffness * 0.2**2

    # Rigid single facet: force and analytical torque broadcast to all vertices.
    state_rigid, _ = system.collider.compute_force(state, system)
    state_rigid, _ = system.force_manager.apply(state_rigid, system)
    f_sphere_rigid = state_rigid.force[-1]
    np.testing.assert_allclose(f_sphere_rigid, expected_sphere_force, rtol=1e-6, atol=tol)
    np.testing.assert_allclose(
        state_rigid.force[:dim],
        np.tile(-expected_sphere_force, (dim, 1)),
        rtol=1e-6,
        atol=tol,
    )
    lever = nearest - np.mean(np.asarray(vertices), axis=0)
    if dim == 2:
        expected_torque = np.array(
            [lever[0] * (-expected_sphere_force[1]) - lever[1] * (-expected_sphere_force[0])]
        )
    else:
        expected_torque = np.cross(lever, -expected_sphere_force)
    np.testing.assert_allclose(
        state_rigid.torque[:dim],
        np.tile(expected_torque, (dim, 1)),
        rtol=1e-6,
        atol=tol,
    )

    # Deformable facet: barycentric distribution and zero nodal torque.
    state_deformable = jdem.State.add_facet(
        jdem.State.create(),
        vertices,
        thickness=0.1,
        species_id=jnp.array([1]),
        rigid=False,
    )
    state_deformable = jdem.State.merge(state_deformable, sphere_state)
    state_deformable, _ = system.collider.compute_force(state_deformable, system)
    state_deformable, _ = system.force_manager.apply(state_deformable, system)

    f_sphere_def = state_deformable.force[-1]
    f_vertices_def = state_deformable.force[:dim]
    np.testing.assert_allclose(f_sphere_def, f_sphere_rigid, rtol=1e-6, atol=tol)
    np.testing.assert_allclose(
        jnp.sum(f_vertices_def, axis=0), -f_sphere_def, rtol=1e-6, atol=tol
    )
    np.testing.assert_allclose(
        f_vertices_def, -expected_weights[:, None] * f_sphere_def, rtol=1e-6, atol=tol
    )
    np.testing.assert_allclose(state_deformable.torque[:dim], 0.0, atol=tol)

    _, _, energy_rigid = system.collider.compute_potential_energy(state, system)
    _, _, energy_deformable = system.collider.compute_potential_energy(
        state_deformable, system
    )
    np.testing.assert_allclose(energy_rigid, expected_energy, rtol=1e-6, atol=tol)
    np.testing.assert_allclose(energy_deformable, energy_rigid, rtol=1e-6, atol=tol)


def test_clump_of_rigid_facets():
    # Two coplanar triangles forming a welded rigid unit square with valid shared COM.
    vertices = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    faces = jnp.array([[0, 1, 2], [2, 1, 3]])
    state = jdem.State.add_mesh(
        jdem.State.create(dim=3),
        vertices,
        faces,
        thickness=0.1,
        rigid=True,
        filled=False,
        mass=6.0,
        species_id=1,
    )
    # Sphere overlapping only the first triangle at nearest point (0.2, 0.2, 0.0).
    sphere_state = jdem.State.create(
        pos=jnp.array([[0.2, 0.2, 0.1]]),
        rad=jnp.array([0.2]),
        mass=jnp.array([1.0]),
        species_id=jnp.array([0]),
    )
    state = jdem.State.merge(state, sphere_state)

    mat = jdem.Material.create("elastic", young=1e4, poisson=0.3, density=1.0)
    mat_table = jdem.MaterialTable.from_materials([mat])
    router = jdem.ForceRouter.from_dict(
        S=2,
        mapping={
            (0, 0): jdem.ForceModel.create("spring"),
            (1, 0): jdem.ForceModel.create("sphere_facet_spring"),
            (1, 1): jdem.ForceModel.create("facet_facet_spring"),
        },
    )
    system = jdem.System.create(
        state.shape,
        domain_type="free",
        collider_type="Naive",
        dt=0.001,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )

    state, _ = system.collider.compute_force(state, system)
    state, _ = system.force_manager.apply(state, system)

    stiffness = float(system.mat_table.young_eff[0, 0])
    expected_sphere_force = np.array([0.0, 0.0, stiffness * 0.2])
    expected_com = np.array([0.5, 0.5, 0.0])
    expected_torque = np.cross(
        np.array([0.2, 0.2, 0.0]) - expected_com, -expected_sphere_force
    )

    np.testing.assert_allclose(state.force[-1], expected_sphere_force, rtol=1e-6)
    np.testing.assert_allclose(
        state.force[:6], np.tile(-expected_sphere_force, (6, 1)), rtol=1e-6, atol=1e-5
    )
    np.testing.assert_allclose(
        state.torque[:6], np.tile(expected_torque, (6, 1)), rtol=1e-6, atol=1e-5
    )


def test_reflect_domain_with_facets():
    state = jdem.State.create()
    # Moving rigid facet penetrating x=0 wall by 0.005 (< |v*dt| = 0.01) at vertex 0 (rad=0.1).
    vertices_mov = jnp.array([[0.095, 0.5, 0.5], [0.9, 0.5, 0.5], [0.5, 0.9, 0.5]])
    state = jdem.State.add_facet(
        state,
        vertices_mov,
        thickness=0.1,
        vel=jnp.array([-10.0, 0.0, 0.0]),
        fixed=False,
        species_id=0,
    )
    # Fixed rigid facet crossing the x=0 wall must remain untouched.
    vertices_fix = jnp.array([[-0.1, 1.5, 1.5], [0.5, 1.5, 1.5], [0.2, 1.9, 1.5]])
    state = jdem.State.add_facet(
        state,
        vertices_fix,
        thickness=0.1,
        vel=jnp.array([0.0, 0.0, 0.0]),
        fixed=True,
        species_id=1,
    )

    mat = jdem.Material.create("elastic", young=1e4, poisson=0.3, density=1.0)
    system = jdem.System.create(
        state.shape,
        domain_type="reflect",
        domain_kw={"box_size": (2.0, 2.0, 2.0), "anchor": (0.0, 0.0, 0.0)},
        collider_type="Naive",
        dt=0.001,
        mat_table=jdem.MaterialTable.from_materials([mat]),
    )
    state_new, _ = system.domain.apply(state, system)

    # Moving rigid facet reflects out of the wall, preserves rigid vertex distances, and gains +x velocity.
    assert np.all(state_new.vel[:3, 0] > 0.0)
    assert np.all(state_new.pos[:3, 0] >= 0.1 - 1e-6)
    np.testing.assert_allclose(
        state_new.pos_c[:3], np.broadcast_to(state_new.pos_c[0], (3, 3)), atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(state_new.pos[:3] - state_new.pos_c[:3], axis=-1),
        np.linalg.norm(state.pos[:3] - state.pos_c[:3], axis=-1),
        rtol=1e-6,
    )
    # Fixed facet remains completely unchanged.
    np.testing.assert_allclose(state_new.pos_c[3:6], state.pos_c[3:6])
    np.testing.assert_allclose(state_new.vel[3:6], 0.0)


@pytest.mark.parametrize("rigid", [False, True])
@pytest.mark.parametrize(
    "center,nearest,weights",
    [
        ([1.0, -0.12, 0.16], [1.0, 0, 0], [0.5, 0.5, 0]),
        ([-0.12, -0.16, 0], [0, 0, 0], [1, 0, 0]),
    ],
)
def test_facet_edge_and_vertex_contacts_have_the_predicted_force_and_moment(
    rigid, center, nearest, weights
):
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    state = jdem.State.add_facet(
        jdem.State.create(dim=3), vertices, thickness=0.1, rigid=rigid, species_id=1
    )
    state = jdem.State.merge(
        state,
        jdem.State.create(
            pos=jnp.array([center]), rad=jnp.array([0.25]), species_id=jnp.array([0])
        ),
    )
    law = jdem.ForceRouter.from_dict(
        2, {(0, 1): jdem.ForceModel.create("sphere_facet_spring")}
    )
    table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=10.0, poisson=0.25, density=1.0)]
    )
    system = jdem.System.create(state=state, force_model=law, mat_table=table)
    actual, system = jdem.System.evaluate_forces(state, system)
    normal = (np.array(center) - nearest) / 0.2
    force = 1.5 * normal
    np.testing.assert_allclose(actual.force[-1], force, atol=2e-6)
    if rigid:
        np.testing.assert_allclose(actual.force[:3], np.tile(-force, (3, 1)), atol=2e-6)
        torque = np.cross(np.array(nearest) - vertices.mean(axis=0), -force)
        np.testing.assert_allclose(
            actual.torque[:3], np.tile(torque, (3, 1)), atol=2e-6
        )
    else:
        np.testing.assert_allclose(
            actual.force[:3], -np.array(weights)[:, None] * force, atol=2e-6
        )
        np.testing.assert_array_equal(actual.torque[:3], 0.0)
    np.testing.assert_allclose(
        jdem.utils.compute_potential_energy(actual, system),
        0.5 * 10 * 0.15**2,
        atol=2e-6,
    )


def test_intersecting_facets_follow_the_reciprocal_normal_convention():
    state = jdem.State.create(dim=2)
    for vertices, thickness in [
        ([[-1.0, 0.0], [1.0, 0.0]], 0.1),
        ([[0.0, -1.0], [0.0, 1.0]], 0.2),
    ]:
        state = jdem.State.add_facet(state, jnp.asarray(vertices), thickness=thickness)
    table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=10.0, poisson=0.25, density=1.0)]
    )
    system = jdem.System.create(
        state=state, mat_table=table, force_model_type="facet_facet_spring"
    )
    actual, system = jdem.System.evaluate_forces(state, system)
    np.testing.assert_allclose(
        actual.force, [[0, 3], [0, 3], [0, -3], [0, -3]], atol=2e-6
    )
    np.testing.assert_allclose(actual.torque, 0, atol=2e-6)
    np.testing.assert_allclose(
        jdem.utils.compute_potential_energy(actual, system), 0.45, atol=2e-6
    )
