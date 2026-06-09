# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for rigid and deformable facet use cases."""

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dim", [2, 3])
def test_facet_rigid_vs_deformable(dim):
    # 1. Set up a single facet and a sphere
    state = jdem.State.create()

    # Define vertices for a segment (2D) or triangle (3D)
    if dim == 2:
        vertices = jnp.array([[0.0, 0.0], [2.0, 0.0]])
        sphere_pos = jnp.array(
            [[1.5, 0.1]]
        )  # Overlapping the segment at barycentric coords [0.25, 0.75]
    else:
        vertices = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        # Overlapping the triangle at barycentric coords [0.25, 0.5, 0.25]
        sphere_pos = jnp.array([[0.5, 1.0, 0.1]])

    # Add the facet
    state = jdem.State.add_facet(
        state,
        vertices,
        thickness=0.1,
        species_id=jnp.array([1]),
    )

    # Add the sphere
    sphere_state = jdem.State.create(
        pos=sphere_pos,
        rad=jnp.array([0.2]),
        mass=jnp.array([1.0]),
        species_id=jnp.array([0]),
    )
    state = jdem.State.merge(state, sphere_state)

    # Set up material table and force router
    mat = jdem.Material.create("elastic", young=1e4, poisson=0.3, density=1.0)
    mat_table = jdem.MaterialTable.from_materials([mat])

    router = jdem.ForceRouter.from_dict(
        S=2,
        mapping={
            (0, 0): jdem.ForceModel.create("spring"),
            (1, 0): jdem.ForceModel.create("sphere_facet_spring", thickness=0.1),
            (1, 1): jdem.ForceModel.create("facet_facet_spring", thickness=0.1),
        },
    )

    # Create the system with naive collider (CPU friendly)
    system = jdem.System.create(
        state.shape,
        domain_type="free",
        collider_type="Naive",
        dt=0.001,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )

    # -------------------------------------------------------------------------
    # TEST CASE 1: RIGID SINGLE FACET
    # -------------------------------------------------------------------------
    # By default, vertices of the facet share the same clump_id
    state_rigid, _ = system.collider.compute_force(state, system)
    state_rigid, _ = system.force_manager.apply(state_rigid, system)

    # Sphere is the last particle
    f_sphere_rigid = state_rigid.force[-1]
    # Vertices are the first dim particles
    f_vertices_rigid = state_rigid.force[:dim]

    # Verify Newton's third law for total force on the rigid body
    total_rigid_facet_force = f_vertices_rigid[0]
    np.testing.assert_allclose(total_rigid_facet_force, -f_sphere_rigid, atol=1e-12)

    # -------------------------------------------------------------------------
    # TEST CASE 3: DEFORMABLE FACET
    # -------------------------------------------------------------------------
    # Create the deformable facet using the new rigid=False parameter
    state_deformable = jdem.State.create()
    state_deformable = jdem.State.add_facet(
        state_deformable,
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

    # Sphere force should be exactly the same in both cases (due to partition of unity)
    np.testing.assert_allclose(f_sphere_def, f_sphere_rigid, atol=1e-11)

    # Sum of forces on deformable vertices must equal the sphere force (equal and opposite)
    total_def_facet_force = jnp.sum(f_vertices_def, axis=0)
    np.testing.assert_allclose(total_def_facet_force, -f_sphere_def, atol=1e-12)

    # Verify barycentric distribution of forces
    # Vertices: 0 is A, 1 is B, 2 is C
    if dim == 2:
        expected_weights = jnp.array([0.25, 0.75])
    else:
        expected_weights = jnp.array([0.25, 0.25, 0.5])

    for idx in range(dim):
        weight = expected_weights[idx]
        np.testing.assert_allclose(
            f_vertices_def[idx], -f_sphere_def * weight, atol=1e-12
        )

    # Verify that torques on all vertices of deformable facet are zero
    np.testing.assert_allclose(state_deformable.torque[:dim], 0.0, atol=1e-12)

    # Verify energy partition
    _, _, energy_rigid = system.collider.compute_potential_energy(state, system)
    _, _, energy_deformable = system.collider.compute_potential_energy(
        state_deformable, system
    )
    np.testing.assert_allclose(energy_deformable, energy_rigid, atol=1e-12)


def test_clump_of_rigid_facets():
    # TEST CASE 2: CLUMPS OF RIGID TRIANGLES
    # Create two facets that share the same clump_id (forming a single arbitrary shaped rigid body)
    state = jdem.State.create()

    # Add facet 1
    state = jdem.State.add_facet(
        state,
        jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        thickness=0.1,
        species_id=jnp.array([1]),
    )

    # Add facet 2 (welded to the first one)
    state = jdem.State.add_facet(
        state,
        jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        thickness=0.1,
        species_id=jnp.array([1]),
    )

    # Weld them by setting their clump_id to 0
    welded_clump_id = jnp.zeros(state.N, dtype=int)
    state = dataclasses.replace(state, clump_id=welded_clump_id)

    # Add a sphere overlapping facet 1
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
            (1, 0): jdem.ForceModel.create("sphere_facet_spring", thickness=0.1),
            (1, 1): jdem.ForceModel.create("facet_facet_spring", thickness=0.1),
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

    # All welded vertices should have the same total force and torque broadcasted from the clump COM
    f_vertices = state.force[:6]
    t_vertices = state.torque[:6]

    for idx in range(1, 6):
        np.testing.assert_allclose(f_vertices[idx], f_vertices[0], atol=1e-12)
        np.testing.assert_allclose(t_vertices[idx], t_vertices[0], atol=1e-12)


def test_facet_multicell_list_invariance():
    # Set up a facet and a sphere
    state = jdem.State.create()

    # We will use 3D
    vertices = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    sphere_pos = jnp.array([[0.5, 1.0, 0.1]])

    state = jdem.State.add_facet(
        state,
        vertices,
        thickness=0.1,
        species_id=jnp.array([1]),
        safety_factor=1.2,
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
            (1, 0): jdem.ForceModel.create("sphere_facet_spring", thickness=0.1),
            (1, 1): jdem.ForceModel.create("facet_facet_spring", thickness=0.1),
        },
    )

    # System with Naive collider
    system_naive = jdem.System.create(
        state.shape,
        domain_type="free",
        collider_type="Naive",
        dt=0.001,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )

    # System with MultiCellList collider
    system_mcl = jdem.System.create(
        state.shape,
        domain_type="free",
        collider_type="MultiCellList",
        collider_kw={"state": state, "cell_size": 1.0, "max_hashes": 27},
        dt=0.001,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )

    # System with CellList collider
    system_cl = jdem.System.create(
        state.shape,
        domain_type="free",
        collider_type="CellList",
        collider_kw={"state": state, "cell_size": 1.0},
        dt=0.001,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )

    # Apply domain to all systems first
    state_naive, system_naive = system_naive.domain.apply(state, system_naive)
    state_mcl, system_mcl = system_mcl.domain.apply(state, system_mcl)
    state_cl, system_cl = system_cl.domain.apply(state, system_cl)

    # Compute forces
    state_naive_res, _ = system_naive.collider.compute_force(state_naive, system_naive)
    state_mcl_res, _ = system_mcl.collider.compute_force(state_mcl, system_mcl)
    state_cl_res, _ = system_cl.collider.compute_force(state_cl, system_cl)

    # Sort states by unique ID to compare sorted vs unsorted colliders
    def sort_by_uid(s):
        idx = jnp.argsort(s.unique_id)
        return jax.tree.map(lambda x: x[idx], s)

    state_naive_sorted = sort_by_uid(state_naive_res)
    state_mcl_sorted = sort_by_uid(state_mcl_res)
    state_cl_sorted = sort_by_uid(state_cl_res)

    np.testing.assert_allclose(
        state_mcl_sorted.force, state_naive_sorted.force, atol=1e-12
    )
    np.testing.assert_allclose(
        state_mcl_sorted.torque, state_naive_sorted.torque, atol=1e-12
    )
    np.testing.assert_allclose(
        state_cl_sorted.force, state_naive_sorted.force, atol=1e-12
    )
    np.testing.assert_allclose(
        state_cl_sorted.torque, state_naive_sorted.torque, atol=1e-12
    )

    # Verify potential energy
    _, _, pe_naive = system_naive.collider.compute_potential_energy(state_naive, system_naive)
    _, _, pe_mcl = system_mcl.collider.compute_potential_energy(state_mcl, system_mcl)
    _, _, pe_cl = system_cl.collider.compute_potential_energy(state_cl, system_cl)
    np.testing.assert_allclose(pe_mcl, pe_naive, atol=1e-12)
    np.testing.assert_allclose(pe_cl, pe_naive, atol=1e-12)


def test_reflect_domain_with_facets():
    # 1. Set up a state with one moving rigid facet and one fixed rigid facet crossing the boundary
    state = jdem.State.create()

    # Moveable rigid facet (species 0)
    vertices_mov = jnp.array([[0.05, 0.5, 0.5], [0.9, 0.5, 0.5], [0.5, 0.9, 0.5]])
    state = jdem.State.add_facet(
        state,
        vertices_mov,
        thickness=0.1,
        vel=jnp.array([-10.0, 0.0, 0.0]),  # moving towards x = 0 wall
        fixed=False,
        species_id=0,
    )

    # Fixed rigid facet (species 1) placed near/crossing the x = 0 wall
    vertices_fix = jnp.array([[-0.1, 1.5, 1.5], [0.5, 1.5, 1.5], [0.2, 1.9, 1.5]])
    state = jdem.State.add_facet(
        state,
        vertices_fix,
        thickness=0.1,
        vel=jnp.array([0.0, 0.0, 0.0]),
        fixed=True,
        species_id=1,
    )

    # Set up reflect system
    mat = jdem.Material.create("elastic", young=1e4, poisson=0.3, density=1.0)
    mat_table = jdem.MaterialTable.from_materials([mat])

    system = jdem.System.create(
        state.shape,
        domain_type="reflect",
        domain_kw={
            "box_size": (2.0, 2.0, 2.0),
            "anchor": (0.0, 0.0, 0.0),
        },
        collider_type="Naive",
        dt=0.001,
        mat_table=mat_table,
    )

    # Apply reflect domain
    state_new, system = system.domain.apply(state, system)

    # Assertions:
    # 1. The moving facet vertices have been pushed back and their velocity is reversed (since vel_x was negative, it should be positive now).
    # Vertices of moving facet are 0, 1, 2.
    assert np.all(state_new.vel[0:3, 0] > 0.0)
    assert np.all(state_new.pos_c[0:3, 0] > 0.0)

    # 2. The fixed facet vertices (indices 3, 4, 5) must NOT have moved (positions and velocities remain unchanged)
    np.testing.assert_allclose(state_new.pos_c[3:6], state.pos_c[3:6])
    np.testing.assert_allclose(state_new.vel[3:6], 0.0)

