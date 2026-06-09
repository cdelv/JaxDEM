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
    state1 = jdem.State.add_facet(state, vertices, thickness=0.1, rigid=True, safety_factor=1.0)
    
    # 2. safety_factor = 1.5
    state2 = jdem.State.add_facet(state, vertices, thickness=0.1, rigid=True, safety_factor=1.5)
    
    # Compare search radii _rad
    assert np.allclose(state2._rad, state1._rad * 1.5)

    # 3. Flexible facet with safety_factor = 2.0
    state3 = jdem.State.add_facet(state, vertices, thickness=0.1, rigid=False, safety_factor=2.0)
    state4 = jdem.State.add_facet(state, vertices, thickness=0.1, rigid=False, safety_factor=1.0)
    assert np.allclose(state3._rad, state4._rad * 2.0)


def test_add_mesh_rigid_and_flexible():
    state = jdem.State.create()
    
    # A simple tetrahedron mesh (4 vertices, 4 faces)
    vertices = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    faces = jnp.array([
        [0, 2, 1], # bottom face
        [0, 1, 3], # front face
        [0, 3, 2], # side face
        [1, 2, 3]  # diagonal face
    ])
    
    # Rigid filled mesh
    state_rigid = jdem.State.add_mesh(
        state, vertices, faces,
        thickness=0.1,
        rigid=True,
        filled=True,
        mass=4.0
    )
    
    # Rigid hollow mesh
    state_hollow = jdem.State.add_mesh(
        state, vertices, faces,
        thickness=0.1,
        rigid=True,
        filled=False,
        mass=4.0
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
        state, vertices, faces,
        thickness=0.1,
        rigid=False,
        mass=4.0
    )
    assert state_flex.N == 12
    # Check that flexible mesh vertices do not share clump_ids
    assert np.unique(state_flex.clump_id).size == 12
