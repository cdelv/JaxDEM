# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Focused contracts for low-level rigid/contact construction."""

import jax.numpy as jnp
import pytest
import numpy as np

import jaxdem as jdem


def test_add_clump_defaults_then_geometry_utility_updates_body_properties():
    state = jdem.State.create(dim=2)
    state = jdem.State.add_clump(
        state, pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]), rad=jnp.full(2, 0.5)
    )
    assert state.N == 2
    table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=1.0, young=1.0e4, poisson=0.3)]
    )
    updated = jdem.utils.compute_clump_properties(state, table, n_samples=32768)
    # Two tangent unit-density disks: union area=pi/2, COM=(1/2,0),
    # I=M*(r^2/2+(d/2)^2)=3*pi/16 by the parallel-axis theorem.
    # Fixed 1-2% geometric integration budgets at 32768 deterministic samples.
    np.testing.assert_allclose(updated.mass, np.full(2, np.pi / 2), rtol=0.01)
    np.testing.assert_allclose(updated.volume, np.full(2, np.pi / 2), rtol=0.01)
    np.testing.assert_allclose(updated.pos_c, [[0.5, 0.0]] * 2, atol=0.01)
    np.testing.assert_allclose(
        updated.inertia[:, 0], np.full(2, 3 * np.pi / 16), rtol=0.02
    )
    np.testing.assert_allclose(updated.pos, state.pos, atol=2e-6)


def test_contact_facets_reject_shared_vertices():
    state = jdem.State.create(pos=jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    state = jdem.State.add_connected_facet(state, [0, 1], rigid=False)
    with pytest.raises(ValueError, match="cannot share vertices"):
        jdem.State.add_connected_facet(state, [1, 2], rigid=False)
