# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Focused contracts for low-level rigid/contact construction."""

from dataclasses import replace
import importlib

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


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("rigid", [False, True])
def test_distribution_rotates_a_whole_body_and_uses_its_bounding_volume(dim, rigid):
    from jaxdem.utils.particle_creation import distribute_bodies
    from scipy.spatial.transform import Rotation

    points = np.pad(
        np.array([[-1.0, 0.0], [0.5, 0.6], [0.5, -0.6]]), ((0, 0), (0, dim - 2))
    )
    if rigid:
        state = jdem.State.create(
            pos=jnp.zeros((3, dim)),
            pos_p=jnp.asarray(points),
            clump_id=jnp.zeros(3, dtype=int),
            rad=jnp.full(3, 0.1),
            mass=jnp.full(3, 2.0),
            volume=jnp.full(3, 1.2),
        )
        state = replace(state, clump_id=jnp.ones(3, dtype=int))
    else:
        state = jdem.State.create(
            pos=jnp.asarray(points),
            bond_id=[[1, 2], [0, 2], [0, 1]],
            rad=jnp.full(3, 0.1),
            mass=jnp.full(3, 2 / 3),
            volume=jnp.full(3, 0.4),
        )
    placed, box = distribute_bodies(state, phi=0.1, seed=27)
    expected_volume = (np.pi if dim == 2 else 4 * np.pi / 3) * 1.1**dim / 0.1
    np.testing.assert_allclose(np.prod(box), expected_volume, rtol=2e-6)
    center = np.mean(np.asarray(placed.pos), axis=0)
    assert np.all((center >= 0) & (center <= box))
    quat = np.r_[np.asarray(placed.q.w[0]), np.asarray(placed.q.xyz[0])]
    rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()[:dim, :dim]
    np.testing.assert_allclose(placed.pos, points @ rotation.T + center, atol=2e-6)
    for name in ("mass", "volume", "inertia", "clump_id", "bond_id"):
        np.testing.assert_array_equal(getattr(placed, name), getattr(state, name))


def test_deformable_builder_assigns_heterogeneous_coefficients_to_whole_bodies():
    from jaxdem.utils.particle_creation import create_dp_container

    points = jnp.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [5.0, 1.0],
        ]
    )
    state = jdem.State.create(
        pos=points,
        bond_id=[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2], [5, 6], [4, 6], [4, 5]],
    )
    model = create_dp_container(
        state,
        ec=jnp.array([3.0, 7.0]),
        em=jnp.array([2.0, 5.0]),
        gamma=jnp.array([0.2, 0.3]),
        el=jnp.array([11.0, 13.0]),
        eb=jnp.array([17.0, 19.0]),
    )
    np.testing.assert_array_equal(model.ec, [3.0, 7.0])
    np.testing.assert_array_equal(np.bincount(model.elements_id), [4, 3])
    np.testing.assert_array_equal(model.em, np.array([2.0, 5.0])[model.elements_id])
    np.testing.assert_allclose(model.gamma, np.array([0.2, 0.3])[model.elements_id])
    np.testing.assert_array_equal(model.el, np.where(model.edges[:, 0] < 4, 11.0, 13.0))
    np.testing.assert_array_equal(
        model.eb,
        np.array([17.0, 19.0])[model.elements_id[model.element_adjacency[:, 0]]],
    )
    current = (
        points.at[:4]
        .multiply(1.1)
        .at[4:]
        .set((points[4:] - jnp.array([5.0, 0.0])) * 1.3 + jnp.array([5.0, 0.0]))
    )
    system = jdem.System.create(state=state, collider_type="", bonded_force_model=model)
    scales, contents, perimeters = (
        np.array([1.1, 1.3]),
        np.array([2.0, 0.5]),
        np.array([6.0, 2 + np.sqrt(2)]),
    )
    expected = np.sum(
        0.5 * np.array([3.0, 7.0]) * contents * (scales**2 - 1) ** 2
        + 0.5 * np.array([2.0, 5.0]) * perimeters * (scales - 1) ** 2
        + np.array([0.2, 0.3]) * perimeters * scales
        + 0.5 * np.array([11.0, 13.0]) * np.array([10.0, 4.0]) * (scales - 1) ** 2
    )
    np.testing.assert_allclose(
        model.compute_potential_energy(current, state, system), expected, rtol=2e-6
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_single_sphere_builder_preserves_geometry_and_target_fraction(dim):
    from jaxdem.utils.particle_creation import build_sphere_system

    state, system = build_sphere_system(
        [0.5], phi=0.1, dim=dim, seed=0, max_n_min_steps_per_outer=0
    )
    volume = np.pi / 4 if dim == 2 else np.pi / 6
    assert state.pos.shape == (1, dim)
    np.testing.assert_allclose(np.prod(system.domain.box_size), volume / 0.1, rtol=2e-6)
    np.testing.assert_array_equal(state.force, np.zeros((1, dim)))
    np.testing.assert_allclose(jdem.utils.compute_packing_fraction(state, system), 0.1)


@pytest.mark.parametrize("builder", ["sphere", "ga"])
def test_packing_builders_report_failed_mechanical_relaxation(monkeypatch, builder):
    from jaxdem.utils import particle_creation as creation

    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [0.9, 0.0]]), rad=jnp.full(2, 0.5)
    )
    box = jnp.full(2, 3.0)
    random_module = importlib.import_module("jaxdem.utils.random_sphere_configuration")
    monkeypatch.setattr(
        random_module, "random_sphere_configuration", lambda **kw: (state.pos, box)
    )
    monkeypatch.setattr(creation, "create_ga_state", lambda **kw: state)
    monkeypatch.setattr(creation, "distribute_bodies", lambda *args, **kw: (state, box))
    with pytest.raises(RuntimeError, match="MINIMIZATION_FAILED.*MAX_STEPS"):
        if builder == "sphere":
            creation.build_sphere_system(
                [0.5, 0.5], phi=0.5, dim=2, seed=0, max_n_min_steps_per_outer=0
            )
        else:
            creation.build_ga_system(
                [0.5, 0.5],
                [1, 1],
                0.5,
                phi=0.5,
                dim=2,
                seed=0,
                max_n_min_steps_per_outer=0,
            )


def test_initial_sphere_relaxation_does_not_hide_failure(monkeypatch):
    from jaxdem.utils.random_sphere_configuration import minimize_sphere_configuration

    minimize = jdem.System.minimize

    def limited(state, system, **kw):
        return minimize(state, system, max_steps=0, force_tol=kw["force_tol"])

    monkeypatch.setattr(jdem.System, "minimize", staticmethod(limited))
    with pytest.raises(RuntimeError, match="Sphere relaxation failed.*MAX_STEPS"):
        minimize_sphere_configuration(
            [0.5, 0.5], [[0.0, 0.0], [0.9, 0.0]], phi=0.1, dim=2
        )
