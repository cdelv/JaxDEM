# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""A force law's configured cutoff governs physics, search, and restart."""

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd


def _system(collider="Naive", cutoff_ratio=2.5, composition="single"):
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [3.2, 0.0]]), rad=jnp.full(2, 0.5)
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("lj", density=1.0, epsilon=1.0)]
    )
    law = jd.ForceModel.create("lennardjones", cutoff_ratio=cutoff_ratio)
    if composition == "combiner":
        law = jd.LawCombiner(laws=(law,))
    elif composition == "router":
        law = jd.ForceRouter.from_dict(1, {(0, 0): law})
    kw = {}
    if collider in ("CellList", "MultiCellList"):
        kw = {"cell_size": 0.25}
    elif collider == "NeighborList":
        kw = {"cutoff": 0.5, "skin": 0.1, "max_neighbors": 4}
    system = jd.System.create(
        state=state,
        force_model=law,
        mat_table=table,
        collider_type=collider,
        collider_kw=kw,
    )
    return state, system


@pytest.mark.parametrize(
    "collider,composition",
    [
        ("Naive", "single"),
        ("CellList", "single"),
        ("MultiCellList", "single"),
        ("NeighborList", "single"),
        ("NeighborList", "combiner"),
        ("NeighborList", "router"),
    ],
)
def test_configured_cutoff_changes_force_energy_and_search(collider, composition):
    state, default = _system(collider, composition=composition)
    inactive, _ = default.collider.compute_force(state, default)
    np.testing.assert_array_equal(inactive.force, 0.0)

    state, system = _system(collider, cutoff_ratio=4.0, composition=composition)
    actual, updated = system.collider.compute_force(state, system)
    _, _, energy = updated.collider.compute_potential_energy(state, updated)
    distance = 3.2
    expected_x = -24.0 / distance * (2.0 / distance**12 - 1.0 / distance**6)
    expected_energy = 4.0 * (distance**-12 - distance**-6 - 4.0**-12 + 4.0**-6)
    np.testing.assert_allclose(actual.force[0], [expected_x, 0.0], rtol=2e-6)
    np.testing.assert_allclose(jnp.sum(energy), expected_energy, rtol=2e-6)
    assert not bool(updated.collider.overflow)


def test_wca_search_covers_physical_cutoff_beyond_diameter():
    # WCA cutoff is 2^(1/6) * (r_i + r_j) ~= 1.1225 > 1.0.
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [1.08, 0.0]]), rad=jnp.full(2, 0.5)
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("lj", density=1.0, epsilon=1.0)]
    )
    naive = jd.System.create(state=state, force_model_type="wca", mat_table=table)
    expected, _ = naive.collider.compute_force(state, naive)
    assert np.linalg.norm(expected.force) > 0.0
    spatial = replace(
        naive,
        collider=jd.Collider.create(
            "NeighborList", state=state, cutoff=0.5, skin=0.1, max_neighbors=4
        ),
    )
    actual, updated = spatial.collider.compute_force(state, spatial)
    np.testing.assert_allclose(actual.force, expected.force, rtol=1e-6, atol=1e-6)
    assert not bool(updated.collider.overflow)


def test_neighbor_cache_rebuilds_when_law_cutoff_changes():
    state, system = _system("NeighborList")
    _, cached = system.collider.compute_force(state, system)
    changed = replace(
        cached, force_model=replace(cached.force_model, cutoff_ratio=jnp.asarray(4.0))
    )
    result, rebuilt = changed.collider.compute_force(state, changed)
    distance = 3.2
    expected_x = -24.0 / distance * (2.0 / distance**12 - 1.0 / distance**6)
    np.testing.assert_allclose(result.force[0], [expected_x, 0.0], rtol=2e-6)
    assert int(rebuilt.collider.n_build_times) == int(cached.collider.n_build_times) + 1
    assert not bool(rebuilt.collider.overflow)


def test_cutoff_is_dynamic_under_vmap():
    state, system = _system("CellList", composition="combiner")

    def evaluate(cutoff):
        law = replace(system.force_model.laws[0], cutoff_ratio=cutoff)
        configured = replace(
            system, force_model=replace(system.force_model, laws=(law,))
        )
        return configured.collider.compute_force(state, configured)[0].force

    forces = jax.jit(jax.vmap(evaluate))(jnp.array([2.5, 4.0]))
    distance = 3.2
    expected_x = -24.0 / distance * (2.0 / distance**12 - 1.0 / distance**6)
    np.testing.assert_array_equal(forces[0], 0.0)
    np.testing.assert_allclose(forces[1, 0], [expected_x, 0.0], rtol=2e-6)


@pytest.mark.parametrize("composition", ["single", "router"])
def test_configured_cutoff_survives_checkpoint(tmp_path, composition):
    state, system = _system("NeighborList", cutoff_ratio=4.0, composition=composition)
    state, system = system.collider.compute_force(state, system)
    path = str(tmp_path / "checkpoint")
    with jd.CheckpointWriter(path) as writer:
        writer.save(state, system)
    with jd.CheckpointLoader(path) as loader:
        restored_state, restored = loader.load()
    np.testing.assert_allclose(
        restored.force_model.search_radii(restored_state, restored), 2.0
    )
    actual, _ = restored.collider.compute_force(restored_state, restored)
    np.testing.assert_allclose(actual.force, state.force, rtol=2e-6)


def test_near_coincident_hertz_contact_uses_unit_normal():
    separation = np.array([3.0e-10, 4.0e-10])
    distance = 5.0e-10
    normal = np.array([-0.6, -0.8])
    state = jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], separation]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    young, poisson = 12.0, 0.25
    material = jd.Material.create("elastic", density=1.0, young=young, poisson=poisson)
    law = jd.forces.HertzianForce()
    system = jd.System.create(
        state=state,
        force_model=law,
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="naive",
    )
    force, _, _ = law.force(
        0, 1, state.pos, state, system, law.init_history((), state.dim)
    )
    effective_young = young / (2.0 * (1.0 - poisson**2))
    overlap = 1.0 - distance
    magnitude = (4.0 / 3.0) * effective_young * overlap * np.sqrt(0.25 * overlap)
    np.testing.assert_allclose(force, magnitude * normal, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("overlap", [0.3, 0.0, -0.2])
def test_unequal_material_hertz_force_energy_and_jacobian(overlap):
    radii = np.array([0.4, 1.2])
    young, poisson = np.array([100.0, 250.0]), np.array([0.2, 0.35])
    direction = np.array([0.6, 0.8])
    distance = radii.sum() - overlap
    state = jd.State.create(
        pos=jnp.asarray([[0, 0], distance * direction]),
        rad=jnp.asarray(radii),
        mat_id=jnp.arange(2),
    )
    table = jd.MaterialTable.from_materials(
        [
            jd.Material.create("elastic", young=e, poisson=nu, density=1.0)
            for e, nu in zip(young, poisson, strict=True)
        ]
    )
    system = jd.System.create(state=state, mat_table=table, force_model_type="hertz")
    law, history = system.force_model, jnp.zeros(0)
    modulus = 1 / np.sum((1 - poisson**2) / young)
    radius = 1 / np.sum(1 / radii)
    k = 4 / 3 * modulus * np.sqrt(radius)
    delta = max(overlap, 0)
    magnitude = k * delta**1.5
    expected_force = -magnitude * direction
    expected_jacobian = (
        magnitude / distance * (np.eye(2) - np.outer(direction, direction))
    )
    expected_jacobian -= 1.5 * k * np.sqrt(delta) * np.outer(direction, direction)

    def force(position):
        return law.force(0, 1, state.pos.at[0].set(position), state, system, history)[0]

    energy = lambda pos: law.energy(0, 1, pos, state, system, history)
    np.testing.assert_allclose(force(state.pos[0]), expected_force, atol=2e-6)
    np.testing.assert_allclose(energy(state.pos), 0.4 * k * delta**2.5, atol=2e-6)
    np.testing.assert_allclose(
        -jax.grad(energy)(state.pos)[0], expected_force, atol=2e-6
    )
    np.testing.assert_allclose(
        jax.jacrev(force)(state.pos[0]), expected_jacobian, atol=2e-5
    )


@pytest.mark.parametrize("name", ["wca", "wcashifted"])
@pytest.mark.parametrize("fraction", [0.9, 1.0, 1.1])
def test_wca_values_and_energy_derivative_at_the_physical_cutoff(name, fraction):
    cutoff = 2 ** (1 / 6) if name == "wca" else 1.0
    distance, epsilon = cutoff * fraction, 0.7
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [distance, 0.0]]), rad=jnp.array([0.25, 0.75])
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("lj", epsilon=epsilon, density=1)]
    )
    system = jd.System.create(state=state, mat_table=table, force_model_type=name)
    law, history = system.force_model, jnp.zeros(0)
    expected_energy = expected_magnitude = 0.0
    if fraction < 1:
        expected_energy = 4 * epsilon * (distance**-12 - distance**-6)
        expected_magnitude = 24 * epsilon * (2 * distance**-13 - distance**-7)
        if name == "wca":
            expected_energy += epsilon
        else:
            expected_energy += (distance - 1) * 24 * epsilon
            expected_magnitude -= 24 * epsilon
    energy = lambda pos: law.energy(0, 1, pos, state, system, history)
    expected_force = [-expected_magnitude, 0]
    np.testing.assert_allclose(
        law.force(0, 1, state.pos, state, system, history)[0], expected_force, atol=2e-5
    )
    np.testing.assert_allclose(energy(state.pos), expected_energy, atol=2e-6)
    np.testing.assert_allclose(
        -jax.grad(energy)(state.pos)[0], expected_force, atol=2e-5
    )
