# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""A force law's configured cutoff governs physics, search, and restart."""

from dataclasses import replace

import jax
import jax.numpy as jnp
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
    "collider", ["Naive", "CellList", "MultiCellList", "NeighborList"]
)
@pytest.mark.parametrize("composition", ["single", "combiner", "router"])
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


def test_neighbor_cache_rebuilds_when_law_cutoff_changes():
    state, system = _system("NeighborList")
    _, cached = system.collider.compute_force(state, system)
    changed = replace(
        cached, force_model=replace(cached.force_model, cutoff_ratio=jnp.asarray(4.0))
    )
    result, rebuilt = changed.collider.compute_force(state, changed)
    assert float(jnp.linalg.norm(result.force)) > 0.0
    assert not bool(rebuilt.collider.overflow)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_cutoff_is_dynamic_under_vmap(batch_size):
    state, system = _system("CellList", composition="combiner")

    def evaluate(cutoff):
        law = replace(system.force_model.laws[0], cutoff_ratio=cutoff)
        configured = replace(
            system, force_model=replace(system.force_model, laws=(law,))
        )
        return configured.collider.compute_force(state, configured)[0].force

    cutoffs = jnp.array([4.0] if batch_size == 1 else [2.5, 4.0])
    forces = jax.jit(jax.vmap(evaluate))(cutoffs)
    assert float(jnp.linalg.norm(forces[-1])) > 0.0
    if batch_size == 2:
        np.testing.assert_array_equal(forces[0], 0.0)


@pytest.mark.parametrize("composition", ["single", "combiner", "router"])
def test_configured_cutoff_survives_checkpoint(tmp_path, composition):
    state, system = _system("NeighborList", cutoff_ratio=4.0, composition=composition)
    state, system = system.collider.compute_force(state, system)
    path = str(tmp_path / "checkpoint")
    with jd.CheckpointWriter(path) as writer:
        writer.save(state, system)
    restored_state, restored = jd.CheckpointLoader(path).load()
    np.testing.assert_allclose(
        restored.force_model.search_radii(restored_state, restored), 2.0
    )
    actual, _ = restored.collider.compute_force(restored_state, restored)
    np.testing.assert_allclose(actual.force, state.force, rtol=2e-6)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_cutoff_rejected(cutoff):
    with pytest.raises(ValueError, match="cutoff_ratio"):
        jd.ForceModel.create("lennardjones", cutoff_ratio=cutoff)
