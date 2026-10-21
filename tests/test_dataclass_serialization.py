# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Data containers retain their numerical contents through Python serialization."""

import copy
from dataclasses import replace
import pickle
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.analysis import LagBinsExact, evaluate_binned


def _pickle_roundtrip(value):
    return pickle.loads(pickle.dumps(value))


@pytest.mark.parametrize("restore", [copy.deepcopy, _pickle_roundtrip])
def test_analysis_result_serialization_preserves_pairs_and_statistics(restore):
    result = evaluate_binned(
        lambda arrays, i, j: arrays["x"][j] - arrays["x"][i],
        {"x": jnp.arange(5.0) ** 2},
        LagBinsExact(5, [1, 2]),
    )
    restored = restore(result)

    assert weakref.ref(restored)() is restored
    assert weakref.ref(restored.pairs)() is restored.pairs

    np.testing.assert_array_equal(restored.counts, [4, 3])
    np.testing.assert_array_equal(restored.sums, [16.0, 24.0])
    np.testing.assert_array_equal(restored.mean, [4.0, 8.0])
    np.testing.assert_array_equal(restored.pairs.pair_i, [0, 1, 2, 3, 0, 1, 2])
    np.testing.assert_array_equal(restored.pairs.pair_j, [1, 2, 3, 4, 2, 3, 4])
    np.testing.assert_array_equal(restored.pairs.bin_id, [0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(restored.pairs.counts_per_bin, [4, 3])

    updated = replace(restored, mean=2 * restored.mean)
    np.testing.assert_array_equal(updated.mean, [8.0, 16.0])
    np.testing.assert_array_equal(restored.mean, [4.0, 8.0])


@pytest.mark.parametrize("restore", [copy.deepcopy, _pickle_roundtrip])
def test_material_table_serialization_preserves_dynamic_properties_and_jit(restore):
    table = jd.MaterialTable.from_materials(
        [
            jd.Material.create("elastic", young=10.0, poisson=0.3, density=2.0),
            jd.Material.create("elastic", young=30.0, poisson=0.3, density=5.0),
        ],
        matcher=jd.MaterialMatchmaker.create("harmonic"),
    )
    restored = restore(table)

    assert weakref.ref(restored)() is restored

    np.testing.assert_array_equal(restored.young, [10.0, 30.0])
    np.testing.assert_array_equal(restored.density, [2.0, 5.0])
    assert jax.tree.structure(restored) == jax.tree.structure(table)
    for before, after in zip(
        jax.tree.leaves(table), jax.tree.leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(
        jax.jit(lambda material: material.young + material.density)(restored),
        [12.0, 35.0],
    )

    updated = replace(
        restored, props={**restored.props, "density": jnp.array([4.0, 8.0])}
    )
    np.testing.assert_array_equal(updated.density, [4.0, 8.0])
    np.testing.assert_array_equal(restored.density, [2.0, 5.0])


def test_system_copy_keeps_configuration_and_runs_independently():
    state = jd.State.create(pos=jnp.zeros((1, 2)), vel=jnp.array([[1.0, -2.0]]))
    system = jd.System.create(state=state, collider_type="", dt=0.125)
    state, system = jd.System.initialize(state, system)
    assert weakref.ref(system)() is system

    copied = copy.deepcopy(system)
    state, evolved = jd.System.step(state, copied, n=2)
    np.testing.assert_allclose(state.pos, [[0.25, -0.5]])
    np.testing.assert_allclose(evolved.time, 0.25)
    np.testing.assert_allclose(system.time, 0.0)
    updated = replace(evolved, dt=jnp.asarray(0.5))
    np.testing.assert_allclose(updated.dt, 0.5)
    np.testing.assert_allclose(evolved.dt, 0.125)


@pytest.mark.parametrize(
    "manager_name",
    [
        "CheckpointWriter",
        "CheckpointLoader",
        "CheckpointModelWriter",
        "CheckpointModelLoader",
    ],
)
def test_checkpoint_managers_preserve_inherited_configuration_and_context(
    tmp_path, manager_name
):
    pytest.importorskip("orbax.checkpoint")
    from jaxdem.writers import checkpoints

    directory = tmp_path / manager_name
    directory.mkdir()
    manager_type = getattr(checkpoints, manager_name)
    options = (
        {"save_every": "2", "max_to_keep": "3"} if "Writer" in manager_name else {}
    )
    with manager_type(directory=directory, **options) as manager:
        assert weakref.ref(manager)() is manager
        assert manager.directory == directory.resolve()
        assert manager.checkpointer is not None
        if "Writer" in manager_name:
            assert manager.save_every == 2
            assert manager.max_to_keep == 3
