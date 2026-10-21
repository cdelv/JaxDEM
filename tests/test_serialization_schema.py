"""Shared Orbax/HDF5 serialization schema contracts."""

import json

import h5py
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
import numpy as np

import jaxdem as jd
from jaxdem.utils import h5
from jaxdem.utils.serialization_schema import (
    make_serialization_manifest,
    serialization_schema_version,
)
from tests.custom_forces import harmonic_trap, harmonic_trap_energy


def test_hdf5_nested_object_tree_preserves_python_types_and_exact_arrays(tmp_path):
    payload = {
        "scalars": [None, True, 17, 1.25, "mu=μ\n", np.float32(0.3), np.int16(7)],
        "empty": ({}, [], (), jnp.empty((0, 3))),
        "arrays": (
            jnp.asarray(2.0, dtype=jnp.float32),
            jnp.array([1, -3], dtype=jnp.int32),
        ),
        "callback": harmonic_trap,
    }
    path = str(tmp_path / "objects.h5")
    h5.save(payload, path)
    restored = h5.load(path)

    def compare(expected, actual):
        if isinstance(expected, (jax.Array, np.ndarray)):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            assert type(actual) is dict and actual.keys() == expected.keys()
            for key in expected:
                compare(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            assert type(actual) is type(expected) and len(actual) == len(expected)
            for left, right in zip(expected, actual, strict=True):
                compare(left, right)
        elif callable(expected):
            assert actual is expected
        else:
            assert type(actual) is type(expected) and actual == expected

    compare(payload, restored)


@pytest.mark.parametrize("payload", [{1: "non-string key"}, {1, 2}, object()])
def test_hdf5_rejects_unsupported_object_tree_nodes(tmp_path, payload):
    with pytest.raises(TypeError):
        h5.save(payload, str(tmp_path / "unsupported.h5"))


def test_independently_authored_legacy_state_field_names(tmp_path):
    # This fixture is authored in the historical node layout, not current writer output.
    path = str(tmp_path / "legacy-state.h5")
    with h5py.File(path, "w") as file:
        root = file.create_group("root")
        root.attrs.update(__kind__="dataclass", __class__="jaxdem.state:State")
        for name, values in {
            "pos_c": [[0.2, 0.3], [1.2, 1.3]],
            "angVel": [[0.4], [-0.7]],
            "clump_ID": [0, 1],
            "deformable_ID": [[1], [0]],
        }.items():
            root.create_dataset(name, data=np.asarray(values))
    with pytest.warns(RuntimeWarning, match="missing saved fields"):
        restored = h5.load(path)
    np.testing.assert_array_equal(restored.pos, [[0.2, 0.3], [1.2, 1.3]])
    np.testing.assert_array_equal(restored.ang_vel, [[0.4], [-0.7]])
    np.testing.assert_array_equal(restored.clump_id, [0, 1])
    np.testing.assert_array_equal(restored.bond_id, [[1], [0]])
    np.testing.assert_array_equal(restored.mass, [1.0, 1.0])


@pytest.mark.parametrize("python_type,value", [(int, 7), (float, 0.25), (bool, True)])
def test_independently_authored_legacy_scalars(tmp_path, python_type, value):
    path = str(tmp_path / "legacy-scalar.h5")
    with h5py.File(path, "w") as file:
        node = file.create_dataset("root", data=value)
        node.attrs["__kind__"] = "scalar"  # Historical scalar nodes lack __pytype__.
    restored = h5.load(path)
    assert type(restored) is python_type and restored == value


def test_shared_manifest_validates_storage_and_rejects_future_schema() -> None:
    manifest = make_serialization_manifest("orbax", "state_system")
    assert serialization_schema_version(manifest, expected_storage="orbax") == 3
    with pytest.raises(RuntimeError, match="expected 'hdf5'"):
        serialization_schema_version(manifest, expected_storage="hdf5")
    manifest["version"] = 4
    with pytest.raises(RuntimeError, match="Unsupported serialization schema version"):
        serialization_schema_version(manifest, expected_storage="orbax")
    with pytest.raises(RuntimeError, match="requires an explicit manifest"):
        serialization_schema_version(None, legacy_version=3, expected_storage="orbax")
    with pytest.raises(ValueError, match="Unsupported orbax serialization payload"):
        make_serialization_manifest("orbax", "custom_object")
    malformed = make_serialization_manifest("orbax", "state_system")
    del malformed["conventions"]["units"]
    with pytest.raises(RuntimeError, match="conventions"):
        serialization_schema_version(malformed, expected_storage="orbax")
    wrong_precision = make_serialization_manifest("orbax", "state_system")
    wrong_precision["jax_enable_x64"] = not wrong_precision["jax_enable_x64"]
    with pytest.raises(RuntimeError, match="precision mode"):
        serialization_schema_version(wrong_precision, expected_storage="orbax")
    missing_precision = make_serialization_manifest("orbax", "state_system")
    del missing_precision["jax_enable_x64"]
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        serialization_schema_version(missing_precision, expected_storage="orbax")


def test_hdf5_rejects_nonportable_callable_payload(tmp_path) -> None:
    def local_callback() -> None:
        return None

    with pytest.raises(TypeError, match="module-level"):
        h5.save(local_callback, str(tmp_path / "local.h5"))


def test_hdf5_schema_preserves_custom_physics_and_continuation(tmp_path) -> None:
    state = jd.State.create(pos=jnp.array([[0.5, 0.0]]))
    system = jd.System.create(
        state.shape,
        force_manager_kw={"force_functions": [(harmonic_trap, harmonic_trap_energy)]},
        dt=0.01,
    )
    state, system = jd.System.initialize(state, system)
    state, system = jd.System.step(state, system)

    path = tmp_path / "continuation.h5"
    h5.save((state, system), str(path))
    with h5py.File(path) as file:
        manifest = json.loads(file.attrs["jaxdem_serialization_manifest"])
    assert manifest["version"] == 3
    assert manifest["storage"] == "hdf5"

    restored_state, restored_system = h5.load(str(path))
    assert restored_system.force_manager.force_functions[0] is harmonic_trap
    assert restored_system.force_manager.energy_functions[0] is harmonic_trap_energy
    expected_state, expected_system = jd.System.step(state, system)
    actual_state, actual_system = jd.System.step(restored_state, restored_system)
    assert jnp.allclose(actual_state.pos, expected_state.pos)
    assert jnp.allclose(actual_state.vel, expected_state.vel)
    assert actual_system.step_count == expected_system.step_count


def test_hdf5_rejects_future_manifest_before_decoding(tmp_path) -> None:
    path = tmp_path / "future.h5"
    h5.save(jnp.ones(2), str(path))
    with h5py.File(path, "r+") as file:
        manifest = json.loads(file.attrs["jaxdem_serialization_manifest"])
        manifest["version"] = 999
        file.attrs["jaxdem_serialization_manifest"] = json.dumps(manifest)
    with pytest.raises(RuntimeError, match="Unsupported serialization schema version"):
        h5.load(str(path))


def test_hdf5_schema_three_rejects_missing_dataclass_fields(tmp_path) -> None:
    path = tmp_path / "incomplete.h5"
    h5.save(jd.State.create(pos=jnp.zeros((1, 2))), str(path))
    with h5py.File(path, "r+") as file:
        del file["root/mass"]
    with pytest.raises(RuntimeError, match="missing=.*mass"):
        h5.load(str(path))


def test_hdf5_nonempty_neighbor_history_continues(tmp_path) -> None:
    from dataclasses import replace

    from tests.test_neighbor_list_lifecycle import RememberingSpring, _neighbor_system

    state, _ = _neighbor_system([[0.0, 0.0], [0.3, 0.0]])
    state = replace(state, fixed=jnp.ones(2, dtype=bool))
    law = RememberingSpring()
    system = jd.System.create(
        state=state,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 3,
            "cutoff": 0.5,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    state, system = jd.System.step(state, system)
    assert jnp.any(system.collider.history != 0)

    path = tmp_path / "history.h5"
    h5.save((state, system), str(path))
    restored_state, restored_system = h5.load(str(path))
    assert jnp.array_equal(restored_system.collider.history, system.collider.history)
    expected = jd.System.step(state, system)
    actual = jd.System.step(restored_state, restored_system)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        assert jnp.array_equal(actual_leaf, expected_leaf)


def test_orbax_writer_emits_shared_schema_three_manifest(tmp_path) -> None:
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]))
    system = jd.System.create(state.shape)
    with jd.CheckpointWriter(tmp_path) as writer:
        writer.save(state, system)

    with jd.CheckpointLoader(tmp_path) as loader:
        metadata = loader.checkpointer.restore(
            0,
            args=ocp.args.Composite(system_metadata=ocp.args.JsonRestore()),
        ).system_metadata
    assert metadata["checkpoint_metadata_version"] == 3
    manifest = metadata["serialization_manifest"]
    assert manifest["storage"] == "orbax"
    assert manifest["payload"] == "state_system"
    # Exercise the public loader's guard, before any missing state data is decoded.
    del metadata["force_model_metadata"]
    damaged = tmp_path / "damaged"
    with ocp.CheckpointManager(damaged) as manager:
        manager.save(
            0,
            args=ocp.args.Composite(
                state_metadata=ocp.args.JsonSave({"shape": [1, 2]}),
                system_metadata=ocp.args.JsonSave(metadata),
            ),
        )
        manager.wait_until_finished()
    with jd.CheckpointLoader(damaged) as loader:
        with pytest.raises(RuntimeError, match="force_model_metadata"):
            loader.load()


@pytest.mark.parametrize("version", [1, 2])
def test_independently_authored_legacy_orbax_metadata_continues(tmp_path, version):
    from tests.test_checkpoints import _assert_leaves_equal

    # Literal historical metadata deliberately bypasses System.metadata and the writer.
    metadata = {
        "state_shape": [2, 2],
        "dt": 0.01,
        "collider_type": "naive",
        "linear_integrator_type": "euler",
        "rotation_integrator_type": "",
        "domain_type": "free",
        "force_model_type": "spring",
    }
    if version == 2:
        metadata["checkpoint_metadata_version"] = 2
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]),
        vel=jnp.array([[0.1, 0.2], [-0.2, 0.1]]),
    )
    system = jd.System.create(
        **{k: v for k, v in metadata.items() if k != "checkpoint_metadata_version"}
    )
    payload = jax.tree.map(
        lambda x: None if isinstance(x, jax.Array) and x.size == 0 else x, system
    )
    # Historical states predate the additional shared-facet incidence axes.
    state_payload = ocp.tree.serialize_tree(state, keep_empty_nodes=True)
    del state_payload["facet_ids_extra"]
    del state_payload["facet_vertices_extra"]
    with ocp.CheckpointManager(tmp_path) as manager:
        manager.save(
            0,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state_payload),
                system=ocp.args.StandardSave(payload),
                state_metadata=ocp.args.JsonSave({"shape": [2, 2]}),
                system_metadata=ocp.args.JsonSave(metadata),
            ),
        )
        manager.wait_until_finished()
    with jd.CheckpointLoader(tmp_path) as loader:
        restored = loader.load()
    _assert_leaves_equal(state, restored[0], "legacy state")
    expected, actual = (
        jd.System.step(state, system, n=3),
        jd.System.step(*restored, n=3),
    )
    _assert_leaves_equal(
        expected[0], actual[0], "legacy continuation", rtol=2e-6, atol=1e-7
    )
    _assert_leaves_equal(expected[1], actual[1], "legacy system", rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize(
    "payload,missing", [({"first": 1, "second": 2}, "second"), ([1, 2, 3], "1")]
)
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Known silent truncation of damaged containers; utils/h5.py is protected from edits",
)
def test_hdf5_rejects_missing_declared_container_members(tmp_path, payload, missing):
    path = str(tmp_path / "damaged-container.h5")
    h5.save(payload, path)
    with h5py.File(path, "r+") as file:
        del file[f"root/{missing}"]
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        h5.load(path)
