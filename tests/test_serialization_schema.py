"""Shared Orbax/HDF5 serialization schema contracts."""

import json

import h5py
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

import jaxdem as jd
from jaxdem.utils import h5
from jaxdem.utils.serialization_schema import (
    make_serialization_manifest,
    serialization_schema_version,
)
from tests.custom_forces import harmonic_trap, harmonic_trap_energy


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
