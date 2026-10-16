"""Checkpoint metadata schema policy tests without filesystem I/O."""

import pytest

from jaxdem.writers.checkpoints import (
    BaseCheckpointManager,
    _checkpoint_schema_version,
    _validate_schema3_system_metadata,
)


def test_missing_checkpoint_version_is_supported_legacy_v1() -> None:
    metadata = {"field": "value"}
    assert _checkpoint_schema_version(metadata) == 1
    assert metadata == {"field": "value"}


def test_current_checkpoint_version_is_consumed() -> None:
    from jaxdem.utils.serialization_schema import make_serialization_manifest

    metadata = {
        "checkpoint_metadata_version": 3,
        "serialization_manifest": make_serialization_manifest("orbax", "state_system"),
    }
    assert _checkpoint_schema_version(metadata) == 3
    assert metadata == {}


@pytest.mark.parametrize("version", [0, 4, True, "future", 1.5])
def test_unsupported_or_malformed_checkpoint_version_is_rejected(
    version: object,
) -> None:
    with pytest.raises(RuntimeError, match="schema version"):
        _checkpoint_schema_version({"checkpoint_metadata_version": version})


def test_checkpoint_clean_uses_shared_protected_directory_policy(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "checkpoint"
    marker.write_text("valuable")
    manager = BaseCheckpointManager(tmp_path)
    manager._prepare_directory(clean=False)
    assert marker.read_text() == "valuable"

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        manager._prepare_directory(clean=True)


REQUIRED_CONFIGURATION = (
    "state_shape",
    "bond_id_shape",
    "linear_integrator_type",
    "linear_integrator_kw",
    "rotation_integrator_type",
    "rotation_integrator_kw",
    "collider_type",
    "collider_kw_metadata",
    "domain_type",
    "domain_kw",
    "force_model_type",
    "force_model_metadata",
    "bonded_force_model_type",
    "bonded_force_model_kw",
    "mat_table_metadata",
    "force_function_metadata",
    "minimizer",
    "target_fn",
    "user_pre_step_actions",
    "user_post_step_actions",
)


@pytest.mark.parametrize("missing", REQUIRED_CONFIGURATION)
def test_schema_three_requires_each_saved_configuration_field(missing) -> None:
    # Independent list: deriving it from the validator would share an omitted-field bug.
    metadata = dict.fromkeys(REQUIRED_CONFIGURATION)
    _validate_schema3_system_metadata(metadata)
    del metadata[missing]
    with pytest.raises(RuntimeError, match=missing):
        _validate_schema3_system_metadata(metadata)
