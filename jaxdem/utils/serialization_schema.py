"""Shared file-level serialization schema for Orbax and HDF5 payloads."""

from __future__ import annotations

from typing import Any

import jax

SCHEMA_NAME = "jaxdem.serialization"
CURRENT_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, CURRENT_SCHEMA_VERSION})
_PAYLOADS = {
    "orbax": frozenset({"state_system"}),
    "hdf5": frozenset({"state", "system", "object_tree"}),
}
_CONVENTIONS = {
    "bond_id": "adjacency_indices_padded_minus_one",
    "clumps": "replicated_body_state_by_clump_id",
    "callables": "validated_import_path_identity",
    "coordinates": "state_pos_equals_pos_c_plus_rotated_pos_p",
    "units": "caller_defined_no_implicit_conversion",
    "dtype": "stored_array_dtype_with_matching_jax_precision_mode",
}


def make_serialization_manifest(storage: str, payload: str) -> dict[str, Any]:
    """Describe a trusted JaxDEM payload independently of its storage format."""
    if storage not in {"orbax", "hdf5"}:
        raise ValueError(f"Unsupported serialization storage {storage!r}")
    if payload not in _PAYLOADS[storage]:
        raise ValueError(
            f"Unsupported {storage} serialization payload {payload!r}; "
            f"expected one of {sorted(_PAYLOADS[storage])}"
        )
    return {
        "name": SCHEMA_NAME,
        "version": CURRENT_SCHEMA_VERSION,
        "storage": storage,
        "payload": payload,
        "trusted_input": True,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "conventions": dict(_CONVENTIONS),
    }


def serialization_schema_version(
    manifest: Any, *, legacy_version: Any = 1, expected_storage: str
) -> int:
    """Validate a manifest, or accept an explicitly supported legacy version."""
    if expected_storage not in _PAYLOADS:
        raise ValueError(f"Unsupported serialization storage {expected_storage!r}")
    if manifest is None:
        raw_version = legacy_version
    else:
        if not isinstance(manifest, dict):
            raise RuntimeError("Serialization manifest must be a JSON object")
        if manifest.get("name") != SCHEMA_NAME:
            raise RuntimeError(
                f"Unsupported serialization schema {manifest.get('name')!r}"
            )
        if manifest.get("storage") != expected_storage:
            raise RuntimeError(
                f"Serialization storage is {manifest.get('storage')!r}, expected {expected_storage!r}"
            )
        if manifest.get("payload") not in _PAYLOADS[expected_storage]:
            raise RuntimeError(
                f"Unsupported {expected_storage} serialization payload "
                f"{manifest.get('payload')!r}"
            )
        if manifest.get("trusted_input") is not True:
            raise RuntimeError(
                "Serialization manifest lacks the trusted-input contract"
            )
        saved_x64 = manifest.get("jax_enable_x64")
        if not isinstance(saved_x64, bool):
            raise RuntimeError(
                "Serialization manifest lacks a valid jax_enable_x64 setting"
            )
        if saved_x64 != bool(jax.config.jax_enable_x64):
            raise RuntimeError(
                "Serialization precision mode does not match this process: "
                f"checkpoint jax_enable_x64={saved_x64}, current "
                f"jax_enable_x64={bool(jax.config.jax_enable_x64)}"
            )
        if manifest.get("conventions") != _CONVENTIONS:
            raise RuntimeError(
                "Serialization manifest conventions do not match this release"
            )
        raw_version = manifest.get("version")

    if isinstance(raw_version, bool):
        raise RuntimeError(f"Invalid serialization schema version {raw_version!r}")
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid serialization schema version {raw_version!r}"
        ) from exc
    if version != raw_version or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"Unsupported serialization schema version {raw_version!r}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if manifest is None and version == CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Serialization schema {CURRENT_SCHEMA_VERSION} requires an explicit manifest"
        )
    if manifest is not None and version != CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Versioned manifests must use schema {CURRENT_SCHEMA_VERSION}; "
            "schemas 1 and 2 are supported only through their legacy layouts"
        )
    return version


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SCHEMA_NAME",
    "SUPPORTED_SCHEMA_VERSIONS",
    "make_serialization_manifest",
    "serialization_schema_version",
]
