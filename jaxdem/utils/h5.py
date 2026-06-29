# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""HDF5 save/load utilities (v2).

Design goals (no API changes):

- Generic object round-trip for JaxDEM dataclasses and common containers.
- Skip callables with a warning (user handles them explicitly, e.g. DP containers).
- Robust schema evolution: warn on unknown fields, warn + default missing fields.
- Enforce Python types for dataclass fields marked metadata={"static": True} to keep
  JAX static hashing happy (e.g. NeighborList.max_neighbors).

This module intentionally does NOT add any top-level file format metadata.
It does use minimal per-node tags (e.g. "__kind__", "__class__") required to
round-trip Python types through HDF5.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import warnings
from typing import Any, TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as np

from .quaternion import Quaternion
from ..forces.force_manager import ForceManager

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


_STR = h5py.string_dtype(encoding="utf-8")

_CLASS_RENAMES = {
    "jaxdem.materials.elasticMats:Elastic": "jaxdem.materials.elastic_mats:Elastic",
    "jaxdem.materials.elasticMats:ElasticFriction": "jaxdem.materials.elastic_mats:ElasticFriction",
    "jaxdem.materials.ljMats:LJMaterial": "jaxdem.materials.lj_mats:LJMaterial",
    "jaxdem.materials.materialTable:MaterialTable": "jaxdem.materials.material_table:MaterialTable",
    "jaxdem.minimizers.fire:LinearFIRE": "jaxdem.integrators:LinearIntegrator",
    "jaxdem.minimizers.fire:RotationFIRE": "jaxdem.integrators:RotationIntegrator",
    "jaxdem.minimizers.gradient_descent:LinearGradientDescent": "jaxdem.integrators:LinearIntegrator",
    "jaxdem.minimizers.gradient_descent:RotationGradientDescent": "jaxdem.integrators:RotationIntegrator",
    "jaxdem.minimizers.optax_optimizer:OptaxOptimizer": "jaxdem.integrators:LinearIntegrator",
    "jaxdem.minimizers.optax_optimizer:OptaxRotationNoOp": "jaxdem.integrators:RotationIntegrator",
}

_FIELD_RENAMES = {
    ("jaxdem.state", "State"): {
        "angVel": "ang_vel",
        "clump_ID": "clump_id",
        "deformable_ID": "bond_id",
    },
    ("jaxdem.colliders.neighbor_list", "NeighborList"): {
        "cell_list": "secondary_collider",
    },
}


def _qualname(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _import_qualname(s: str) -> Any:
    s = _CLASS_RENAMES.get(s, s)
    mod_name, _, qual = s.partition(":")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _warn(kind: str, msg: str) -> None:
    # Larger stacklevel so warnings point at user's save/load call.
    warnings.warn(f"h5: {kind}: {msg}", RuntimeWarning, stacklevel=6)


def _is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray))


def _to_numpy(x: Any) -> np.ndarray:
    # Ensure host numpy for h5py
    return np.asarray(jax.device_get(x))


def _py_static(x: Any) -> Any:
    """Convert JAX/NumPy scalar-like values into plain Python objects suitable for
    use as JAX "static" fields/args (i.e., hashable cache keys).
    """
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (jax.Array, np.ndarray)):
        arr = np.asarray(jax.device_get(x))
        if arr.size == 1:
            return arr.reshape(()).item()
        # Non-scalar static values must still be hashable; numpy arrays are not.
        # In practice, JaxDEM static fields are scalars/tuples. Keep as-is but warn.
        _warn(
            "static",
            f"non-scalar static value shape={arr.shape} is not hashable; leaving as numpy array",
        )
        return arr
    if isinstance(x, tuple):
        return tuple(_py_static(v) for v in x)
    if isinstance(x, list):
        return [_py_static(v) for v in x]
    if isinstance(x, dict):
        return {k: _py_static(v) for k, v in x.items()}
    return x


def _write_any(g: h5py.Group, name: str, obj: Any) -> bool:
    """Write obj under g[name]. Returns True if something was written; False if skipped."""
    # Callable: skip (no API changes; user handles explicitly)
    if callable(obj):
        _warn(
            "callable", f"skipping callable field '{name}' ({obj!r}); handle explicitly"
        )
        return False

    # None
    if obj is None:
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "none"
        return True

    # Quaternion
    if isinstance(obj, Quaternion):
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "quaternion"
        sg.create_dataset("w", data=_to_numpy(obj.w))
        sg.create_dataset("xyz", data=_to_numpy(obj.xyz))
        return True

    # Arrays
    if _is_array(obj):
        ds = g.create_dataset(name, data=_to_numpy(obj))
        ds.attrs["__kind__"] = "array"
        return True

    # Scalars / strings
    if isinstance(obj, (bool, int, float, np.bool_, np.number, str)):
        ds = g.create_dataset(
            name, data=obj, dtype=_STR if isinstance(obj, str) else None
        )
        ds.attrs["__kind__"] = "scalar"
        # Record the Python-side type so the scalar round-trips as the same
        # type (bool/int/float/str) instead of a 0-d JAX array (which would
        # also downcast float64 -> float32 under default JAX x64 settings).
        ds.attrs["__pytype__"] = type(obj).__name__
        return True

    # Dict[str, ...]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError(
                f"Only dict[str, ...] supported. Got keys: {list(obj.keys())[:5]}"
            )
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "dict"
        sg.attrs["__keys__"] = json.dumps(list(obj.keys()))
        for k, v in obj.items():
            _write_any(sg, k, v)
        return True

    # list/tuple
    if isinstance(obj, (list, tuple)):
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "list" if isinstance(obj, list) else "tuple"
        sg.attrs["__len__"] = len(obj)
        for i, v in enumerate(obj):
            _write_any(sg, str(i), v)
        return True

    # Dataclass
    if dataclasses.is_dataclass(obj):
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "dataclass"
        sg.attrs["__class__"] = _qualname(type(obj))
        for f in dataclasses.fields(obj):
            _write_any(sg, f.name, getattr(obj, f.name))
        return True

    raise TypeError(f"Unsupported type at {name}: {type(obj)}")


def _construct_default_state_from_group(g: h5py.Group) -> State:
    if "pos_c" not in g:
        raise KeyError("Cannot bootstrap State: missing dataset 'pos_c'")
    shape = tuple(g["pos_c"].shape)

    from ..state import State  # lazy import

    return State.create(pos=jnp.zeros(shape, dtype=float))


def _construct_default_system_from_group(
    g: h5py.Group, state_shape: tuple[int, ...] | None = None
) -> System:
    if state_shape is None:
        if "force_manager" in g and "external_force" in g["force_manager"]:
            state_shape = tuple(g["force_manager"]["external_force"].shape)
        elif "force_manager" in g and "external_force_com" in g["force_manager"]:
            state_shape = tuple(g["force_manager"]["external_force_com"].shape)
        else:
            raise KeyError(
                "Cannot bootstrap System: missing 'force_manager/external_force' (or '_com') "
                "to infer state_shape; pass state_shape explicitly"
            )

    from ..system import System  # lazy import

    # Use safe scalar defaults; overwritten during merge.
    return System.create(state_shape=state_shape, dt=0.005, time=0.0)


def _read_any(
    node: h5py.Group | h5py.Dataset,
    *,
    warn_missing: bool = True,
    warn_unknown: bool = True,
    state_shape: tuple[int, ...] | None = None,
) -> Any:
    # dataset
    if isinstance(node, h5py.Dataset):
        kind = node.attrs.get("__kind__", None)
        if kind == "scalar":
            x = node[()]
            if isinstance(x, (bytes, np.bytes_)):
                return x.decode("utf-8")
            pytype = node.attrs.get("__pytype__", None)
            if pytype == "bool":
                return bool(x)
            if pytype == "int":
                return int(x)
            if pytype == "float":
                return float(x)
            if pytype == "str":
                return str(x)
            # Legacy files (no __pytype__) or numpy scalars: return a host
            # scalar with the stored dtype, never a 0-d (float32) JAX array.
            if isinstance(x, np.generic):
                return x.item() if pytype is None else x
            return x
        if kind in (None, "array"):
            x = node[()]
            if isinstance(x, (bytes, np.bytes_)):
                return x.decode("utf-8")
            return jnp.asarray(x)
        raise ValueError(f"Unknown dataset kind {kind!r}")

    # group
    g = node
    kind = g.attrs.get("__kind__", None)

    if kind == "none":
        return None
    if kind == "quaternion":
        w = jnp.asarray(g["w"][...])
        xyz = jnp.asarray(g["xyz"][...])
        return Quaternion.create(w=w, xyz=xyz)
    if kind == "dict":
        keys = json.loads(g.attrs["__keys__"])
        return {
            k: _read_any(
                g[k],
                warn_missing=warn_missing,
                warn_unknown=warn_unknown,
                state_shape=state_shape,
            )
            for k in keys
            if k in g
        }
    if kind in ("list", "tuple"):
        indices = sorted(int(k) for k in g)
        items = [
            _read_any(
                g[str(i)],
                warn_missing=warn_missing,
                warn_unknown=warn_unknown,
                state_shape=state_shape,
            )
            for i in indices
        ]
        return items if kind == "list" else tuple(items)
    if kind == "dataclass":
        return _read_dataclass_merge(
            g,
            warn_missing=warn_missing,
            warn_unknown=warn_unknown,
            state_shape=state_shape,
        )

    raise ValueError(f"Unknown group kind {kind!r}")


def _read_dataclass_merge(
    g: h5py.Group,
    *,
    warn_missing: bool,
    warn_unknown: bool,
    state_shape: tuple[int, ...] | None = None,
) -> Any:
    cls = _import_qualname(g.attrs["__class__"])
    fields = list(dataclasses.fields(cls))
    field_names = {f.name for f in fields}
    fields_by_name = {f.name: f for f in fields}
    field_renames = _FIELD_RENAMES.get((cls.__module__, cls.__name__), {})
    saved_to_field = {name: field_renames.get(name, name) for name in g.keys()}
    saved_names = set(saved_to_field.values())

    unknown = sorted(saved_names - field_names)
    missing = sorted(field_names - saved_names)

    is_state = cls.__name__ == "State" and cls.__module__.endswith(".state")
    is_system = cls.__name__ == "System" and cls.__module__.endswith(".system")

    obj: Any
    if is_state:
        obj = _construct_default_state_from_group(g)
    elif is_system:
        obj = _construct_default_system_from_group(g, state_shape=state_shape)
    else:
        # Best-effort: construct with known saved fields only.
        kw = {}
        for saved_name, field_name in sorted(saved_to_field.items()):
            if field_name not in field_names:
                continue
            val = _read_any(
                g[saved_name],
                warn_missing=warn_missing,
                warn_unknown=warn_unknown,
                state_shape=state_shape,
            )
            f = fields_by_name.get(field_name)
            if f is not None and (
                f.metadata.get("static", False)
                or f.metadata.get("jax.tree.static", False)
            ):
                val = _py_static(val)
            kw[field_name] = val
        if warn_unknown and unknown:
            _warn(cls.__name__, f"unknown saved fields {unknown} - skipping")
        if warn_missing and missing:
            _warn(
                cls.__name__,
                f"missing saved fields {missing} - falling back to default values",
            )
        # `inv_box_size` is a pure derived quantity (Domain.create defines it as
        # 1/box_size). Always recompute it from the loaded box_size so that both
        # missing (old files) and stale (files saved after a box rescale) values
        # stay consistent with box_size.
        if "inv_box_size" in field_names and "box_size" in kw:
            kw["inv_box_size"] = 1.0 / kw["box_size"]
        return cls(**kw)

    if warn_unknown and unknown:
        _warn(cls.__name__, f"unknown saved fields {unknown} - skipping")
    if warn_missing and missing:
        _warn(
            cls.__name__,
            f"missing saved fields {missing} - falling back to default values",
        )

    # Overwrite fields present in file + current class definition.
    for saved_name, name in sorted(saved_to_field.items()):
        if name not in field_names:
            continue
        val = _read_any(
            g[saved_name],
            warn_missing=warn_missing,
            warn_unknown=warn_unknown,
            state_shape=state_shape,
        )

        f = fields_by_name.get(name)
        if f is not None and (
            f.metadata.get("static", False) or f.metadata.get("jax.tree.static", False)
        ):
            val = _py_static(val)

        try:
            setattr(obj, name, val)
        except (AttributeError, TypeError):
            object.__setattr__(obj, name, val)

    if is_state and "_rad" in missing and "rad" in field_names:
        object.__setattr__(obj, "_rad", jnp.copy(obj.rad))

    return obj


def _same_callable(a: Any, b: Any) -> bool:
    """Best-effort callable identity check across reloads."""
    if a is b:
        return True
    return getattr(a, "__module__", None) == getattr(b, "__module__", None) and getattr(
        a, "__qualname__", None
    ) == getattr(b, "__qualname__", None)


def _repair_loaded_system(system: Any) -> Any:
    """Restore runtime-only invariants after generic HDF5 deserialization."""
    bonded_model = getattr(system, "bonded_force_model", None)
    if bonded_model is None:
        return system

    fm = system.force_manager
    bonded_force_fn, bonded_energy_fn, bonded_is_com = bonded_model.force_and_energy_fns
    if any(
        _same_callable(force_fn, bonded_force_fn) for force_fn in fm.force_functions
    ):
        return system

    force_entries = [
        (force_fn, energy_fn, is_com)
        for force_fn, energy_fn, is_com in zip(
            fm.force_functions, fm.energy_functions, fm.is_com_force, strict=False
        )
    ]
    force_entries.append((bonded_force_fn, bonded_energy_fn, bonded_is_com))

    repaired_fm = ForceManager.create(
        fm.external_force.shape,
        gravity=fm.gravity,
        force_functions=force_entries,
    )
    repaired_fm = dataclasses.replace(
        repaired_fm,
        external_force=fm.external_force,
        external_force_com=fm.external_force_com,
        external_torque=fm.external_torque,
    )
    return dataclasses.replace(system, force_manager=repaired_fm)


def save(obj: Any, path: str, *, overwrite: bool = True) -> None:
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError(path)
    with h5py.File(path, "w") as f:
        _write_any(f, "root", obj)


def load(
    path: str,
    *,
    warn_missing: bool = True,
    warn_unknown: bool = True,
    state_shape: tuple[int, ...] | None = None,
) -> Any:
    """Load an object saved with :func:`save`.

    ``state_shape`` is an optional ``(N, dim)`` hint used to bootstrap a
    ``System`` skeleton when the file does not contain the datasets needed
    to infer it.
    """
    with h5py.File(path, "r") as f:
        obj = _read_any(
            f["root"],
            warn_missing=warn_missing,
            warn_unknown=warn_unknown,
            state_shape=state_shape,
        )
    return _repair_loaded_systems_in_tree(obj)


def _repair_loaded_systems_in_tree(obj: Any) -> Any:
    """Apply :func:`_repair_loaded_system` to every ``System`` in a loaded tree.

    Saved objects may nest systems inside tuples/lists/dicts (e.g. a
    ``(state, system)`` pair); each one needs its bonded-force functions
    re-registered after deserialization.
    """
    if type(obj).__name__ == "System":
        return _repair_loaded_system(obj)
    if isinstance(obj, tuple):
        return tuple(_repair_loaded_systems_in_tree(v) for v in obj)
    if isinstance(obj, list):
        return [_repair_loaded_systems_in_tree(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _repair_loaded_systems_in_tree(v) for k, v in obj.items()}
    return obj
