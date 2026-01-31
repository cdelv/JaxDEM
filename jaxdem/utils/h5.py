# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
HDF5 save/load utilities (v2).

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
from typing import Any, Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from .quaternion import Quaternion

_STR = h5py.string_dtype(encoding="utf-8")


def _qualname(cls) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _import_qualname(s: str):
    mod_name, _, qual = s.partition(":")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _warn(kind: str, msg: str) -> None:
    # Larger stacklevel so warnings point at user's save/load call.
    warnings.warn(f"h5: {kind}: {msg}", RuntimeWarning, stacklevel=6)


def _is_array(x) -> bool:
    return isinstance(x, (jax.Array, np.ndarray))


def _to_numpy(x):
    # Ensure host numpy for h5py
    return np.asarray(jax.device_get(x))


def _py_static(x):
    """
    Convert JAX/NumPy scalar-like values into plain Python objects suitable for
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
        _warn("static", f"non-scalar static value shape={arr.shape} is not hashable; leaving as numpy array")
        return arr
    if isinstance(x, tuple):
        return tuple(_py_static(v) for v in x)
    if isinstance(x, list):
        return [_py_static(v) for v in x]
    if isinstance(x, dict):
        return {k: _py_static(v) for k, v in x.items()}
    return x


def _write_any(g: h5py.Group, name: str, obj) -> bool:
    """
    Write obj under g[name]. Returns True if something was written; False if skipped.
    """
    # Callable: skip (no API changes; user handles explicitly)
    if callable(obj):
        _warn("callable", f"skipping callable field '{name}' ({obj!r}); handle explicitly")
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
    if isinstance(obj, (bool, int, float, np.number, str)):
        ds = g.create_dataset(name, data=obj, dtype=_STR if isinstance(obj, str) else None)
        ds.attrs["__kind__"] = "scalar"
        return True

    # Dict[str, ...]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj.keys()):
            raise TypeError(f"Only dict[str, ...] supported. Got keys: {list(obj.keys())[:5]}")
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


def _construct_default_state_from_group(g: h5py.Group):
    if "pos_c" not in g:
        raise KeyError("Cannot bootstrap State: missing dataset 'pos_c'")
    shape = tuple(g["pos_c"].shape)

    from ..state import State  # lazy import

    return State.create(pos=jnp.zeros(shape, dtype=float))


def _construct_default_system_from_group(g: h5py.Group):
    if "force_manager" in g and "external_force" in g["force_manager"]:
        state_shape = tuple(g["force_manager"]["external_force"].shape)
    elif "force_manager" in g and "external_force_com" in g["force_manager"]:
        state_shape = tuple(g["force_manager"]["external_force_com"].shape)
    else:
        raise KeyError(
            "Cannot bootstrap System: missing 'force_manager/external_force' (or '_com') to infer state_shape"
        )

    from ..system import System  # lazy import

    # Use safe scalar defaults; overwritten during merge.
    return System.create(state_shape=state_shape, dt=0.005, time=0.0)


def _read_any(node, *, warn_missing: bool = True, warn_unknown: bool = True):
    # dataset
    if isinstance(node, h5py.Dataset):
        kind = node.attrs.get("__kind__", None)
        if kind in (None, "array", "scalar"):
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
        return {k: _read_any(g[k], warn_missing=warn_missing, warn_unknown=warn_unknown) for k in keys if k in g}
    if kind in ("list", "tuple"):
        indices = sorted(int(k) for k in g.keys())
        items = [
            _read_any(g[str(i)], warn_missing=warn_missing, warn_unknown=warn_unknown)
            for i in indices
        ]
        return items if kind == "list" else tuple(items)
    if kind == "dataclass":
        return _read_dataclass_merge(g, warn_missing=warn_missing, warn_unknown=warn_unknown)

    raise ValueError(f"Unknown group kind {kind!r}")


def _read_dataclass_merge(g: h5py.Group, *, warn_missing: bool, warn_unknown: bool):
    cls = _import_qualname(g.attrs["__class__"])
    fields = list(dataclasses.fields(cls))
    field_names = {f.name for f in fields}
    fields_by_name = {f.name: f for f in fields}
    saved_names = set(g.keys())

    unknown = sorted(saved_names - field_names)
    missing = sorted(field_names - saved_names)

    is_state = cls.__name__ == "State" and cls.__module__.endswith(".state")
    is_system = cls.__name__ == "System" and cls.__module__.endswith(".system")

    if is_state:
        obj = _construct_default_state_from_group(g)
    elif is_system:
        obj = _construct_default_system_from_group(g)
    else:
        # Best-effort: construct with known saved fields only.
        kw = {}
        for k in (saved_names & field_names):
            val = _read_any(g[k], warn_missing=warn_missing, warn_unknown=warn_unknown)
            f = fields_by_name.get(k)
            if f is not None and f.metadata.get("static", False):
                val = _py_static(val)
            kw[k] = val
        if warn_unknown and unknown:
            _warn(cls.__name__, f"unknown saved fields {unknown} - skipping")
        if warn_missing and missing:
            _warn(cls.__name__, f"missing saved fields {missing} - falling back to default values")
        return cls(**kw)

    if warn_unknown and unknown:
        _warn(cls.__name__, f"unknown saved fields {unknown} - skipping")
    if warn_missing and missing:
        _warn(cls.__name__, f"missing saved fields {missing} - falling back to default values")

    # Overwrite fields present in file + current class definition.
    for name in sorted(saved_names & field_names):
        val = _read_any(g[name], warn_missing=warn_missing, warn_unknown=warn_unknown)

        f = fields_by_name.get(name)
        if f is not None and f.metadata.get("static", False):
            val = _py_static(val)

        try:
            setattr(obj, name, val)
        except (AttributeError, TypeError):
            object.__setattr__(obj, name, val)

    return obj


def save(obj, path: str, *, overwrite: bool = True) -> None:
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError(path)
    with h5py.File(path, "w") as f:
        _write_any(f, "root", obj)


def load(path: str, *, warn_missing: bool = True, warn_unknown: bool = True):
    with h5py.File(path, "r") as f:
        return _read_any(f["root"], warn_missing=warn_missing, warn_unknown=warn_unknown)

