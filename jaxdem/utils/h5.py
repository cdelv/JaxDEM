# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cde
import dataclasses
import importlib
import json
import warnings

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


def _is_array(x) -> bool:
    return isinstance(x, (jax.Array, np.ndarray))


def _to_numpy(x):
    # ensure host numpy for h5py
    return np.asarray(jax.device_get(x))


def _as_int(x) -> int:
    # Handles python scalars, numpy scalars, and jax arrays
    arr = np.asarray(jax.device_get(x))
    if arr.size == 1:
        return int(arr.reshape(()))
    _warn("scalar", f"expected int scalar, got shape={arr.shape}; using first element")
    return int(arr.flat[0])


def _as_float(x) -> float:
    arr = np.asarray(jax.device_get(x))
    if arr.size == 1:
        return float(arr.reshape(()))
    _warn("scalar", f"expected float scalar, got shape={arr.shape}; using first element")
    return float(arr.flat[0])


def _warn(kind: str, msg: str) -> None:
    # In notebooks, warnings display the source line where they are emitted.
    # Use a larger stacklevel so the warning points to the user's call to `load(...)`.
    warnings.warn(f"h5.load: {kind}: {msg}", RuntimeWarning, stacklevel=6)


def _write_any(g: h5py.Group, name: str, obj, *, allow_callables: bool = False, skip_bad_callables: bool = False):
    # None
    if obj is None:
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "none"
        return

    # Quaternion
    if isinstance(obj, Quaternion):
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "quaternion"
        sg.create_dataset("w", data=_to_numpy(obj.w))
        sg.create_dataset("xyz", data=_to_numpy(obj.xyz))
        return

    # Arrays
    if _is_array(obj):
        g.create_dataset(name, data=_to_numpy(obj))
        g[name].attrs["__kind__"] = "array"
        return

    # Scalars / strings (JSON-safe)
    if isinstance(obj, (bool, int, float, np.number, str)):
        ds = g.create_dataset(
            name, data=obj, dtype=_STR if isinstance(obj, str) else None
        )
        ds.attrs["__kind__"] = "scalar"
        return

    # Dict[str, ...]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj.keys()):
            raise TypeError(
                f"Only dict[str, ...] supported. Got keys: {list(obj.keys())[:5]}"
            )
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "dict"
        sg.attrs["__keys__"] = json.dumps(list(obj.keys()))
        for k, v in obj.items():
            _write_any(sg, k, v, allow_callables=allow_callables, skip_bad_callables=skip_bad_callables)
        return

    # list/tuple
    if isinstance(obj, (list, tuple)):
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "list" if isinstance(obj, list) else "tuple"
        sg.attrs["__len__"] = len(obj)
        for i, v in enumerate(obj):
            _write_any(sg, str(i), v, allow_callables=allow_callables, skip_bad_callables=skip_bad_callables)
        return

    # Callable (optional, only if importable)
    if callable(obj):
        mod = getattr(obj, "__module__", None)
        qual = getattr(obj, "__qualname__", None)
        is_bad = not mod or not qual or "<locals>" in qual
        if is_bad:
            if skip_bad_callables:
                return  # silently skip
            raise TypeError(f"Callable not importable by qualname: {obj!r}")
        if not allow_callables:
            raise TypeError(
                f"Refusing to serialize callable {obj!r}. Set allow_callables=True or drop it."
            )
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "callable"
        sg.attrs["__import__"] = f"{mod}:{qual}"
        return

    # Dataclass (covers State/System/MaterialTable/ForceModels/etc.)
    if dataclasses.is_dataclass(obj):
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "dataclass"
        sg.attrs["__class__"] = _qualname(type(obj))
        for f in dataclasses.fields(obj):
            _write_any(
                sg, f.name, getattr(obj, f.name), allow_callables=allow_callables, skip_bad_callables=skip_bad_callables
            )
        return

    raise TypeError(f"Unsupported type at {name}: {type(obj)}")


def _construct_default_state_from_group(g: h5py.Group):
    """
    Bootstrap a State instance with default values based on minimal shape info
    stored in the H5 group.
    """
    if "pos_c" not in g:
        raise KeyError("Cannot bootstrap State: missing dataset 'pos_c'")
    shape = tuple(g["pos_c"].shape)

    from ..state import State  # lazy import to avoid import cycles

    return State.create(pos=jnp.zeros(shape, dtype=float))


def _construct_default_system_from_group(g: h5py.Group):
    """
    Bootstrap a System instance with default values based on minimal shape info
    stored in the H5 group.
    """
    if "force_manager" in g and "external_force" in g["force_manager"]:
        state_shape = tuple(g["force_manager"]["external_force"].shape)
    elif "force_manager" in g and "external_force_com" in g["force_manager"]:
        state_shape = tuple(g["force_manager"]["external_force_com"].shape)
    else:
        raise KeyError(
            "Cannot bootstrap System: missing 'force_manager/external_force' (or '_com') to infer state_shape"
        )

    # dt/time may be scalar or vector (e.g., stacked systems). We avoid coercing here:
    # construct with safe scalar defaults, then overwrite from file during merge.
    dt = 0.005
    time = 0.0

    from ..system import System  # lazy import to avoid import cycles

    # Note: System.create cannot accept key/step_count; we overwrite them after load if present.
    return System.create(state_shape=state_shape, dt=dt, time=time)


def _read_dataclass_merge(g: h5py.Group, *, warn_missing: bool, warn_unknown: bool):
    cls = _import_qualname(g.attrs["__class__"])
    field_names = {f.name for f in dataclasses.fields(cls)}
    saved_names = set(g.keys())

    unknown = sorted(saved_names - field_names)
    missing = sorted(field_names - saved_names)

    # Bootstrap a "default" instance for key dataclasses so newly-added fields are filled with defaults.
    is_state = cls.__name__ == "State" and cls.__module__.endswith(".state")
    is_system = cls.__name__ == "System" and cls.__module__.endswith(".system")

    if is_state:
        obj = _construct_default_state_from_group(g)
    elif is_system:
        obj = _construct_default_system_from_group(g)
    else:
        # Best-effort generic path: only pass known fields. This is safer than the previous
        # behavior (passing every saved field) but still may fail if required args are missing.
        kw = {k: _read_any(g[k], warn_missing=warn_missing, warn_unknown=warn_unknown) for k in (saved_names & field_names)}
        if warn_unknown and unknown:
            _warn(cls.__name__, f"unknown saved fields {unknown} - skipping")
        if warn_missing and missing:
            _warn(
                cls.__name__,
                f"missing saved fields {missing} - falling back to default values",
            )
        return cls(**kw)

    if warn_unknown and unknown:
        _warn(cls.__name__, f"unknown saved fields {unknown} - skipping")
    if warn_missing and missing:
        _warn(
            cls.__name__,
            f"missing saved fields {missing} - falling back to default values",
        )

    # Overwrite fields that exist in both the file + current class definition.
    for name in sorted(saved_names & field_names):
        val = _read_any(g[name], warn_missing=warn_missing, warn_unknown=warn_unknown)
        if is_system and name in ("dt", "time"):
            arr = np.asarray(jax.device_get(val))
            if arr.size == 1:
                val = jnp.asarray(_as_float(val), dtype=float)
        if is_system and name in ("dim", "step_count"):
            arr = np.asarray(jax.device_get(val))
            if arr.size == 1:
                val = jnp.asarray(_as_int(val), dtype=int)
        try:
            setattr(obj, name, val)
        except (AttributeError, TypeError):
            object.__setattr__(obj, name, val)

    return obj


def _read_any(node, *, warn_missing: bool = True, warn_unknown: bool = True):
    # dataset
    if isinstance(node, h5py.Dataset):
        kind = node.attrs.get("__kind__", None)
        if kind in (None, "array", "scalar"):
            x = node[()]
            # bytes -> str
            if isinstance(x, (bytes, np.bytes_)):
                return x.decode("utf-8")
            # numeric -> jax array (keeps System/State happy)
            if np.isscalar(x):
                return jnp.asarray(x)
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
        return {k: _read_any(g[k], warn_missing=warn_missing, warn_unknown=warn_unknown) for k in keys}

    if kind in ("list", "tuple"):
        # Read only keys that exist (some may have been skipped during save)
        indices = sorted(int(k) for k in g.keys())
        items = [
            _read_any(g[str(i)], warn_missing=warn_missing, warn_unknown=warn_unknown)
            for i in indices
        ]
        return items if kind == "list" else tuple(items)

    if kind == "callable":
        fn = _import_qualname(g.attrs["__import__"])
        return fn

    if kind == "dataclass":
        return _read_dataclass_merge(g, warn_missing=warn_missing, warn_unknown=warn_unknown)

    raise ValueError(f"Unknown group kind {kind!r}")


def save(obj, path: str, *, overwrite: bool = True, allow_callables: bool = False, skip_bad_callables: bool = True):
    import os

    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError(path)
    with h5py.File(path, "w") as f:
        _write_any(f, "root", obj, allow_callables=allow_callables, skip_bad_callables=skip_bad_callables)


def load(path: str):
    with h5py.File(path, "r") as f:
        return _read_any(f["root"])


def load_with_warnings(path: str, *, warn_missing: bool = True, warn_unknown: bool = True):
    """
    Load an object from an HDF5 file with optional schema-compatibility warnings.

    This behaves like :func:`load`, but it propagates warning configuration into the loader.
    """
    with h5py.File(path, "r") as f:
        return _read_any(f["root"], warn_missing=warn_missing, warn_unknown=warn_unknown)
