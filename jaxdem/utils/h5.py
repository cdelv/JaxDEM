import dataclasses
import importlib
import json

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


def _write_any(g: h5py.Group, name: str, obj, *, allow_callables: bool = False):
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
            _write_any(sg, k, v, allow_callables=allow_callables)
        return

    # list/tuple
    if isinstance(obj, (list, tuple)):
        sg = g.create_group(name)
        sg.attrs["__kind__"] = "list" if isinstance(obj, list) else "tuple"
        sg.attrs["__len__"] = len(obj)
        for i, v in enumerate(obj):
            _write_any(sg, str(i), v, allow_callables=allow_callables)
        return

    # Callable (optional, only if importable)
    if callable(obj):
        if not allow_callables:
            raise TypeError(
                f"Refusing to serialize callable {obj!r}. Set allow_callables=True or drop it."
            )
        mod = getattr(obj, "__module__", None)
        qual = getattr(obj, "__qualname__", None)
        if not mod or not qual:
            raise TypeError(f"Callable not importable by qualname: {obj!r}")
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
                sg, f.name, getattr(obj, f.name), allow_callables=allow_callables
            )
        return

    raise TypeError(f"Unsupported type at {name}: {type(obj)}")


def _read_any(node):
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
        return {k: _read_any(g[k]) for k in keys}

    if kind in ("list", "tuple"):
        n = int(g.attrs["__len__"])
        items = [_read_any(g[str(i)]) for i in range(n)]
        return items if kind == "list" else tuple(items)

    if kind == "callable":
        fn = _import_qualname(g.attrs["__import__"])
        return fn

    if kind == "dataclass":
        cls = _import_qualname(g.attrs["__class__"])
        kw = {k: _read_any(g[k]) for k in g.keys()}
        return cls(**kw)

    raise ValueError(f"Unknown group kind {kind!r}")


def save(obj, path: str, *, overwrite: bool = True, allow_callables: bool = False):
    import os

    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError(path)
    with h5py.File(path, "w") as f:
        _write_any(f, "root", obj, allow_callables=allow_callables)


def load(path: str):
    with h5py.File(path, "r") as f:
        return _read_any(f["root"])
