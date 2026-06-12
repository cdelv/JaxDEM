# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
import importlib
from typing import Any
from collections.abc import Callable


def encode_callable(fn: Callable[..., Any]) -> str:
    """Return a dotted path like 'jax._src.nn.functions.gelu'."""
    mod = getattr(fn, "__module__", None)
    name = getattr(fn, "__name__", None)
    if not (mod and name):
        raise TypeError(f"Callable must be a plain module-level function, got: {fn!r}")
    return f"{mod}.{name}"


def decode_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a dotted path string."""
    module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"Invalid callable path: {path!r}")
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"Imported object is not callable: {path!r}")
    return fn
