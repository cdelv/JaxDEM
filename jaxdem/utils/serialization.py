# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Import-path encoding and restoration of checkpoint callables."""

import importlib
import inspect
from typing import Any
from collections.abc import Callable


def supported_kwargs(
    fn: Callable[..., Any], kw: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Split saved keyword arguments into those ``fn`` accepts and the rest.

    Saved constructor arguments can outlive a parameter that a later version
    removed. Returns the accepted arguments and the sorted names of the
    dropped ones. Functions with ``**kwargs`` or no inspectable signature
    accept everything.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kw), []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kw), []
    kept = {k: v for k, v in kw.items() if k in parameters}
    return kept, sorted(set(kw).difference(kept))


def encode_callable(fn: Callable[..., Any]) -> str:
    """Return an import path after proving it resolves to this callable.

    Checkpoints store callables by reference, so local functions, closures and
    aliases that do not round-trip by identity are not portable and are
    rejected at save time.
    """
    mod = getattr(fn, "__module__", None)
    name = getattr(fn, "__qualname__", None)
    if not (mod and name) or mod == "__main__" or "<locals>" in name:
        raise TypeError(f"Callable must be a plain module-level function, got: {fn!r}")
    path = f"{mod}.{name}"
    try:
        restored = decode_callable(path)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            f"Callable {fn!r} is not importable by its module path {path!r}. "
            "Define it at module scope in an importable module."
        ) from exc
    if restored is not fn:
        raise TypeError(
            f"Callable {fn!r} does not round-trip by identity through {path!r}. "
            "Checkpoint callbacks and force functions must be importable module-level objects."
        )
    return path


def decode_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a dotted path string.

    Importing executes module code. Callers must only decode metadata from a
    trusted source.
    """
    parts = path.split(".")
    fn: Any = None
    for split_at in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split_at])
        try:
            fn = importlib.import_module(module_path)
        except ImportError:
            continue
        for attr in parts[split_at:]:
            fn = getattr(fn, attr)
        break
    if fn is None:
        raise ImportError(f"Could not import callable path: {path!r}")
    if not callable(fn):
        raise TypeError(f"Imported object is not callable: {path!r}")
    return fn
