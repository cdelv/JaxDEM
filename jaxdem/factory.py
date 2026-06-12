# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""The factory defines and instantiates specific simulation components."""

from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
    cast,
)

import jax

# TypeVars for type-preserving decorators & methods
RootT = TypeVar("RootT", bound="Factory")
SubT = TypeVar("SubT", bound="Factory")


def _normalize_key(key: str) -> str:
    """Normalize a registry key: lowercase with spaces, underscores, and hyphens removed.

    Keys are normalized both at registration and at lookup, so any spelling
    style (``"CellList"``, ``"cell_list"``, ``"celllist"``) resolves to the
    same registered class.
    """
    return key.replace(" ", "").replace("_", "").replace("-", "").lower()


@partial(
    jax.tree_util.register_dataclass, drop_fields=["_registry", "__registry_name__"]
)
@dataclass
class Factory(ABC):
    """Base factory class for pluggable components. This abstract base class provides a mechanism for registering and creating
    subclasses based on a string key.

    Notes:
    ------
    Each concrete subclass gets its own private registry. Keys are strings and
    are normalized before use: lookup is case-insensitive and ignores spaces,
    underscores, and hyphens (``"CellList"``, ``"cell_list"``, and
    ``"celllist"`` are the same key).

    Example:
    --------
    Use Factory as a base class for a specific component type (e.g., `Foo`):

    >>> class Foo(Factory["Foo"], ABC):
    >>>   ...

    Register a concrete subclass of `Foo`:

    >>> @Foo.register("bar")
    >>> class bar:
    >>>     ...

    To instantiate the subclass instance:

    >>> Foo.create("bar", **bar_kw)

    """

    if not TYPE_CHECKING:
        __slots__ = ()

    __registry_name__: ClassVar[str | None]
    _registry: ClassVar[dict[str, type[Factory]]] = {}
    """Dictionary to store the registered subclasses."""

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # subclass hook – each concrete root gets its own private registry
        cls._registry = {}
        cls.__registry_name__ = None

        if "create" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} is not allowed to override the `create` method. "
                "Use `Create` instead for custom instantiation logic."
            )

    @classmethod
    def registry_name(cls) -> str:
        """Returns the key under which this class is registered."""
        name = getattr(cls, "__registry_name__", None)
        if name is None:
            raise KeyError(f"{cls.__name__} is not registered in the Factory.")
        return str(name)

    @property
    def type_name(self) -> str:
        """Returns the key under which this instance's class is registered."""
        return type(self).registry_name()

    @property
    def metadata(self) -> dict[str, Any]:
        """Automatically serialize the component's dataclass fields for checkpointing/restoration."""
        import dataclasses
        from inspect import signature

        meta: dict[str, Any] = {}

        create_or_ctor = getattr(type(self), "Create", None)
        if create_or_ctor is None:
            create_or_ctor = getattr(type(self), "__init__", None) or type(self)
        try:
            sig = signature(create_or_ctor)
            expected_params = set(sig.parameters.keys())
            expected_params.discard("self")
        except Exception:
            expected_params = set()

        if hasattr(self, "__dataclass_fields__") and len(dataclasses.fields(self)) > 0:
            for f in dataclasses.fields(self):
                if f.name in ("_registry", "__registry_name__") or f.name.startswith(
                    "_"
                ):
                    continue
                is_expected = (
                    not expected_params
                    or f.name in expected_params
                    or f"{f.name}_type" in expected_params
                    or f"{f.name}_kw" in expected_params
                )
                if is_expected:
                    val = getattr(self, f.name)
                    meta[f.name] = self._serialize_value(val)
        else:
            # Fallback to public properties/attributes (excluding callables/methods)
            for k in dir(self):
                if k.startswith("_") or k in (
                    "metadata",
                    "type_name",
                    "registry_name",
                    "Create",
                ):
                    continue
                is_expected = (
                    not expected_params
                    or k in expected_params
                    or f"{k}_type" in expected_params
                    or f"{k}_kw" in expected_params
                )
                if is_expected:
                    try:
                        val = getattr(self, k)
                    except AttributeError:
                        continue
                    if callable(val) and not (expected_params and k in expected_params):
                        continue
                    meta[k] = self._serialize_value(val)
        return meta

    def _serialize_value(self, val: Any) -> Any:
        """Helper to recursively serialize nested components, lists, arrays."""
        if isinstance(val, Factory):
            return {"type": val.type_name, "kw": val.metadata}
        elif isinstance(val, (list, tuple)):
            return [self._serialize_value(x) for x in val]
        elif isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        elif hasattr(val, "ndim"):  # jax/numpy arrays
            if val.ndim == 0:
                item = val.item()
                if isinstance(item, (float, int, bool)):
                    return item
                return float(item)
            return val.tolist()
        elif isinstance(val, (int, float, bool, str, type(None))):
            return val
        elif callable(val):
            from .utils import encode_callable

            try:
                return {"_callable": True, "path": encode_callable(val)}
            except Exception:
                return val
        else:
            if hasattr(val, "metadata") and not callable(val.metadata):
                return val.metadata
            return val

    @classmethod
    def _deserialize_kws(cls, val: Any) -> Any:
        if isinstance(val, dict):
            if "type" in val and "kw" in val and len(val) == 2:
                return cls._deserialize_component(val)
            if "_callable" in val and "path" in val and len(val) == 2:
                from .utils import decode_callable

                return decode_callable(val["path"])
            return {k: cls._deserialize_kws(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [cls._deserialize_kws(x) for x in val]
        elif isinstance(val, tuple):
            return tuple(cls._deserialize_kws(x) for x in val)
        return val

    @staticmethod
    def _factory_subclasses() -> list[type[Factory]]:
        """All direct and indirect subclasses of :class:`Factory`, breadth-first."""
        out: list[type[Factory]] = []
        seen: set[type[Factory]] = set()
        queue: list[type[Factory]] = list(Factory.__subclasses__())
        while queue:
            sub = queue.pop(0)
            if sub in seen:
                continue
            seen.add(sub)
            out.append(sub)
            queue.extend(sub.__subclasses__())
        return out

    @classmethod
    def _deserialize_component(cls, data: dict[str, Any]) -> Any:
        type_name = data["type"]
        kw = dict(data.get("kw") or {})
        key = _normalize_key(type_name)
        if cls is not Factory and key in cls._registry:
            kw = cls._deserialize_kws(kw)
            return cls.create(type_name, **kw)
        for root in Factory._factory_subclasses():
            if key in root._registry:
                kw = root._deserialize_kws(kw)
                return root.create(type_name, **kw)
        if key in cls._registry:
            kw = cls._deserialize_kws(kw)
            return cls.create(type_name, **kw)
        return data

    @classmethod
    @partial(jax.named_call, name="Factory.register")
    def register(
        cls: type[RootT], key: str | None = None
    ) -> Callable[[type[SubT]], type[SubT]]:
        """Registers a subclass with the factory's registry.

        This method returns a decorator that can be applied to a class
        to register it under a specific `key`.

        Parameters
        ----------
        key : str or None, optional
            The string key under which to register the subclass. If `None`,
            the lowercase name of the subclass itself will be used as the key.
            Keys are normalized (lowercase; spaces, underscores, and hyphens
            stripped), so ``"CellList"``, ``"cell_list"``, and ``"celllist"``
            all denote the same key.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            A decorator function that takes a class and registers it, returning
            the class unchanged.

        Raises
        ------
        ValueError
            If the provided `key` (or the default class name) is already
            registered in the factory's registry for a *different* class.
            Re-registering the same class under the same key (e.g. when
            re-running a notebook cell) is allowed and idempotent.

        Example
        -------
        Register a class named "MyComponent" under the key "mycomp":

        >>> @MyFactory.register("mycomp")
        >>> class MyComponent:
        >>>     ...

        Register a class named "DefaultComponent" using its own name as the key:

        >>> @MyFactory.register()
        >>> class DefaultComponent:
        >>>     ...

        """

        def decorator(sub_cls: type[SubT]) -> type[SubT]:
            # Preserve an explicitly provided empty-string key instead of
            # defaulting to the subclass name. Only fall back to the class name
            # when the caller passes `None`.
            k = _normalize_key(sub_cls.__name__ if key is None else key)
            existing_cls = cls._registry.get(k)
            if existing_cls is not None and not (
                existing_cls.__qualname__ == sub_cls.__qualname__
                and existing_cls.__module__ == sub_cls.__module__
            ):
                raise ValueError(
                    f"{cls.__name__}: key '{k}' already registered for {existing_cls.__name__}"
                )
            cls._registry[k] = sub_cls

            # Stamp the registered name on the class.
            # Only check for an explicit override on the subclass itself. Base
            # classes may already be registered (e.g., under the empty string
            # key) and should not block subclasses from choosing their own
            # registration name.
            existing = sub_cls.__dict__.get("__registry_name__", None)

            if existing is not None and existing != k:
                raise ValueError(
                    f"{sub_cls.__name__} has __registry_name__={existing!r}, "
                    f"but is being registered as {k!r}."
                )
            sub_cls.__registry_name__ = k

            return sub_cls

        return decorator

    @staticmethod
    def _resolve_factory_class(ann: Any) -> type[Factory] | None:
        import collections.abc
        import typing
        from inspect import Parameter
        from typing import get_args, get_origin

        if ann is None or ann is typing.Any or ann is Parameter.empty:
            return None
        if isinstance(ann, str):
            import re

            # Extract the identifiers named by the annotation string
            # ("LinearIntegrator | None" -> {"LinearIntegrator", "None"},
            # "Sequence[ForceModel]" -> {"Sequence", "ForceModel"}) and match
            # type names exactly, so e.g. "Material" does not match
            # "MaterialMatchmaker" by substring.
            names = {
                token.split(".")[-1]
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", ann)
            }
            for root in Factory._factory_subclasses():
                if root.__name__ in names:
                    return root
            return None
        origin = get_origin(ann)
        if origin is typing.Union or (
            hasattr(typing, "UnionType") and origin is typing.UnionType
        ):
            for arg in get_args(ann):
                res = Factory._resolve_factory_class(arg)
                if res is not None:
                    return res
            return None
        if origin in (
            list,
            tuple,
            set,
            typing.Sequence,
            collections.abc.Sequence,
            collections.abc.Iterable,
        ):
            args = get_args(ann)
            if args:
                return Factory._resolve_factory_class(args[0])
        if isinstance(ann, type) and issubclass(ann, Factory):
            return ann
        return None

    @classmethod
    @partial(jax.named_call, name="Factory.create")
    def create(cls: type[RootT], key: str, /, **kw: Any) -> RootT:
        """Creates and returns an instance of a registered subclass.

        This method looks up the subclass associated with the given `key`
        in the factory's registry and then calls its constructor with the
        provided arguments. If the subclass defines a `Create` method (capitalized),
        that method will be called instead of the constructor. This allows
        subclasses to validate or preprocess arguments before instantiation.

        Parameters
        ----------
        key : str
            The registration key of the subclass to be created.

        **kw : Any
            Arbitrary keyword arguments to be passed directly to the constructor of the registered subclass.

        Returns
        -------
        T
            An instance of the registered subclass.

        Raises
        ------
        KeyError
            If the provided `key` is not found in the factory's registry.
        TypeError
            If the provided `**kw` arguments do not match the signature
            of the registered subclass's constructor.

        Example
        -------
        Given `Foo` factory and `Bar` registered:

        >>> bar_instance = Foo.create("bar", value=42)
        >>> print(bar_instance)
        Bar(value=42)

        """
        try:
            sub_cls = cls._registry[_normalize_key(key)]
        except KeyError as err:
            raise KeyError(
                f"Unknown {cls.__name__} '{key}'. Available: {list(cls._registry)}"
            ) from err

        # Prefer 'Create' if present, else call the class constructor
        create_or_ctor = getattr(sub_cls, "Create", None)
        if create_or_ctor is None:
            create_or_ctor = getattr(sub_cls, "__init__", None) or sub_cls

        # Tell the type checker this callable returns RootT
        factory_callable = cast(
            Callable[..., RootT], getattr(sub_cls, "Create", None) or sub_cls
        )

        sig = signature(create_or_ctor)
        expected_params = list(sig.parameters.keys())
        if "self" in expected_params:
            expected_params.remove("self")
        has_var_keyword = any(
            param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
        )

        # Deserialize keyword arguments contextually based on type annotations
        kw = dict(kw)
        for p, val in list(kw.items()):
            param = sig.parameters.get(p)
            target_cls = None
            if param is not None and param.annotation is not param.empty:
                target_cls = cls._resolve_factory_class(param.annotation)

            if target_cls is not None:
                if (
                    isinstance(val, dict)
                    and "type" in val
                    and "kw" in val
                    and len(val) == 2
                ):
                    kw[p] = target_cls._deserialize_component(val)
                elif isinstance(val, list) and all(
                    isinstance(x, dict) and "type" in x and "kw" in x and len(x) == 2
                    for x in val
                ):
                    kw[p] = [target_cls._deserialize_component(x) for x in val]
                else:
                    kw[p] = cls._deserialize_kws(val)
            else:
                kw[p] = cls._deserialize_kws(val)

        # Unpack nested Factory objects/dicts into _type and _kw arguments if the signature expects them
        for p in list(kw.keys()):
            if p not in expected_params:
                if f"{p}_type" in expected_params or f"{p}_kw" in expected_params:
                    val = kw.pop(p)
                    if isinstance(val, Factory):
                        if f"{p}_type" in expected_params:
                            kw[f"{p}_type"] = val.type_name
                        if f"{p}_kw" in expected_params:
                            kw[f"{p}_kw"] = val.metadata
                    elif isinstance(val, dict) and "type" in val:
                        if f"{p}_type" in expected_params:
                            kw[f"{p}_type"] = val["type"]
                        if f"{p}_kw" in expected_params:
                            kw[f"{p}_kw"] = val.get("kw") or {}

        # If the signature does not accept **kwargs, drop (and warn about) any
        # keyword arguments not expected by the signature. Silently swallowing
        # them would turn a simple typo into a wrong-results-without-error bug.
        if not has_var_keyword:
            dropped = [p for p in kw if p not in expected_params]
            for p in dropped:
                kw.pop(p)
            if dropped:
                warnings.warn(
                    f"{cls.__name__}.create('{key}'): ignoring unknown keyword(s) "
                    f"{dropped}. Expected signature: "
                    f"{sub_cls.__name__}.{create_or_ctor.__name__}{sig}",
                    stacklevel=2,
                )

        # Optional: friendly arg check
        try:
            sig.bind_partial(**kw)
        except TypeError as err:
            raise TypeError(
                f"Invalid keyword(s) for {sub_cls.__name__}: {err}. "
                f"Expected signature: {sub_cls.__name__}.{create_or_ctor.__name__}{sig}"
            ) from None

        return factory_callable(**kw)
