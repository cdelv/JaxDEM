# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
The factory defines and instantiates specific simulation components.
"""
from __future__ import annotations

import jax

from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, Type, TypeVar, cast
from inspect import signature

# TypeVars for type-preserving decorators & methods
RootT = TypeVar("RootT", bound="Factory")
SubT = TypeVar("SubT", bound="Factory")


@partial(jax.tree_util.register_dataclass, drop_fields=["_registry"])
@dataclass(frozen=True)
class Factory(ABC):
    """
    Base factory class for pluggable components. This abstract base class provides a mechanism for registering and creating
    subclasses based on a string key.

    Notes
    -----
    Each concrete subclass gets its own private registry. Keys are strings and not case sensitive.

    Example
    -------
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

    __slots__ = ()
    _registry: ClassVar[Dict[str, Type["Factory"]]] = {}
    """ Dictionary to store the registered subclases."""

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # subclass hook – each concrete root gets its own private registry
        cls._registry = {}

        if "create" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} is not allowed to override the `create` method. "
                "Use `Create` instead for custom instantiation logic."
            )

    @classmethod
    def register(
        cls: Type[RootT], key: str | None = None
    ) -> Callable[[Type[SubT]], Type[SubT]]:
        """
        Registers a subclass with the factory's registry.

        This method returns a decorator that can be applied to a class
        to register it under a specific `key`.

        Parameters
        ----------
        key : str or None, optional
            The string key under which to register the subclass. If `None`,
            the lowercase name of the subclass itself will be used as the key.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            A decorator function that takes a class and registers it, returning
            the class unchanged.

        Raises
        ------
        ValueError
            If the provided `key` (or the default class name) is already
            registered in the factory's registry.

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

        def decorator(sub_cls: Type[SubT]) -> Type[SubT]:
            k = (key or sub_cls.__name__).lower()
            if k in cls._registry:
                raise ValueError(
                    f"{cls.__name__}: key '{k}' already registered for {cls._registry[k].__name__}"
                )
            cls._registry[k] = sub_cls

            # Stamp the registered name on the class.
            existing = getattr(sub_cls, "__registry_name__", None)
            if existing is not None and existing != k:
                raise ValueError(
                    f"{sub_cls.__name__} has __registry_name__={existing!r}, "
                    f"but is being registered as {k!r}."
                )
            setattr(sub_cls, "__registry_name__", k)

            # Class-level accessor: SubClass.registry_name() -> "key"
            if not hasattr(sub_cls, "registry_name"):

                @classmethod
                def registry_name(c) -> str:
                    name = getattr(c, "__registry_name__", None)
                    if name is None:
                        raise KeyError(
                            f"{c.__name__} is not registered in {cls.__name__}."
                        )
                    return name

                sub_cls.registry_name = registry_name

            # Instance-level accessor: instance.type_name -> "key"
            if not hasattr(sub_cls, "type_name"):

                @property
                def type_name(self) -> str:
                    return type(self).registry_name()

                sub_cls.type_name = type_name  # type: ignore[attr-defined]

            return sub_cls  # <-- type-preserving

        return decorator

    @classmethod
    def create(cls: Type[RootT], key: str, /, **kw: Any) -> RootT:
        """
        Creates and returns an instance of a registered subclass.

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
            sub_cls = cls._registry[key.lower()]
        except KeyError as err:
            raise KeyError(
                f"Unknown {cls.__name__} '{key}'. " f"Available: {list(cls._registry)}"
            ) from err

        # Prefer 'Create' if present, else call the class constructor
        create_or_ctor = getattr(sub_cls, "Create", None) or sub_cls

        # Tell the type checker this callable returns RootT
        factory_callable = cast(Callable[..., RootT], create_or_ctor)

        # Optional: friendly arg check
        sig = signature(create_or_ctor)
        try:
            sig.bind_partial(**kw)
        except TypeError as err:
            raise TypeError(
                f"Invalid keyword(s) for {sub_cls.__name__}: {err}. "
                f"Expected signature: {sub_cls.__name__}.{create_or_ctor.__name__}{sig}"
            ) from None

        return factory_callable(**kw)
