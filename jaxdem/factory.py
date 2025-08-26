# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
The factory defines and instantiates specific simulation components.
"""

from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, Generic, Type, TypeVar
from inspect import signature

import jax

T = TypeVar("T", bound="Factory")


@partial(jax.tree_util.register_dataclass, drop_fields=["_registry"])
@dataclass(frozen=True)
class Factory(ABC, Generic[T]):
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

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._registry = (
            {}
        )  # subclass hook – each concrete root gets its own private registry

    @classmethod
    def register(cls, key: str | None = None) -> Callable[[Type[T]], Type[T]]:
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

        def decorator(sub_cls: Type[T]) -> Type[T]:
            k = (key or sub_cls.__name__).lower()
            if k in cls._registry:
                raise ValueError(
                    f"{cls.__name__}: key '{k}' already registered for {cls._registry[k].__name__}"
                )
            cls._registry[k] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def create(cls: Type[T], key: str, /, **kw: Any) -> T:
        """
        Creates and returns an instance of a registered subclass.

        This method looks up the subclass associated with the given `key`
        in the factory's registry and then calls its constructor with the
        provided arguments. If the subclass defines a `_from_factory` method,
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

        # Prefer _from_factory if defined
        factory_method = getattr(sub_cls, "_create", sub_cls)

        try:
            signature(factory_method).bind_partial(**kw)
        except TypeError as err:
            sig = signature(factory_method)
            raise TypeError(
                f"Invalid keyword(s) for {sub_cls.__name__}: {err}. "
                f"Expected signature: {sub_cls.__name__}.{factory_method.__name__}{sig}"
            ) from None

        return factory_method(**kw)  # type: ignore[arg-type]
