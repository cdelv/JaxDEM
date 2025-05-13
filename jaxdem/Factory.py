# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax

from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Type, TypeVar, Generic, Callable

T = TypeVar("T", bound="Factory")
@partial(jax.tree_util.register_dataclass, drop_fields=["_registry"])
@dataclass
class Factory(ABC, Generic[T]):
    """
    A generic abstract base class that provides registration and instantiation
    capabilities for simulation-related configuration or strategy classes.
    """
    _registry: Dict[str, Type[T]] = field(default_factory=dict)

    def __init_subclass__(cls, **kwargs):
        """
        Give each subclass its *own* `_registry` dict.
        """
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls: Type[T], name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator for registering a subclass under a given name.

        Parameters
        ----------
        name : str
            The string identifier under which to register the subclass.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            A decorator function that takes the subclass to be registered and
            returns it unchanged. The subclass is automatically added to the
            `_registry` dictionary.

        Example
        -------
        >>> @Factory.register("my_impl")
        ... class MyImplementation(Factory):
        ...     pass
        """
        def decorator(impl: Type[T]) -> Type[T]:
            cls._registry[name] = impl
            return impl
        return decorator

    @classmethod
    def create(cls: Type[T], name: str, **kwargs) -> T:
        """
        Instantiate a registered subclass by name, passing any provided keyword
        arguments to its constructor.

        Parameters
        ----------
        name : str
            The string identifier of the registered subclass to instantiate.
        **kwargs
            Additional keyword arguments forwarded to the subclass constructor.

        Returns
        -------
        T
            An instance of the registered subclass corresponding to the given name.

        Raises
        ------
        KeyError
            If no subclass is registered under the specified `name`.
        """
        if name not in cls._registry:
            raise KeyError(f"No class registered under {name}. Available: {list(cls._registry.keys())}.")

        return cls._registry[name](**kwargs)


