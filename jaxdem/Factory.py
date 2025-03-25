from abc import ABC
from typing import Dict, Type, TypeVar, Generic, Callable

T = TypeVar("T", bound="Factory")

class Factory(ABC, Generic[T]):
    """
    A generic abstract base class that provides registration and instantiation
    capabilities for simulation-related configuration or strategy classes.

    Subclasses of Factory can:
      - Register specific implementations under user-friendly names.
      - Create instances of those implementations by name.

    This design pattern allows the library to dynamically discover and
    instantiate specialized classes at runtime based on a string identifier.
    """
    _registry: Dict[str, Type[T]] = {}
    """
    A mapping of string identifiers to concrete subclasses. Each entry
    associates a user-friendly name (the key) with the class object (the value).
    """

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

        Example
        -------
        >>> instance = Factory.create("my_impl", param=123)
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"No class registered under '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)
