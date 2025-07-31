# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, Generic, Type, TypeVar
from inspect import signature

import jax

T = TypeVar("T", bound="Factory")

@partial(jax.tree_util.register_dataclass, drop_fields=["_registry"])
@dataclass
class Factory(ABC, Generic[T]):
    """
    Base class for pluggable components.
    """
    __slots__ = ()
    _registry: ClassVar[Dict[str, Type["Factory"]]] = {}
    
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._registry = {}  # subclass hook – each concrete root gets its own private registry

    # ---------------- registration decorator ------------------------ #
    @classmethod
    def register(cls, key: str | None = None) -> Callable[[Type[T]], Type[T]]:
        def decorator(sub_cls: Type[T]) -> Type[T]:
            k = (key or sub_cls.__name__).lower()
            if k in cls._registry:
                raise ValueError(
                    f"{cls.__name__}: key '{k}' already registered for {cls._registry[k].__name__}"
                )
            cls._registry[k] = sub_cls
            return sub_cls

        return decorator

    # ---------------- factory method -------------------------------- #
    @classmethod
    def create(cls: Type[T], key: str, /, **kw: Any) -> T:
        try:
            sub_cls = cls._registry[key.lower()]
        except KeyError as err:
            raise KeyError(
                f"Unknown {cls.__name__} '{key}'. "
                f"Available: {list(cls._registry)}"
            ) from err
        try:
            signature(sub_cls).bind_partial(**kw)
        except TypeError as err:
            raise TypeError(
                f"Invalid keyword(s) for {sub_cls.__name__}: {err}"
            ) from None
        return sub_cls(**kw)  # type: ignore[arg-type]