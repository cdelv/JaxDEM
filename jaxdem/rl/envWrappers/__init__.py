# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Contains wrappers for modifying rl environments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, fields
from typing import Callable, Type
from functools import partial

from ..environments import Environment


@partial(jax.named_call, name="envWrappers._wrap_env")
def _wrap_env(
    env: "Environment", method_transform: Callable, prefix: str = "Wrapped"
) -> "Environment":
    """
    Internal helper to create a new environment subclass with transformed
    static methods.

    Parameters
    ----------
    env : Environment
        The environment instance to wrap.
    method_transform : Callable
        A function (name: str, func: callable) -> callable
        that returns the transformed function for each staticmethod.

    Returns
    -------
    Environment
        A new environment instance with transformed static methods.
    """
    cls = env.__class__
    name_space: dict[str, object] = {}

    for name, attr in cls.__dict__.items():
        if isinstance(attr, staticmethod) and name not in name_space:
            new_func = method_transform(name, attr.__func__)
            name_space[name] = staticmethod(new_func)

    # Customizable name
    NewCls = type(f"{prefix}{cls.__name__}", (cls,), name_space)
    NewCls = dataclass(slots=True)(NewCls)
    NewCls = jax.tree_util.register_dataclass(NewCls)

    # Remember the scalar base class
    # Preserve the original base class if already wrapped
    base_cls = getattr(cls, "_base_env_cls", cls)
    NewCls._base_env_cls = base_cls

    field_vals = {f.name: getattr(env, f.name) for f in fields(env)}
    return NewCls(**field_vals)


@partial(jax.named_call, name="envWrappers.vectorise_env")
def vectorise_env(env: "Environment") -> "Environment":
    """
    Promote an environment instance to a parallel version by applying
    `jax.vmap(...)` to its static methods.
    """
    return _wrap_env(env, lambda name, fn: jax.vmap(fn), prefix="Vec")


@partial(jax.named_call, name="envWrappers.clip_action_env")
def clip_action_env(
    env: "Environment", min_val: float = -1.0, max_val: float = 1.0
) -> "Environment":
    """
    Wrap an environment so that its `step` method clips the action
    before calling the original step.
    """

    def transform(name, fn):
        if name == "step":

            @jax.jit
            def clipped_step(env_obj, action):
                clipped_action = jnp.clip(action, min_val, max_val)
                return fn(env_obj, clipped_action)

            return clipped_step
        return fn

    return _wrap_env(env, transform, prefix="Clipped")


@partial(jax.named_call, name="envWrappers.is_wrapped")
def is_wrapped(env: "Environment") -> bool:
    """
    Check whether an environment instance is a wrapped environment.

    Parameters
    ----------
    env : Environment
        The environment instance to check.

    Returns
    -------
    bool
        True if the environment is wrapped (i.e., has a `_base_env_cls` attribute),
        False otherwise.
    """
    cls = env.__class__
    # Note: _base_env_cls is a ClassVar on the base class, so it may not
    # exist on unwrapped classes (annotation alone doesn’t create the attr).
    base_cls: Type["Environment"] = getattr(cls, "_base_env_cls", cls)
    return base_cls is not cls


def unwrap(env: "Environment") -> "Environment":
    """
    Unwrap an environment to its original base class while preserving all
    current field values.

    Parameters
    ----------
    env : Environment
        The wrapped environment instance.

    Returns
    -------
    Environment
        A new instance of the original base environment class with the same
        field values as the wrapped instance.
    """
    if not is_wrapped(env):
        return env  # already the base class

    cls = env.__class__
    base_cls: Type["Environment"] = getattr(cls, "_base_env_cls", cls)

    # dataclasses.fields() ignores ClassVar entries, so this won’t include
    # _base_env_cls and friends. :contentReference[oaicite:1]{index=1}
    field_vals = {f.name: getattr(env, f.name) for f in fields(env)}
    return base_cls(**field_vals)


__all__ = ["vectorise_env", "clip_action_env", "is_wrapped", "unwrap"]
