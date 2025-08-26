# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Contains wrappers for modifying rl environments.
"""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, fields
from typing import Callable

from .environment import Environment


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
    NewCls = dataclass(slots=True, frozen=True)(NewCls)
    NewCls = jax.tree_util.register_dataclass(NewCls)

    # Remember the scalar base class
    # Preserve the original base class if already wrapped
    base_cls = getattr(cls, "_base_env_cls", cls)
    NewCls._base_env_cls = base_cls

    field_vals = {f.name: getattr(env, f.name) for f in fields(env)}
    return NewCls(**field_vals)


def vectorise_env(env: "Environment") -> "Environment":
    """
    Promote an environment instance to a parallel version by applying
    `jax.vmap(...)` to its static methods.
    """
    return _wrap_env(env, lambda name, fn: jax.vmap(fn), prefix="Vec")


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
    return hasattr(env.__class__, "_base_env_cls")


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
    cls = env.__class__
    base_cls = getattr(cls, "_base_env_cls", cls)

    # Collect all dataclass field values from the current env
    field_vals = {f.name: getattr(env, f.name) for f in fields(env)}

    # Reconstruct the base class with the same values
    return base_cls(**field_vals)
