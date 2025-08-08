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


def _wrap_env(env: "Environment", method_transform: Callable) -> "Environment":
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
        if isinstance(attr, staticmethod):
            new_func = method_transform(name, attr.__func__)
            name_space[name] = staticmethod(new_func)

    NewCls = type(f"Wrapped{cls.__name__}", (cls,), name_space)
    NewCls = dataclass(slots=True, frozen=True)(NewCls)
    NewCls = jax.tree_util.register_dataclass(NewCls)

    field_vals = {f.name: getattr(env, f.name) for f in fields(env)}
    return NewCls(**field_vals)


def vectorise_env(env: "Environment") -> "Environment":
    """
    Promote an environment instance to a parallel version by applying
    `jax.vmap(...)` to its static methods.
    """
    return _wrap_env(env, lambda name, fn: jax.vmap(fn))


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

    return _wrap_env(env, transform)
