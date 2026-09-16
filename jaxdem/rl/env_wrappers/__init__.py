# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Wrappers that modify RL environments."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp

from ..environments import Environment
from ...utils.environment import _mask_agent_actions

# Cache of generated wrapper classes keyed by (base class, prefix, extra key).
# Reusing the class keeps pytree treedefs identical across repeated wrapping,
# which preserves jit caches and avoids leaking one class per call.
_WRAP_CACHE: dict[tuple[Any, ...], type] = {}


@partial(jax.named_call, name="env_wrappers._wrap_env")
def _wrap_env(
    env: Environment,
    method_transform: Callable[[str, Callable[..., Any]], Callable[..., Any]],
    prefix: str = "Wrapped",
    cache_key: Any = None,
) -> Environment:
    """Create a new environment subclass with transformed static methods.

    Parameters
    ----------
    env : Environment
        The environment instance to wrap.
    method_transform : Callable
        A function (name: str, func: callable) -> callable
        that returns the transformed function for each staticmethod.
    prefix : str
        Prefix for the name of the generated subclass.
    cache_key : Any
        Extra hashable key identifying the transform's parameters (e.g. clip
        bounds), so distinct transforms get distinct cached classes.

    Returns
    -------
    Environment
        A new environment instance with transformed static methods.

    """
    cls = env.__class__
    key = (cls, prefix, cache_key)
    NewCls = _WRAP_CACHE.get(key)
    if NewCls is None:
        name_space: dict[str, object] = {}

        # Walk the MRO so inherited staticmethods (e.g. ``reset_if_done``,
        # ``info``) are transformed too; the most-derived definition wins.
        for klass in cls.__mro__:
            if not (isinstance(klass, type) and issubclass(klass, Environment)):
                continue
            for name, attr in vars(klass).items():
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
        cast(Any, NewCls)._base_env_cls = base_cls
        _WRAP_CACHE[key] = NewCls

    field_vals = {f.name: getattr(env, f.name) for f in fields(env)}
    return cast(Any, NewCls)(**field_vals)


@partial(jax.named_call, name="env_wrappers.vectorise_env")
def vectorise_env(env: Environment, n: int | None = None) -> Environment:
    """Promote an environment instance to a parallel version by applying
    `jax.vmap(...)` to its static methods.

    Parameters
    ----------
    env : Environment
        The environment to vectorize. May already carry a leading batch
        dimension (e.g. produced with ``jax.vmap``).
    n : int, optional
        When given, the wrapper first broadcasts the (scalar) environment to
        a batch of ``n`` identical copies, so callers do not need to write
        ``jax.vmap(lambda _: env)(jnp.arange(n))`` themselves.
        Then call ``env.reset(env, keys)`` to randomize each copy.

    Example
    -------
    >>> env = vectorise_env(env, n=32)
    >>> env = env.reset(env, jax.random.split(key, 32))
    """
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive Python integer.")
        env = jax.vmap(lambda _: env)(jnp.arange(n))
    return _wrap_env(env, lambda name, fn: jax.vmap(fn), prefix="Vec")


@partial(jax.named_call, name="env_wrappers.clip_action_env")
def clip_action_env(
    env: Environment, min_val: float = -1.0, max_val: float = 1.0
) -> Environment:
    """Clip actions consistently for checkpointing and physics steps.

    Both ``checkpoint(env, action)`` and ``step(env, action)`` receive the
    action clipped to ``[min_val, max_val]``. This keeps action-dependent
    historical baselines consistent with the applied dynamics. Inactive slots
    stay zero even when zero lies outside the clipping interval. This wrapper
    requires floating-point continuous actions; integer categories are rejected.
    """

    if not math.isfinite(min_val) or not math.isfinite(max_val) or min_val > max_val:
        raise ValueError(
            "Action clipping requires finite bounds with min_val <= max_val."
        )

    # Capture this wrapper layer's method, rather than looking it up on the
    # final class: a later vmap wrapper passes scalar slices with that class.
    agent_mask_fn = env.agent_mask

    def transform(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in {"step", "checkpoint"}:

            @jax.jit
            def clipped_action_call(env_obj: Environment, action: jax.Array) -> Environment:
                if not jnp.issubdtype(action.dtype, jnp.floating):
                    raise TypeError("clip_action_env requires continuous floating-point actions")
                clipped_action = _mask_agent_actions(
                    jnp.clip(action, min_val, max_val), agent_mask_fn(env_obj)
                )
                return fn(env_obj, clipped_action)

            return clipped_action_call
        return fn

    return _wrap_env(
        env, transform, prefix="Clipped", cache_key=(float(min_val), float(max_val))
    )


@partial(jax.named_call, name="env_wrappers.is_wrapped")
def is_wrapped(env: Environment) -> bool:
    """Check whether an environment instance is a wrapped environment.

    Parameters
    ----------
    env : Environment
        The environment instance to check.

    Returns
    -------
    bool
        True if the environment is wrapped, False otherwise.

    """
    cls = env.__class__
    # Note: _base_env_cls is a ClassVar on the base class, so it may not
    # exist on unwrapped classes (annotation alone does not create the attr).
    base_cls: type[Environment] = getattr(cls, "_base_env_cls", cls)
    return base_cls is not cls


def unwrap(env: Environment) -> Environment:
    """Unwrap an environment to its original base class. Keeps all current
    field values.

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
    base_cls: type[Environment] = getattr(cls, "_base_env_cls", cls)

    # dataclasses.fields() ignores ClassVar entries, so this will not include
    # _base_env_cls and similar class-level attributes.
    field_vals = {f.name: getattr(env, f.name) for f in fields(env)}
    return base_cls(**field_vals)


__all__ = ["clip_action_env", "is_wrapped", "unwrap", "vectorise_env"]
