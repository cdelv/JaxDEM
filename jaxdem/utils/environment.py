# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to handle environments.
"""

from __future__ import annotations

import jax

from typing import TYPE_CHECKING, Callable
from functools import partial

if TYPE_CHECKING:
    from ..rl.environments import Environment


@staticmethod
@partial(jax.jit, static_argnames=("model", "n", "stride"))
@partial(jax.named_call, name="utils.env_trajectory_rollout")
def env_trajectory_rollout(
    env: "Environment", model: Callable, *, n: int, stride: int = 1, **kw: Any
) -> Tuple["Environment", "Environment"]:
    """
    Roll out a trajectory by applying `model` in chunks of `stride` steps and
    collecting the environment after each chunk.

    Parameters
    ----------
    env : Environment
        Initial environment pytree.
    model : Callable
        Callable with signature `model(obs, **kw) -> action`.
    n : int
        Number of chunks to roll out. Total internal steps = `n * stride`.
    stride : int
        Steps per chunk between recorded snapshots.
    **kw : Any
        Extra keyword arguments passed to `model` on every step.

    Returns
    -------
    Environment
        Environment after `n * stride` steps.
    Environment
        Stacked pytree of environments with length `n`, each snapshot taken
        after a chunk of `stride` steps.

    Examples
    --------
    >>> env, traj = env_trajectory_rollout(env, model, n=100, stride=5, objective=goal)
    """

    def body(env, _):
        env = env_step(env, model, n=stride, **kw)
        return env, env

    env, env_traj = jax.lax.scan(body, env, length=n, xs=None)
    return env, env_traj


@staticmethod
@partial(jax.jit, static_argnames=("model", "n"))
@partial(jax.named_call, name="utils.env_step")
def env_step(
    env: "Environment", model: Callable, *, n: int = 1, **kw: Any
) -> "Environment":
    """
    Advance the environment `n` steps using actions from `model`.

    Parameters
    ----------
    env : Environment
        Initial environment pytree (batchable).
    model : Callable
        Callable with signature `model(obs, **kw) -> action`.
    n : int
        Number of steps to perform.
    **kw : Any
        Extra keyword arguments forwarded to `model`.

    Returns
    -------
    Environment
        Environment after `n` steps.

    Examples
    --------
    >>> env = env_step(env, model, n=10, objective=goal)
    """

    def body(env, _):
        return _env_step(env, model, **kw), None

    env, _ = jax.lax.scan(body, env, length=n, xs=None)
    return env


@staticmethod
@partial(jax.jit, static_argnames=("model",))
@partial(jax.named_call, name="utils._env_step")
def _env_step(env: "Environment", model: Callable, **kw: Any) -> "Environment":
    """
    Single environment step driven by `model`.

    Parameters
    ----------
    env : Environment
        Current environment pytree.
    model : Callable
        Callable with signature `model(obs, **kw) -> action`.
    **kw : Any
        Extra keyword arguments passed to `model`.

    Returns
    -------
    Environment
        Updated environment after applying `env.step(env, action)`.
    """
    obs = env.observation(env)
    action = model(obs, **kw)
    env = env.step(env, action)
    return env
