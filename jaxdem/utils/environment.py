# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to handle environments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING, Callable, Tuple, Any
from functools import partial

if TYPE_CHECKING:
    from ..rl.environments import Environment


@partial(jax.jit, static_argnames=("model", "n", "stride"))
@partial(jax.named_call, name="utils.env_trajectory_rollout")
def env_trajectory_rollout(
    env: "Environment",
    model: Callable[[jax.Array, jax.Array, Any], Any],
    key: jax.Array,
    *,
    n: int,
    stride: int = 1,
    **kw: Any,
) -> Tuple["Environment", "Environment"]:
    """
    Roll out a trajectory by applying `model` in chunks of `stride` steps and
    collecting the environment after each chunk.

    Parameters
    ----------
    env : Environment
        Initial environment pytree.
    model : Callable
        Callable with signature `model(obs, key, **kw) -> action`.
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

    def body(
        carry: Tuple["Environment", jax.Array], _: None
    ) -> Tuple[Tuple["Environment", jax.Array], "Environment"]:
        env, key = carry
        key, subkey = jax.random.split(key)
        env = env_step(env, model, subkey, n=stride, **kw)
        return (env, key), env

    (env, key), env_traj = jax.lax.scan(body, (env, key), length=n, xs=None)
    return env, env_traj


@partial(jax.jit, static_argnames=("model", "n"))
@partial(jax.named_call, name="utils.env_step")
def env_step(
    env: "Environment",
    model: Callable[[jax.Array, jax.Array, Any], Any],
    key: jax.Array,
    *,
    n: int = 1,
    **kw: Any,
) -> "Environment":
    """
    Advance the environment `n` steps using actions from `model`.

    Parameters
    ----------
    env : Environment
        Initial environment pytree (batchable).
    model : Callable
        Callable with signature `model(obs, key, **kw) -> action`.
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

    def body(
        carry: Tuple["Environment", jax.Array], _: None
    ) -> Tuple[Tuple["Environment", jax.Array], None]:
        env, key = carry
        key, subkey = jax.random.split(key)
        env = _env_step(env, model, subkey, **kw)
        return (env, key), None

    (env, key), _ = jax.lax.scan(body, (env, key), length=n, xs=None)
    return env


@partial(jax.jit, static_argnames=("model",))
@partial(jax.named_call, name="utils._env_step")
def _env_step(
    env: "Environment",
    model: Callable[[jax.Array, jax.Array, Any], Any],
    key: jax.Array,
    **kw: Any,
) -> "Environment":
    """
    Single environment step driven by `model`.

    Parameters
    ----------
    env : Environment
        Current environment pytree.
    model : Callable
        Callable with signature `model(obs, key, **kw) -> action`.
    **kw : Any
        Extra keyword arguments passed to `model`.

    Returns
    -------
    Environment
        Updated environment after applying `env.step(env, action)`.
    """
    obs = env.observation(env)
    action = model(obs, key, **kw)
    env = env.step(env, action)
    return env


@jax.jit
@partial(jax.named_call, name="utils.lidar")
def lidar(env: "Environment") -> jax.Array:
    nbins = env.n_lidar_rays
    indices = jax.lax.iota(int, env.max_num_agents)

    def lidar_for_i(i: jax.Array) -> jax.Array:
        rij = jax.vmap(
            lambda j: env.system.domain.displacement(
                env.state.pos[i], env.state.pos[j], env.system
            )
        )(indices)
        r = jnp.vecdot(rij, rij)
        r = jnp.sqrt(r)
        r = r.at[i].set(jnp.inf)

        theta = jnp.arctan2(rij[..., 1], rij[..., 0])
        bins = jnp.floor((theta + jnp.pi) * (nbins / (2.0 * jnp.pi))).astype(int)

        d_in = jnp.where(r < env.env_params["lidar_range"], r, jnp.inf)
        d_bins = (
            jnp.full((nbins,), jnp.inf, dtype=env.state.pos.dtype).at[bins].min(d_in)
        )

        proximity = jnp.where(
            jnp.isfinite(d_bins),
            jnp.maximum(0.0, env.env_params["lidar_range"] - d_bins),
            0.0,
        )
        return proximity

    return jax.vmap(lidar_for_i)(indices)


__all__ = ["env_trajectory_rollout", "env_step", "lidar"]
