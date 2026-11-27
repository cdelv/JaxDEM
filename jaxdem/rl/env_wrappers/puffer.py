# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Vectorized GPU environment wrapper for external un-JITted RL trainers (e.g. PufferLib)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Any, cast

from ..action_spaces import ActionSpace
from ..environments import Environment
from ...utils.environment import _mask_agent_actions, advance_action
from . import clip_action_env, vectorise_env


class PufferEnvWrapper:
    """Wrap a JaxDEM :class:`Environment` for an un-JITted external trainer loop.

    The outer rollout loop stays in Python/C++ (no JAX FFI callbacks), while
    each environment step runs as a single fused ``@jax.jit`` call on GPU:

    1. Optional continuous action transform (e.g. :class:`MaxNormSpace` or
       :class:`BoxSpace` forward map) so the external trainer can operate in
       unconstrained Gaussian latent coordinates with exact importance ratios.
    2. ``advance_action(..., skip_frames=skip_frames)`` (physics + checkpoint).
    3. Endpoint ``reward``, ``done = terminated | truncated``, and GPU-side
       episode return / step reward tracking.
    4. Zero-copy 0/1 ``while_loop`` ``reset_if_done`` on finished environments.
    5. Next ``observation`` returned as contiguous ``float32`` GPU arrays shaped
       ``(total_agents, obs_dim)``, ``(total_agents,)``, ``(total_agents,)``.

    Parameters
    ----------
    env : Environment
        Unvectorized JaxDEM RL environment instance.
    num_envs : int
        Number of parallel environment replicas.
    skip_frames : int, optional
        Extra physics frames per policy decision (``1 + skip_frames`` physics
        steps per ``step()`` call). Default ``0``.
    action_space : ActionSpace, optional
        Optional bijector whose ``forward`` maps unconstrained trainer actions
        into the environment's valid action domain.
    clip_actions : bool, optional
        If ``True`` (and ``action_space`` is ``None``), wrap the environment
        with :func:`clip_action_env`. Default ``False``.
    clip_range : tuple[float, float], optional
        Clipping bounds when ``clip_actions=True``. Default ``(-1.0, 1.0)``.
    reward_scale : float, optional
        Scalar multiplier applied to rewards returned to the trainer. Useful
        when the external trainer clamps per-step rewards (e.g. PufferLib's
        ``[-1, 1]`` clamp). Default ``1.0``.
    seed : int, optional
        PRNG seed used when ``key`` is not provided. Default ``1``.
    key : jax.Array, optional
        Explicit JAX PRNG key.
    """

    def __init__(
        self,
        env: Environment,
        *,
        num_envs: int,
        skip_frames: int = 0,
        action_space: ActionSpace | None = None,
        clip_actions: bool = False,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        discrete: bool = False,
        reward_scale: float = 1.0,
        seed: int = 1,
        key: ArrayLike | None = None,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if (
            isinstance(skip_frames, bool)
            or not isinstance(skip_frames, int)
            or skip_frames < 0
        ):
            raise ValueError(
                f"skip_frames must be a nonnegative integer, got {skip_frames!r}"
            )
        if clip_actions and action_space is not None:
            raise ValueError("Pass either action_space or clip_actions=True, not both.")
        if discrete and (action_space is not None or clip_actions):
            raise ValueError(
                "discrete=True cannot be combined with continuous action_space or clip_actions."
            )

        if clip_actions:
            env = clip_action_env(
                env, min_val=float(clip_range[0]), max_val=float(clip_range[1])
            )

        self.num_envs = num_envs
        self.skip_frames = skip_frames
        self.max_num_agents = int(env.max_num_agents)
        self.total_agents = self.num_envs * self.max_num_agents
        self.obs_size = int(env.observation_space_size)
        self.action_shape = tuple(int(d) for d in env.action_space_shape)
        self.action_size = int(env.action_space_size)
        self.action_space = action_space
        self.discrete = bool(discrete)
        self.reward_scale = float(reward_scale)

        self._vec_env: Environment = vectorise_env(env, n=num_envs)
        self._key: jax.Array = (
            jax.random.key(int(seed)) if key is None else jnp.asarray(key)
        )
        self._device = self._key.device

        # GPU-resident episode & step reward accumulators (read only at log intervals).
        # Explicitly committed to self._device so consume_metrics() resets never
        # change JAX argument sharding and trigger recompilation.
        self._ep_returns = jax.device_put(
            jnp.zeros((num_envs, self.max_num_agents), dtype=jnp.float32), self._device
        )
        self._completed_return_sum = jax.device_put(
            jnp.asarray(0.0, dtype=jnp.float32), self._device
        )
        self._completed_episodes = jax.device_put(
            jnp.asarray(0, dtype=jnp.int32), self._device
        )
        self._step_reward_sum = jax.device_put(
            jnp.asarray(0.0, dtype=jnp.float32), self._device
        )
        self._step_count = jax.device_put(
            jnp.asarray(0, dtype=jnp.int32), self._device
        )

        total_agents = self.total_agents
        obs_size = self.obs_size
        num_agents = self.max_num_agents
        action_shape = self.action_shape
        reward_scale_f = jnp.asarray(self.reward_scale, dtype=jnp.float32)
        act_space = self.action_space
        is_discrete = self.discrete

        @jax.jit
        def _jit_reset(
            vec_env: Environment, rng_key: jax.Array
        ) -> tuple[Environment, jax.Array, jax.Array, jax.Array, jax.Array]:
            rng_key, subkey = jax.random.split(rng_key)
            subkeys = jax.random.split(subkey, num_envs)
            vec_env = vec_env.reset(vec_env, subkeys)
            obs = jnp.asarray(vec_env.observation(vec_env), dtype=jnp.float32).reshape(
                total_agents, obs_size
            )
            rew = jnp.zeros((total_agents,), dtype=jnp.float32)
            term = jnp.zeros((total_agents,), dtype=jnp.float32)
            return vec_env, obs, rew, term, rng_key

        @jax.jit(donate_argnums=(0, 2, 3))
        def _jit_step(
            vec_env: Environment,
            actions_flat: jax.Array,
            rng_key: jax.Array,
            ep_returns: jax.Array,
            completed_return_sum: jax.Array,
            completed_episodes: jax.Array,
            step_reward_sum: jax.Array,
            step_count: jax.Array,
        ) -> tuple[
            Environment,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ]:
            rng_key, reset_root = jax.random.split(rng_key)
            if is_discrete:
                action = actions_flat.reshape(num_envs, num_agents)
            else:
                action = actions_flat.reshape(num_envs, num_agents, *action_shape)
                if act_space is not None:
                    action = cast(Any, act_space).forward(action)

            agent_mask = vec_env.agent_mask(vec_env)
            action = _mask_agent_actions(action, agent_mask)

            vec_env, terminated, truncated = advance_action(
                vec_env, action, skip_frames=skip_frames
            )
            done = terminated | truncated
            raw_reward = jnp.asarray(vec_env.reward(vec_env), dtype=jnp.float32)
            raw_reward = jnp.where(agent_mask, raw_reward, 0.0)

            next_agent_mask = vec_env.agent_mask(vec_env)
            agent_deactivated = agent_mask & ~next_agent_mask
            term_agents = (
                jnp.broadcast_to(terminated[..., None], raw_reward.shape) & agent_mask
            ) | agent_deactivated
            trunc_agents = (
                jnp.broadcast_to(truncated[..., None], raw_reward.shape) & agent_mask
            ) & ~term_agents
            done_agents = term_agents | trunc_agents

            # Update GPU-resident metrics over active agents before resetting.
            active_count = jnp.maximum(jnp.sum(agent_mask.astype(jnp.float32)), 1.0)
            ep_returns = jnp.where(agent_mask, ep_returns + raw_reward, ep_returns)
            completed_return_sum = completed_return_sum + jnp.sum(
                jnp.where(done_agents, ep_returns, 0.0)
            )
            completed_episodes = completed_episodes + jnp.sum(
                done_agents.astype(jnp.int32)
            )
            ep_returns = jnp.where(done_agents | ~agent_mask, 0.0, ep_returns)
            step_reward_sum = step_reward_sum + jnp.sum(raw_reward) / active_count
            step_count = step_count + jnp.asarray(1, dtype=jnp.int32)

            # Reset finished environments in-place via a zero-copy 0/1 while_loop.
            _, vec_env = jax.lax.while_loop(
                lambda carry: carry[0],
                lambda carry: (
                    jnp.asarray(False),
                    carry[1].reset_if_done(
                        carry[1], done, jax.random.split(reset_root, num_envs)
                    ),
                ),
                (jnp.any(done), vec_env),
            )

            next_obs = jnp.asarray(
                vec_env.observation(vec_env), dtype=jnp.float32
            ).reshape(total_agents, obs_size)
            rew_flat = (raw_reward * reward_scale_f).reshape(total_agents)
            term_flat = term_agents.astype(jnp.float32).reshape(total_agents)
            trunc_flat = trunc_agents.astype(jnp.float32).reshape(total_agents)
            return (
                vec_env,
                next_obs,
                rew_flat,
                term_flat,
                trunc_flat,
                rng_key,
                ep_returns,
                completed_return_sum,
                completed_episodes,
                step_reward_sum,
                step_count,
            )

        self._jit_reset = _jit_reset
        self._jit_step = _jit_step

    @property
    def env(self) -> Environment:
        """Current vectorized JaxDEM environment state."""
        return self._vec_env

    def reset(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Reset all environments and return ``(obs, rewards, terminals)`` on GPU."""
        self._vec_env, obs, rew, term, self._key = self._jit_reset(
            self._vec_env, self._key
        )
        self._device = obs.device
        self._ep_returns = jax.device_put(
            jnp.zeros((self.num_envs, self.max_num_agents), dtype=jnp.float32),
            self._device,
        )
        self._completed_return_sum = jax.device_put(
            jnp.asarray(0.0, dtype=jnp.float32), self._device
        )
        self._completed_episodes = jax.device_put(
            jnp.asarray(0, dtype=jnp.int32), self._device
        )
        self._step_reward_sum = jax.device_put(
            jnp.asarray(0.0, dtype=jnp.float32), self._device
        )
        self._step_count = jax.device_put(
            jnp.asarray(0, dtype=jnp.int32), self._device
        )
        return obs, rew, term

    def step_split(
        self, actions: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Advance all environments and return ``(obs, rew, terminated, truncated)``."""
        (
            self._vec_env,
            obs,
            rew,
            term,
            trunc,
            self._key,
            self._ep_returns,
            self._completed_return_sum,
            self._completed_episodes,
            self._step_reward_sum,
            self._step_count,
        ) = self._jit_step(
            self._vec_env,
            actions,
            self._key,
            self._ep_returns,
            self._completed_return_sum,
            self._completed_episodes,
            self._step_reward_sum,
            self._step_count,
        )
        return obs, rew, term, trunc

    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Advance all environments by one policy decision and return ``(obs, rew, done)``."""
        obs, rew, term, trunc = self.step_split(actions)
        return obs, rew, jnp.maximum(term, trunc)

    def consume_metrics(self) -> dict[str, float]:
        """Fetch and reset accumulated episode/step reward metrics (one host sync)."""
        active_mask = self._vec_env.agent_mask(self._vec_env)
        active_ep_mean = jnp.sum(
            jnp.where(active_mask, self._ep_returns, 0.0)
        ) / jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
        ret_sum, ep_cnt, step_rew_sum, steps, cur_ep_mean = jax.device_get(
            (
                self._completed_return_sum,
                self._completed_episodes,
                self._step_reward_sum,
                self._step_count,
                active_ep_mean,
            )
        )
        self._completed_return_sum = jax.device_put(
            jnp.asarray(0.0, dtype=jnp.float32), self._device
        )
        self._completed_episodes = jax.device_put(
            jnp.asarray(0, dtype=jnp.int32), self._device
        )
        self._step_reward_sum = jax.device_put(
            jnp.asarray(0.0, dtype=jnp.float32), self._device
        )
        self._step_count = jax.device_put(
            jnp.asarray(0, dtype=jnp.int32), self._device
        )

        episodes = int(ep_cnt)
        steps_i = int(steps)
        return {
            "episodes": float(episodes),
            "episode_return": float(ret_sum) / episodes if episodes > 0 else float("nan"),
            "active_return": float(cur_ep_mean),
            "step_reward": float(step_rew_sum) / max(steps_i, 1),
        }

    def check_overflow(self) -> bool:
        """Return ``True`` if any environment collider reported an overflow."""
        return bool(jax.device_get(jnp.any(self._vec_env.system.collider.overflow)))


__all__ = ["PufferEnvWrapper"]
