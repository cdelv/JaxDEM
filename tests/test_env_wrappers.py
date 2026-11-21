# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for RL environment wrappers using synthetic environments."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.env_wrappers import (
    clip_action_env,
    is_wrapped,
    unwrap,
    vectorise_env,
)
from jaxdem.rl.environments import Environment
from jaxdem.state import State
from jaxdem.system import System
from jaxdem.utils.environment import advance_action


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _ActionProbeEnv(Environment):
    """Minimal synthetic 2-agent environment (slot 0 active, slot 1 inactive)."""

    @classmethod
    def Create(cls, dim: int = 2) -> _ActionProbeEnv:
        state = State.create(pos=jnp.zeros((2, dim)))
        return cls(
            state=state,
            system=System.create(state.shape),
            env_params={
                "seed": jnp.asarray(0.0),
                "step_count": jnp.asarray(0, dtype=jnp.int32),
                "checkpoint_action": jnp.full((2, 2), -99.0),
                "step_action": jnp.full((2, 2), -99.0),
            },
        )

    @staticmethod
    def reset(env: _ActionProbeEnv, key: jax.Array) -> _ActionProbeEnv:
        seed = jax.random.uniform(key, shape=(), minval=0.0, maxval=10.0)
        return replace(
            env,
            env_params={
                "seed": seed,
                "step_count": jnp.asarray(0, dtype=jnp.int32),
                "checkpoint_action": jnp.zeros((2, 2)),
                "step_action": jnp.zeros((2, 2)),
            },
        )

    @staticmethod
    def checkpoint(env: _ActionProbeEnv, action: jax.Array) -> _ActionProbeEnv:
        params = dict(env.env_params)
        params["checkpoint_action"] = action
        return replace(env, env_params=params)

    @staticmethod
    def step(env: _ActionProbeEnv, action: jax.Array) -> _ActionProbeEnv:
        params = dict(env.env_params)
        params["step_count"] = params["step_count"] + 1
        params["step_action"] = action
        return replace(env, env_params=params)

    @staticmethod
    def observation(env: _ActionProbeEnv) -> jax.Array:
        return env.env_params["step_action"] + env.env_params["seed"]

    @staticmethod
    def reward(env: _ActionProbeEnv) -> jax.Array:
        return jnp.sum(env.env_params["step_action"], axis=-1)

    @staticmethod
    def agent_mask(env: _ActionProbeEnv) -> jax.Array:
        del env
        return jnp.array([True, False], dtype=bool)


def test_clip_action_env_clips_and_masks_inactive_slots_for_checkpoint_and_step():
    # Zero is intentionally outside [0.5, 2.0] so we verify inactive slots stay 0.0.
    env = clip_action_env(_ActionProbeEnv.Create(), min_val=0.5, max_val=2.0)
    raw_action = jnp.array([[-3.0, 5.0], [1.2, 4.0]], dtype=float)
    expected = np.array([[0.5, 2.0], [0.0, 0.0]])

    checked = env.checkpoint(env, raw_action)
    np.testing.assert_allclose(checked.env_params["checkpoint_action"], expected)

    stepped = env.step(checked, raw_action)
    np.testing.assert_allclose(stepped.env_params["step_action"], expected)
    np.testing.assert_allclose(stepped.env_params["checkpoint_action"], expected)

    # Integer actions must be rejected on both checkpoint and step.
    int_action = jnp.array([[1, 2], [0, 0]], dtype=jnp.int32)
    with pytest.raises(TypeError, match="floating-point"):
        env.step(env, int_action)
    with pytest.raises(TypeError, match="floating-point"):
        env.checkpoint(env, int_action)


@pytest.mark.parametrize(
    ("min_val", "max_val"),
    [(2.0, 1.0), (float("nan"), 1.0), (-1.0, float("inf"))],
)
def test_clip_action_env_rejects_invalid_bounds(min_val, max_val):
    with pytest.raises(ValueError, match="finite bounds"):
        clip_action_env(_ActionProbeEnv.Create(), min_val=min_val, max_val=max_val)


def test_vectorise_env_matches_independent_runs_and_rejects_invalid_n():
    base = _ActionProbeEnv.Create()
    vec = vectorise_env(base, n=3)
    keys = jax.random.split(jax.random.key(11), 3)
    vec = vec.reset(vec, keys)
    actions = jnp.arange(12, dtype=float).reshape(3, 2, 2)
    vec = vec.step(vec, actions)

    expected_obs = []
    expected_rew = []
    for idx in range(3):
        scalar = base.reset(base, keys[idx])
        scalar = scalar.step(scalar, actions[idx])
        expected_obs.append(scalar.observation(scalar))
        expected_rew.append(scalar.reward(scalar))

    np.testing.assert_allclose(vec.observation(vec), jnp.stack(expected_obs, axis=0))
    np.testing.assert_allclose(vec.reward(vec), jnp.stack(expected_rew, axis=0))

    for bad_n in (0, -2, True, 1.5):
        with pytest.raises(ValueError, match="positive Python integer"):
            vectorise_env(base, n=bad_n)  # type: ignore[arg-type]


def test_wrapper_composition_advance_action_unwrap_and_treedef_caching():
    base = _ActionProbeEnv.Create()
    assert not is_wrapped(base)
    assert unwrap(base) is base

    wrapped_a = vectorise_env(clip_action_env(base, min_val=0.25, max_val=1.5), n=2)
    wrapped_b = vectorise_env(clip_action_env(base, min_val=0.25, max_val=1.5), n=2)
    wrapped_diff = vectorise_env(clip_action_env(base, min_val=-1.0, max_val=1.0), n=2)

    assert is_wrapped(wrapped_a)
    # _WRAP_CACHE must reuse the class and pytree treedef for identical configs.
    assert type(wrapped_a) is type(wrapped_b)
    assert jax.tree_util.tree_structure(wrapped_a) == jax.tree_util.tree_structure(
        wrapped_b
    )
    assert type(wrapped_a) is not type(wrapped_diff)

    keys = jax.random.split(jax.random.key(3), 2)
    wrapped_a = wrapped_a.reset(wrapped_a, keys)
    raw = jnp.array(
        [[[-1.0, 9.0], [1.0, 1.0]], [[0.5, 0.1], [2.0, -2.0]]], dtype=float
    )
    advanced, terminated, truncated = advance_action(wrapped_a, raw, skip_frames=2)

    expected = np.array(
        [[[0.25, 1.5], [0.0, 0.0]], [[0.5, 0.25], [0.0, 0.0]]]
    )
    np.testing.assert_array_equal(terminated, [False, False])
    np.testing.assert_array_equal(truncated, [False, False])
    np.testing.assert_array_equal(advanced.env_params["step_count"], [3, 3])
    np.testing.assert_allclose(advanced.env_params["checkpoint_action"], expected)
    np.testing.assert_allclose(advanced.env_params["step_action"], expected)

    # unwrap restores the original base class while keeping all field values.
    unwrapped = unwrap(advanced)
    assert type(unwrapped) is _ActionProbeEnv
    assert not is_wrapped(unwrapped)
    np.testing.assert_array_equal(unwrapped.env_params["step_count"], [3, 3])
    np.testing.assert_allclose(unwrapped.env_params["step_action"], expected)
