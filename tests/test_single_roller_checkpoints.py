"""Rolling rewards use action-start distance and the live physics endpoint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils.environment import advance_action


def moving_roller(num_envs=None):
    env = Environment.create("SingleRoller")
    key = jax.random.key(22)
    if num_envs is not None:
        env = vectorise_env(env, n=num_envs)
        key = jax.random.split(key, num_envs)
    env = env.reset(env, key)
    env.state.pos_c = jnp.broadcast_to(jnp.array([10.0, 10.0, 1.0]), env.state.pos_c.shape)
    env.env_params["objective"] = env.state.pos_c + jnp.array([2.0, 0.0, 0.0])
    env.state.vel = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), env.state.vel.shape)
    # Leave the reset baseline stale: advance_action must capture the live start.
    return env


def distance(env):
    return jnp.linalg.norm(env.state.pos_c - env.env_params["objective"], axis=-1)


@pytest.mark.parametrize("skip_frames", [0, 1, 49])
def test_reward_spans_repeated_actions_and_telescopes(skip_frames):
    initial = moving_roller()
    action = jnp.array([[0.0, 0.2, 0.0]])
    first, _, _ = advance_action(initial, action, skip_frames=skip_frames)
    second, _, _ = advance_action(first, action, skip_frames=skip_frames)
    whole, _, _ = advance_action(initial, action, skip_frames=2 * skip_frames + 1)

    for before, after in [(initial, first), (first, second)]:
        np.testing.assert_allclose(after.env_params["prev_dist"], distance(before))
        np.testing.assert_allclose(
            after.reward(after), jnp.exp(-2 * distance(after)) - jnp.exp(-2 * distance(before)),
            rtol=2e-5, atol=1e-7,
        )
    np.testing.assert_allclose(second.state.pos_c, whole.state.pos_c)
    np.testing.assert_allclose(first.reward(first) + second.reward(second), whole.reward(whole), atol=1e-7)

    checkpoint = initial.checkpoint(initial, action)
    raw = jax.lax.fori_loop(0, skip_frames + 1, lambda _, e: e.step(e, action), checkpoint)
    np.testing.assert_array_equal(raw.env_params["prev_dist"], checkpoint.env_params["prev_dist"])
    np.testing.assert_allclose(raw.observation(raw), first.observation(first), atol=1e-12)
    np.testing.assert_allclose(raw.reward(raw), first.reward(first), atol=1e-12)


def test_live_distance_reward_observations_and_reset():
    env = moving_roller()
    action = jnp.zeros((1, 3))
    env = env.checkpoint(env, action)
    np.testing.assert_array_equal(env.reward(env), 0.0)
    env.state.pos_c = env.state.pos_c.at[0, 0].add(1.0)
    env.state.vel = jnp.array([[3.0, 4.0, 5.0]])
    env.state.ang_vel = jnp.array([[6.0, 7.0, 8.0]])
    np.testing.assert_allclose(env.reward(env), [np.exp(-2.0) - np.exp(-4.0)])
    np.testing.assert_allclose(env.observation(env), [[-1.0, 0.0, -1.0, 0.0, 3.0, 4.0, 6.0, 7.0, 8.0]])
    env = env.checkpoint(env, action)
    env.state.vel = -env.state.vel
    env.state.ang_vel = -env.state.ang_vel
    np.testing.assert_array_equal(env.reward(env), 0.0)

    env.state.pos_c = env.state.pos_c.at[0, 0].add(-1.0)
    np.testing.assert_allclose(env.reward(env), [np.exp(-4.0) - np.exp(-2.0)])
    env.env_params["objective"] = env.state.pos_c
    observation = np.asarray(env.observation(env))
    assert observation.shape == (1, 9)
    assert np.isfinite(observation).all()
    np.testing.assert_array_equal(observation[..., :4], 0.0)
    env = env.checkpoint(env, action)
    np.testing.assert_array_equal(env.reward(env), 0.0)
    reset = env.reset(env, jax.random.key(31))
    np.testing.assert_allclose(reset.env_params["prev_dist"], distance(reset))
    np.testing.assert_array_equal(reset.reward(reset), 0.0)


def test_vectorized_truncation_uses_each_actual_endpoint():
    initial = moving_roller(num_envs=3)
    initial.env_params["max_steps"] = jnp.array([2, 4, 100])
    action = jnp.broadcast_to(jnp.array([0.0, 0.2, 0.0]), (3, 1, 3))
    final, terminated, truncated = advance_action(initial, action, skip_frames=4)
    np.testing.assert_array_equal(final.system.step_count, [2, 4, 5])
    np.testing.assert_array_equal(terminated, [False, False, False])
    np.testing.assert_array_equal(truncated, [True, True, False])
    np.testing.assert_allclose(
        final.reward(final), jnp.exp(-2 * distance(final)) - jnp.exp(-2 * distance(initial)),
        rtol=2e-5, atol=1e-7,
    )
    continued, _, _ = advance_action(final, action, skip_frames=4,
                                   terminated=terminated, truncated=truncated)
    np.testing.assert_array_equal(continued.system.step_count, [2, 4, 10])
    np.testing.assert_array_equal(continued.reward(continued)[:2], 0.0)
    np.testing.assert_array_equal(continued.observation(continued)[:2], final.observation(final)[:2])
