"""Goal-weighted kinetic energy uses both endpoints of each policy action."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils.environment import advance_action


def make_env(num_envs=None):
    env = Environment.create(
        "multiNavigator", N=2, kinetic_energy_coeff=0.5, kinetic_energy_scale=2.0
    )
    key = jax.random.key(73)
    if num_envs is not None:
        env = vectorise_env(env, n=num_envs)
        key = jax.random.split(key, num_envs)
    env = env.reset(env, key)
    np.testing.assert_array_equal(env.reward(env), 0.0)
    env.state.pos_c = jnp.broadcast_to(
        jnp.array([[10.0, 5.0], [10.0, 15.0]]), env.state.pos_c.shape
    )
    env.env_params["objective"] = env.state.pos_c
    env.state.vel = jnp.broadcast_to(jnp.array([1.0, 0.0]), env.state.vel.shape)
    return env


def potential(env):
    distance = jnp.linalg.norm(env.state.pos_c - env.env_params["objective"], axis=-1)
    energy = 0.5 * env.state.mass * jnp.sum(env.state.vel**2, axis=-1)
    return jnp.exp(-((distance / (2 * env.state.rad)) ** 4) - 0.5 * energy / 2.0)


def test_slowing_near_goal_is_rewarded_and_far_energy_effect_is_small():
    env = make_env()
    env.state.mass = jnp.array([1.0, 2.0])
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    env.state.vel = jnp.zeros_like(env.state.vel)
    expected = 1 - np.exp(-np.array([0.125, 0.25]))
    np.testing.assert_allclose(env.reward(env), expected)
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    env.state.vel = jnp.ones_like(env.state.vel).at[:, 1].set(0.0)
    np.testing.assert_allclose(env.reward(env), -expected)

    env.state.pos_c = env.state.pos_c.at[:, 0].add(4.0)
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    env.state.vel = jnp.zeros_like(env.state.vel)
    np.testing.assert_allclose(env.reward(env), np.exp(-16) * expected)
    assert np.max(env.reward(env)) < 3e-8


@pytest.mark.parametrize("skip_frames", [0, 4, 49])
def test_checkpoint_energy_is_fixed_and_action_rewards_telescope(skip_frames):
    env = make_env()
    action = jnp.array([[0.2, 0.0], [-0.3, 0.1]])
    first, _, _ = advance_action(env, action, skip_frames=skip_frames)
    second, _, _ = advance_action(first, -action, skip_frames=skip_frames)
    for before, after in [(env, first), (first, second)]:
        np.testing.assert_allclose(
            after.env_params["prev_potential"], potential(before)
        )
        np.testing.assert_allclose(
            after.reward(after), potential(after) - potential(before), atol=1e-12
        )
    np.testing.assert_allclose(
        first.reward(first) + second.reward(second),
        potential(second) - potential(env),
        atol=1e-12,
    )
    checkpoint = env.checkpoint(env, action)
    raw = jax.lax.fori_loop(
        0, skip_frames + 1, lambda _, e: e.step(e, action), checkpoint
    )
    np.testing.assert_array_equal(
        raw.env_params["prev_potential"], checkpoint.env_params["prev_potential"]
    )
    np.testing.assert_allclose(raw.reward(raw), first.reward(first), atol=1e-12)
    reset = second.reset(second, jax.random.key(74))
    np.testing.assert_array_equal(reset.reward(reset), 0.0)


def test_vectorized_truncation_preserves_potential_and_actuator_history():
    env = make_env(num_envs=3)
    env.env_params["max_steps"] = jnp.array([2, 4, 100])
    action = jnp.ones((3, 2, 2))
    final, terminated, truncated = advance_action(env, action, skip_frames=4)
    np.testing.assert_array_equal(final.system.step_count, [2, 4, 5])
    np.testing.assert_allclose(
        final.reward(final), potential(final) - potential(env), atol=1e-12
    )
    continued, _, _ = advance_action(
        final, -action, skip_frames=4, terminated=terminated, truncated=truncated
    )
    np.testing.assert_array_equal(continued.reward(continued)[:2], 0.0)
    np.testing.assert_array_equal(
        continued.env_params["applied_action"][:2],
        final.env_params["applied_action"][:2],
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("kinetic_energy_coeff", -1.0),
        ("kinetic_energy_coeff", float("nan")),
        ("kinetic_energy_coeff", float("inf")),
        ("kinetic_energy_scale", 0.0),
        ("kinetic_energy_scale", -1.0),
        ("kinetic_energy_scale", float("nan")),
        ("kinetic_energy_scale", float("inf")),
    ],
)
def test_invalid_energy_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        Environment.create("multiNavigator", **{name: value})
