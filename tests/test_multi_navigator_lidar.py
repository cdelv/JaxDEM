"""Multi-navigator LiDAR senses other agents and walls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import vectorise_env


@pytest.mark.parametrize("num_envs", [None, 2])
def test_lidar_senses_agents_and_walls_independently_of_objectives(num_envs):
    env = Environment.create("multiNavigator", N=2, n_lidar_rays=8, lidar_range=10.0)
    key = jax.random.key(72)
    if num_envs is not None:
        env = vectorise_env(env, n=num_envs)
        key = jax.random.split(key, num_envs)
    env = env.reset(env, key)
    env.state.pos_c = jnp.broadcast_to(
        jnp.array([[0.0, 10.0], [4.0, 10.0]]), env.state.pos_c.shape
    )
    before = env.observation(env)
    assert env.observation_space_size == 14
    assert before.shape == (*env.state.pos_c.shape[:-1], env.observation_space_size)
    # East: the other agent at distance 4. West: the wall at x=-2.5.
    np.testing.assert_allclose(before[..., 0, 6 + 4], 0.6)
    np.testing.assert_allclose(before[..., 0, 6], 0.75)
    env.env_params["objective"] = env.env_params["objective"] + 1.0
    np.testing.assert_array_equal(env.observation(env)[..., 6:], before[..., 6:])


def test_default_lidar_observation_size():
    env = Environment.create("multiNavigator", N=2)
    env = env.reset(env, jax.random.key(74))
    assert env.observation_space_size == 22
    assert env.observation(env).shape == (2, 22)
