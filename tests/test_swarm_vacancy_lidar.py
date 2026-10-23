# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Opt-in vacancy sensing preserves defaults, reward, and local information."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from benchmarks.swarm_mean_shift.conservative_variants import make_environment
from benchmarks.swarm_mean_shift.variants import (
    vacant_objective_lidar as experimental_lidar,
)
from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.environments._swarm import filter_objectives, vacant_objective_lidar
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils import advance_action


@jax.jit
def _original_observation(env):
    peers, goals, ids = SwarmNavigator._sense(env)
    goals = filter_objectives(
        goals,
        peers,
        ids,
        env,
        n_elevation=env.n_lidar_elevation if env.state.dim == 3 else None,
    )
    lr = env.env_params["lidar_range"]
    return jnp.concatenate([env.state.vel, goals / lr, peers / lr], axis=-1)


def _reset(*, enabled=False, dim=2, goals=2, agents=2, **kwargs):
    env = SwarmNavigator.Create(
        N=agents,
        num_objectives=goals,
        dim=dim,
        vacancy_aware_lidar=enabled,
        **kwargs,
    )
    return env.reset(env, jax.random.key(12))


@pytest.mark.parametrize("dim", [2, 3])
def test_default_and_explicit_false_are_bit_exact_original_observations(dim):
    default = SwarmNavigator.Create(N=4, num_objectives=4, dim=dim)
    assert default.vacancy_aware_lidar is False
    default = default.reset(default, jax.random.key(9))
    explicit = _reset(enabled=False, dim=dim, agents=4, goals=4)
    explicit = explicit.reset(explicit, jax.random.key(9))
    np.testing.assert_array_equal(
        default.observation(default), _original_observation(default)
    )
    np.testing.assert_array_equal(
        default.observation(default), explicit.observation(explicit)
    )
    np.testing.assert_array_equal(default.reward(default), explicit.reward(explicit))
    action = jnp.full((4, dim), 0.2)
    default, _, _ = advance_action(default, action, skip_frames=2)
    np.testing.assert_array_equal(
        default.observation(default), _original_observation(default)
    )


@pytest.mark.parametrize("scale", [1, 2])
def test_opt_in_matches_experimental_vacancy_observations_and_reward(scale):
    reference = make_environment(f"vacancy_s{scale}", N=8, num_objectives=8, seed=15)
    env = _reset(enabled=True, agents=8, goals=8)
    # Compare identical physical arrays and history in each geometry; the
    # experimental reset itself already checks the exact 1x/2x transform.
    env.state = jax.tree.map(jnp.array, reference.state)
    env.system = jax.tree.map(jnp.array, reference.system)
    env.env_params = jax.tree.map(jnp.array, reference.env_params)
    np.testing.assert_array_equal(
        vacant_objective_lidar(env), experimental_lidar(reference)
    )
    np.testing.assert_array_equal(
        env.observation(env), reference.observation(reference)
    )
    np.testing.assert_array_equal(env.reward(env), reference.reward(reference))
    action = jnp.full((8, 2), 0.2)
    env, _, _ = advance_action(env, action, skip_frames=2)
    reference, _, _ = advance_action(reference, action, skip_frames=2)
    np.testing.assert_array_equal(env.state.pos, reference.state.pos)
    np.testing.assert_array_equal(env.reward(env), reference.reward(reference))
    np.testing.assert_array_equal(
        env.observation(env), reference.observation(reference)
    )
    assert env.observation_space_size == 26


def test_vacancy_reveals_occluded_goal_and_preserves_observers_own_goal():
    env = _reset(enabled=True)
    env.state.pos_c = jnp.asarray([[2.0, 2.0], [5.0, 2.0]])
    env.env_params["objective"] = jnp.asarray([[5.0, 2.0], [8.0, 2.0]])
    old = _original_observation(env)
    new = env.observation(env)
    np.testing.assert_array_equal(old[0, 2:14], 0)
    assert float(new[0, 2:14].max()) == (16 - 6) / 16
    assert float(new[1, 2:14].max()) == 1
    np.testing.assert_array_equal(new[:, :2], old[:, :2])
    np.testing.assert_array_equal(new[:, 14:], old[:, 14:])
    env = env.checkpoint(env, jnp.zeros((2, 2)))
    before = env.env_params["prev_potential"]
    env.vacancy_aware_lidar = False
    np.testing.assert_array_equal(env._potential(env), before)
    np.testing.assert_array_equal(env.reward(env), 0)


@pytest.mark.parametrize("dim", [2, 3])
def test_hide_false_returns_exact_raw_observation_even_with_opt_in(dim):
    enabled = _reset(enabled=True, dim=dim, hide_occupied_objectives=False)
    disabled = _reset(enabled=False, dim=dim, hide_occupied_objectives=False)
    # Include occupied and occluded goals rather than testing an empty mask.
    enabled.state.pos_c = jnp.asarray([[2.0] * dim, [5.0] + [2.0] * (dim - 1)])
    enabled.env_params["objective"] = jnp.asarray(
        [[5.0] + [2.0] * (dim - 1), [8.0] + [2.0] * (dim - 1)]
    )
    disabled.state = jax.tree.map(jnp.array, enabled.state)
    disabled.env_params = jax.tree.map(jnp.array, enabled.env_params)
    np.testing.assert_array_equal(
        enabled.observation(enabled), disabled.observation(disabled)
    )
    np.testing.assert_array_equal(
        enabled.observation(enabled), _original_observation(enabled)
    )
    np.testing.assert_array_equal(enabled.reward(enabled), disabled.reward(disabled))
    # Existing runtime switches still work without reconstructing the env.
    enabled.env_params["hide_occupied_objectives"] = jnp.asarray(True)
    assert not np.array_equal(
        enabled.observation(enabled), disabled.observation(disabled)
    )


@pytest.mark.parametrize("peer_distance", [16.0, 16.2, 35.0])
def test_invisible_peer_cannot_hide_a_visible_goal(peer_distance):
    env = _reset(enabled=True, goals=1)
    env.state.pos_c = jnp.asarray([[0.0, 0.0], [peer_distance, 0.0]])
    env.env_params["objective"] = jnp.asarray([[15.8, 0.0]])
    # A peer on or just beyond range may physically cover this goal, but is
    # unavailable to the observer. Its position cannot influence the scan.
    result = env.observation(env)[0]
    assert float(result[2:14].max()) > 0
    env.state.pos_c = env.state.pos_c.at[1].set(jnp.asarray([40.0, 40.0]))
    np.testing.assert_array_equal(env.observation(env)[0], result)


def test_nonlocal_goal_and_peer_positions_do_not_influence_observer():
    env = _reset(enabled=True, agents=3, goals=3)
    env.state.pos_c = jnp.asarray([[2.0, 2.0], [6.0, 2.0], [32.0, 32.0]])
    env.env_params["objective"] = jnp.asarray([[6.0, 2.0], [10.0, 2.0], [33.0, 32.0]])
    before = np.asarray(env.observation(env)[0])
    env.state.pos_c = env.state.pos_c.at[2].set(jnp.asarray([42.0, 42.0]))
    env.env_params["objective"] = (
        env.env_params["objective"].at[2].set(jnp.asarray([46.0, 43.0]))
    )
    np.testing.assert_array_equal(env.observation(env)[0], before)


def test_peer_occupancy_uses_its_own_radius_and_has_no_half_range_limit():
    env = _reset(enabled=True, goals=1)
    env.state.pos_c = jnp.asarray([[0.0, 0.0], [12.0, 0.0]])
    env.env_params["objective"] = jnp.asarray([[13.5, 0.0]])
    # Goal is beyond half range and 1.5 units from the visible peer.
    env.state.rad = jnp.asarray([1.0, 2.0])
    np.testing.assert_array_equal(vacant_objective_lidar(env)[0], 0)
    env.state.rad = jnp.asarray([1.0, 1.0])
    assert float(vacant_objective_lidar(env)[0].max()) > 0


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("goals", [0, 3])
def test_shapes_empty_goals_and_vectorized_reset_are_finite(dim, goals):
    env = _reset(enabled=True, dim=dim, goals=goals)
    assert env.observation(env).shape == (2, env.observation_space_size)
    np.testing.assert_array_equal(vacant_objective_lidar(env), experimental_lidar(env))
    assert np.isfinite(env.observation(env)).all()
    if goals == 0:
        np.testing.assert_array_equal(env.reward(env), 0)
        np.testing.assert_array_equal(
            env.observation(env)[
                :, dim : dim + (env.observation_space_size - dim) // 2
            ],
            0,
        )
    batch = vectorise_env(env, n=2)
    batch = batch.reset(batch, jax.random.split(jax.random.key(3), 2))
    assert batch.observation(batch).shape == (2, 2, env.observation_space_size)
    assert batch.vacancy_aware_lidar is True
    assert np.isfinite(batch.observation(batch)).all()
