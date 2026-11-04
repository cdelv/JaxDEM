# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Focused geometry, local-sensing, and reward-history ablation checks."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from benchmarks.swarm_mean_shift.conservative_variants import (
    VARIANT_NAMES,
    cosine_shared_potential,
    make_environment,
)
from benchmarks.swarm_mean_shift.spacing_followup import (
    make_environment as make_spacing,
)
from benchmarks.swarm_mean_shift.variants import vacant_objective_lidar
from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.environments._swarm import shared_potential
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils import advance_action


@pytest.mark.parametrize("scale", [1, 2])
def test_baseline_matches_current_environment_and_prior_double_spacing(scale):
    key = jax.random.key(11)
    reference = (
        SwarmNavigator.Create(N=8, num_objectives=8, max_steps=200000)
        if scale == 1
        else make_spacing("spacing_baseline", N=8, num_objectives=8)
    )
    reference = reference.reset(reference, key)
    env = make_environment(
        f"baseline_s{scale}", N=8, num_objectives=8, max_steps=200000, seed=11
    )
    np.testing.assert_allclose(env.state.pos, reference.state.pos, rtol=2e-7, atol=1e-6)
    np.testing.assert_allclose(
        env.env_params["objective"],
        reference.env_params["objective"],
        rtol=2e-7,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        env.observation(env), reference.observation(reference), rtol=2e-6, atol=1e-6
    )
    np.testing.assert_allclose(env.reward(env), reference.reward(reference), atol=1e-7)
    for name in ("box_size", "anchor", "inv_box_size"):
        np.testing.assert_array_equal(
            getattr(env.system.domain, name), getattr(reference.system.domain, name)
        )
    action = jnp.ones((8, 2))
    env, _, _ = advance_action(env, action, skip_frames=2)
    reference, _, _ = advance_action(reference, action, skip_frames=2)
    np.testing.assert_allclose(env.state.pos, reference.state.pos, rtol=2e-7, atol=1e-6)
    np.testing.assert_allclose(env.reward(env), reference.reward(reference), atol=2e-7)


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_geometry_reset_idempotence_observation_shape_and_fixed_physics(variant):
    scale = int(variant[-1])
    env = make_environment(variant, N=4, num_objectives=4, seed=13)
    source = SwarmNavigator.Create(N=4, num_objectives=4)
    source = source.reset(source, jax.random.key(13))
    np.testing.assert_allclose(
        env.state.pos, scale * source.state.pos, rtol=2e-7, atol=1e-6
    )
    np.testing.assert_allclose(
        env.env_params["objective"],
        scale * source.env_params["objective"],
        rtol=2e-7,
        atol=1e-6,
    )
    source_goals = np.asarray(source.env_params["objective"])
    goals = np.asarray(env.env_params["objective"])
    distances = lambda x: np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    np.testing.assert_allclose(
        distances(goals), scale * distances(source_goals), rtol=2e-7, atol=4e-6
    )
    saved_positions = np.asarray(env.state.pos)
    saved_goals = np.asarray(env.env_params["objective"])
    for _ in range(2):
        env = env.reset(env, jax.random.key(13))
        np.testing.assert_array_equal(env.state.pos, saved_positions)
        np.testing.assert_array_equal(env.env_params["objective"], saved_goals)
        np.testing.assert_array_equal(
            env.system.domain.box_size, [30 * scale, 30 * scale]
        )
        np.testing.assert_allclose(env.reward(env), 0, atol=1e-7)
    np.testing.assert_array_equal(env.state.rad, source.state.rad)
    np.testing.assert_array_equal(env.system.dt, source.system.dt)
    for name in ("friction", "lidar_range", "sharing_range"):
        np.testing.assert_array_equal(env.env_params[name], source.env_params[name])
    assert env.observation_space_size == 26
    assert env.observation(env).shape == (4, 26)
    batch = vectorise_env(env, n=2)
    batch = batch.reset(batch, jax.random.split(jax.random.key(9), 2))
    assert batch.observation(batch).shape == (2, 4, 26)
    assert np.isfinite(batch.reward(batch)).all()


def _synthetic_scans():
    env = make_environment("cosine_s1", N=2, num_objectives=1, n_lidar_rays=4, seed=2)
    goals = jnp.asarray([[12.0, 0, 0, 0], [0, 0, 0, 0]])  # Distance 4, range 16.
    peers = jnp.zeros((2, 4))
    ids = jnp.asarray([[0, 0, 0, 0], [1, 1, 1, 1]])
    return env, goals, peers, ids


def test_cosine_analytic_values_peer_sharing_and_ignored_shape_controls():
    env, goals, peers, ids = _synthetic_scans()
    expected = np.exp(-((4 / 1.5) ** 4)) + 0.1 * (1 + np.cos(np.pi * 4 / 16)) / 2
    np.testing.assert_allclose(
        cosine_shared_potential(goals, peers, ids, env), [expected, 0], rtol=2e-6
    )
    # A visible peer exactly at the goal has one unit quartic claim.
    peers = peers.at[0, 0].set(12)
    ids = ids.at[0, 0].set(1)
    score = cosine_shared_potential(goals, peers, ids, env)
    np.testing.assert_allclose(score, [expected / 2, 0], rtol=2e-6)
    for name in ("attraction_constant", "attraction_quadratic", "attraction_decay"):
        env.env_params[name] = jnp.asarray(70.0)
    np.testing.assert_array_equal(
        cosine_shared_potential(goals, peers, ids, env), score
    )


def test_cosine_preserves_baseline_settling_and_claim_weights_exactly():
    env, _, _, _ = _synthetic_scans()
    env.env_params["attraction_coeff"] = jnp.asarray(0.0)
    goals = jnp.asarray([[15.7, 15.0, 13.0, 0.0], [12.0, 15.3, 0.0, 14.5]])
    peers = jnp.asarray([[14.0, 11.0, 0.0, 0.0], [10.0, 0.0, 13.0, 0.0]])
    ids = jnp.asarray([[1, -1, 0, 0], [0, 1, -1, 1]])
    np.testing.assert_array_equal(
        cosine_shared_potential(goals, peers, ids, env),
        shared_potential(goals, peers, ids, env),
    )


@pytest.mark.parametrize(
    "peer_id,proximity", [(-1, 12.0), (0, 12.0), (1, 0.0), (2, 12.0)]
)
def test_walls_self_empty_and_invalid_peer_returns_make_no_claim(peer_id, proximity):
    env, goals, peers, ids = _synthetic_scans()
    expected = cosine_shared_potential(goals, peers, ids, env)
    peers = peers.at[0, 0].set(proximity)
    ids = ids.at[0, 0].set(peer_id)
    np.testing.assert_array_equal(
        cosine_shared_potential(goals, peers, ids, env), expected
    )


def test_cosine_goes_continuously_to_zero_at_sensor_range():
    env, _, peers, ids = _synthetic_scans()

    def score(distance):
        goals = jnp.zeros((2, 4)).at[0, 0].set(jnp.maximum(16 - distance, 0))
        return cosine_shared_potential(goals, peers, ids, env)[0]

    np.testing.assert_array_equal(score(16.0), 0)
    np.testing.assert_array_equal(score(17.0), 0)
    assert float(score(15.99)) < 1e-6
    assert abs(float(jax.grad(score)(15.99999))) < 1e-7
    values = np.asarray(jax.vmap(score)(jnp.linspace(0, 16, 65)))
    assert np.all(np.diff(values) <= 1e-7)


@pytest.mark.parametrize("variant", ["cosine_s1", "cosine_vacancy_s2"])
def test_checkpoint_delta_history_and_locality(variant):
    env = make_environment(variant, N=2, num_objectives=2, seed=8)
    env.state.pos_c = jnp.asarray([[2.0, 2.0], [24.0, 24.0]])
    env.env_params["objective"] = jnp.asarray([[6.0, 2.0], [22.0, 22.0]])
    local_before = np.asarray(env._potential(env))[0]
    local_observation = np.asarray(env.observation(env))[0]
    env.state.pos_c = env.state.pos_c.at[1].set(jnp.asarray([30.0, 30.0]))
    env.env_params["objective"] = (
        env.env_params["objective"].at[1].set(jnp.asarray([32.0, 32.0]))
    )
    np.testing.assert_array_equal(env._potential(env)[0], local_before)
    np.testing.assert_array_equal(env.observation(env)[0], local_observation)
    env = env.checkpoint(env, jnp.zeros((2, 2)))
    before = np.asarray(env.env_params["prev_potential"])
    env.state.pos_c = env.state.pos_c.at[0, 0].add(1)
    after = np.asarray(env._potential(env))
    np.testing.assert_allclose(env.reward(env), after - before)
    assert env.reward(env)[0] > 0
    np.testing.assert_array_equal(env.env_params["prev_potential"], before)
    env, _, _ = advance_action(env, jnp.zeros((2, 2)), skip_frames=2)
    np.testing.assert_allclose(
        env.reward(env), env._potential(env) - env.env_params["prev_potential"]
    )


def test_vacancy_observation_reveals_occluded_goal_and_preserves_own_goal():
    env = make_environment("vacancy_s1", N=2, num_objectives=2, seed=1)
    env.state.pos_c = jnp.asarray([[2.0, 2.0], [5.0, 2.0]])
    env.env_params["objective"] = jnp.asarray([[5.0, 2.0], [8.0, 2.0]])
    old_goal_channels = np.asarray(SwarmNavigator.observation(env))[0, 2:14]
    new_goal_channels = np.asarray(env.observation(env))[0, 2:14]
    np.testing.assert_array_equal(old_goal_channels, 0)
    assert np.count_nonzero(new_goal_channels) == 1
    np.testing.assert_allclose(new_goal_channels.max(), (16 - 6) / 16)
    np.testing.assert_array_equal(
        env.observation(env)[:, 2:14], vacant_objective_lidar(env) / 16
    )
    np.testing.assert_array_equal(
        env.observation(env)[:, 14:], SwarmNavigator.observation(env)[:, 14:]
    )
    assert np.asarray(env.observation(env))[1, 2:14].max() == 1
    np.testing.assert_allclose(env._potential(env), SwarmNavigator._potential(env))


@pytest.mark.parametrize(
    "variant", ["baseline_s1", "cosine_s1", "vacancy_s2", "cosine_vacancy_s2"]
)
def test_empty_objectives_and_three_dimensional_shapes_are_valid(variant):
    env = make_environment(variant, N=2, num_objectives=0, seed=3)
    np.testing.assert_array_equal(env.reward(env), 0)
    assert np.isfinite(env.observation(env)).all()
    spatial = make_environment(variant, N=2, num_objectives=1, dim=3, seed=3)
    assert spatial.observation(spatial).shape == (2, spatial.observation_space_size)
    assert np.isfinite(spatial.observation(spatial)).all()
    np.testing.assert_allclose(spatial.reward(spatial), 0, atol=1e-7)
