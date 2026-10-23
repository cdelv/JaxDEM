# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Opt-in cosine attraction matches the experiment and preserves defaults."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from benchmarks.swarm_mean_shift.conservative_variants import (
    cosine_shared_potential,
    make_environment,
)
from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.environments._swarm import shared_potential
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils import advance_action


def _reset(*, kernel="legacy", dim=2, goals=4, agents=4, **kwargs):
    env = SwarmNavigator.Create(
        N=agents,
        num_objectives=goals,
        dim=dim,
        n_lidar_rays=4,
        n_lidar_elevation=2,
        approach_kernel=kernel,
        **kwargs,
    )
    return env.reset(env, jax.random.key(12))


@jax.jit
def _default_shared_score(env):
    peers, objectives, ids = SwarmNavigator._sense(env)
    return shared_potential(
        objectives,
        peers,
        ids,
        env,
        n_elevation=env.n_lidar_elevation if env.state.dim == 3 else None,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_default_and_explicit_legacy_are_bit_exact(dim):
    default = SwarmNavigator.Create(
        N=4,
        num_objectives=4,
        dim=dim,
        n_lidar_rays=4,
        n_lidar_elevation=2,
    )
    assert default.approach_kernel == "legacy"
    default = default.reset(default, jax.random.key(12))
    explicit = _reset(dim=dim)
    np.testing.assert_array_equal(
        _default_shared_score(default),
        explicit._potential(explicit),
    )
    np.testing.assert_array_equal(
        default.observation(default), explicit.observation(explicit)
    )
    action = jnp.full((4, dim), 0.2)
    default, _, _ = advance_action(default, action, skip_frames=2)
    explicit, _, _ = advance_action(explicit, action, skip_frames=2)
    np.testing.assert_array_equal(default.state.pos, explicit.state.pos)
    np.testing.assert_array_equal(default.reward(default), explicit.reward(explicit))
    np.testing.assert_array_equal(
        default.observation(default), explicit.observation(explicit)
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("scale", [1, 2])
def test_cosine_exactly_matches_experimental_score_and_action_interval(dim, scale):
    reference = make_environment(
        f"cosine_s{scale}",
        N=4,
        num_objectives=4,
        dim=dim,
        n_lidar_rays=4,
        n_lidar_elevation=2,
        seed=15,
    )
    env = _reset(kernel="cosine", dim=dim)
    env.state = jax.tree.map(jnp.array, reference.state)
    env.system = jax.tree.map(jnp.array, reference.system)
    env.env_params = jax.tree.map(jnp.array, reference.env_params)
    peers, objectives, ids = env._sense(env)
    np.testing.assert_array_equal(
        env._potential(env), cosine_shared_potential(objectives, peers, ids, env)
    )
    np.testing.assert_array_equal(env.reward(env), reference.reward(reference))
    np.testing.assert_array_equal(
        env.observation(env), reference.observation(reference)
    )
    action = jnp.full((4, dim), 0.2)
    env, _, _ = advance_action(env, action, skip_frames=2)
    reference, _, _ = advance_action(reference, action, skip_frames=2)
    np.testing.assert_array_equal(env.state.pos, reference.state.pos)
    np.testing.assert_array_equal(env.reward(env), reference.reward(reference))
    np.testing.assert_array_equal(
        env.observation(env), reference.observation(reference)
    )


def test_cosine_does_not_use_the_three_legacy_shape_controls():
    env = _reset(kernel="cosine")
    before = np.asarray(env._potential(env))
    env.env_params["attraction_constant"] = jnp.asarray(4.0)
    env.env_params["attraction_quadratic"] = jnp.asarray(5.0)
    env.env_params["attraction_decay"] = jnp.asarray(2.0)
    np.testing.assert_array_equal(env._potential(env), before)
    # Those same changes still affect the opt-in-independent legacy branch.
    env.approach_kernel = "legacy"
    assert not np.array_equal(env._potential(env), before)


def test_zero_attraction_retains_exact_settling_and_peer_sharing():
    env = _reset(kernel="cosine", agents=2, goals=1, attraction_coeff=0)
    # Both agents make a nonzero shared claim on the same nearby objective.
    env.state.pos_c = jnp.asarray([[2.0, 2.0], [3.0, 2.0]])
    env.env_params["objective"] = jnp.asarray([[2.5, 2.0]])
    cosine = np.asarray(env._potential(env))
    env.approach_kernel = "legacy"
    np.testing.assert_array_equal(env._potential(env), cosine)
    assert np.all(cosine > 0)
    assert np.all(cosine < np.exp(-((0.5 / 1.5) ** 4)))


@pytest.mark.parametrize("kernel", ["", "unknown", None])
def test_invalid_approach_kernel_is_rejected(kernel):
    with pytest.raises(ValueError, match="approach_kernel"):
        SwarmNavigator.Create(approach_kernel=kernel)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("goals", [0, 3])
def test_cosine_empty_goals_and_vectorized_reset(dim, goals):
    env = _reset(kernel="cosine", dim=dim, goals=goals, agents=2)
    np.testing.assert_array_equal(env.reward(env), 0)
    assert np.isfinite(env._potential(env)).all()
    if goals == 0:
        np.testing.assert_array_equal(env._potential(env), 0)
    batch = vectorise_env(env, n=2)
    batch = batch.reset(batch, jax.random.split(jax.random.key(3), 2))
    assert batch.approach_kernel == "cosine"
    assert batch.observation(batch).shape == (2, 2, env.observation_space_size)
    np.testing.assert_array_equal(batch.reward(batch), 0)
