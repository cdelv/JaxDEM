# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Raw sensing must retain occupied goals without changing reward or physics."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from benchmarks.swarm_mean_shift.conservative_variants import (
    make_environment as make_conservative,
)
from benchmarks.swarm_mean_shift.unmasked_variants import (
    MATCHED_COSINE_COEFF,
    TypedUnmaskedSwarmNavigator,
    VARIANT_NAMES,
    make_environment,
    signed_peer_wall_lidar,
)
from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils import advance_action


@pytest.mark.parametrize("scale", [1, 2])
def test_masked_control_is_exactly_the_existing_baseline(scale):
    kwargs = dict(N=4, num_objectives=4, n_lidar_rays=4, seed=12)
    old = make_conservative(f"baseline_s{scale}", **kwargs)
    env = make_environment(f"masked_s{scale}", **kwargs)
    for first, second in (
        (env.state.pos, old.state.pos),
        (env.env_params["objective"], old.env_params["objective"]),
        (env.observation(env), old.observation(old)),
        (env.reward(env), old.reward(old)),
    ):
        np.testing.assert_array_equal(first, second)
    action = jnp.full((4, 2), 0.2)
    env, _, _ = advance_action(env, action, skip_frames=2)
    old, _, _ = advance_action(old, action, skip_frames=2)
    np.testing.assert_array_equal(env.state.pos, old.state.pos)
    np.testing.assert_array_equal(env.reward(env), old.reward(old))
    np.testing.assert_array_equal(env.observation(env), old.observation(old))


@pytest.mark.parametrize("scale", [1, 2])
def test_raw_reward_and_physics_are_identical_to_masked_control(scale):
    kwargs = dict(N=4, num_objectives=4, n_lidar_rays=4, seed=7)
    control = make_environment(f"masked_s{scale}", **kwargs)
    raw = make_environment(f"raw_s{scale}", **kwargs)
    action = jnp.full((4, 2), 0.1)
    np.testing.assert_array_equal(raw._potential(raw), control._potential(control))
    raw, _, _ = advance_action(raw, action, skip_frames=2)
    control, _, _ = advance_action(control, action, skip_frames=2)
    np.testing.assert_array_equal(raw.state.pos, control.state.pos)
    np.testing.assert_array_equal(raw.state.vel, control.state.vel)
    np.testing.assert_array_equal(raw.reward(raw), control.reward(control))


@pytest.mark.parametrize("family", ["raw", "raw_cosine_matched"])
def test_raw_channels_keep_near_occupied_and_own_goals(family):
    env = make_environment(f"{family}_s1", N=2, num_objectives=2, seed=1)
    env.state.pos_c = jnp.asarray([[2.0, 2.0], [5.0, 2.0]])
    env.env_params["objective"] = jnp.asarray([[5.0, 2.0], [8.0, 2.0]])
    peers, goals, _ = SwarmNavigator._sense(env)
    expected = jnp.concatenate([env.state.vel, goals / 16, peers / 16], axis=-1)
    observed = env.observation(env)
    np.testing.assert_array_equal(observed, expected)
    # Observer 0 still sees the near goal occupied by observer 1; unlike a
    # vacancy sensor it does not reveal the farther goal in the same bin.
    np.testing.assert_allclose(np.max(observed[0, 2:14]), 13 / 16)
    assert np.count_nonzero(observed[0, 2:14]) == 1
    assert np.max(observed[1, 2:14]) == 1  # Observer 1 keeps its own goal.
    env.env_params["hide_occupied_objectives"] = jnp.asarray(True)
    np.testing.assert_array_equal(env.observation(env)[0, 2:14], 0)


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_scaled_geometry_is_once_only_and_checkpoints_initialize_to_zero(variant):
    scale = int(variant[-1])
    env = make_environment(variant, N=4, num_objectives=4, seed=5)
    original = make_conservative("baseline_s1", N=4, num_objectives=4, seed=5)
    np.testing.assert_array_equal(env.state.pos, original.state.pos * scale)
    np.testing.assert_allclose(
        env.env_params["objective"],
        original.env_params["objective"] * scale,
        rtol=2e-7,
        atol=1e-6,
    )
    positions, goals = np.asarray(env.state.pos), np.asarray(
        env.env_params["objective"]
    )
    for _ in range(2):
        env = env.reset(env, jax.random.key(5))
        np.testing.assert_array_equal(env.state.pos, positions)
        np.testing.assert_array_equal(env.env_params["objective"], goals)
        np.testing.assert_array_equal(env.reward(env), 0)
        np.testing.assert_array_equal(
            env.system.domain.box_size, [30 * scale, 30 * scale]
        )
    assert env.observation_space_size == 26
    assert env.vacancy_aware_lidar is False
    assert bool(env.env_params["hide_occupied_objectives"]) == variant.startswith(
        "masked_"
    )


def test_matched_cosine_amplitude_and_reward_history():
    env = make_environment("raw_cosine_matched_s1", N=1, num_objectives=1, seed=3)
    env.state.pos_c = jnp.asarray([[4.0, 4.0]])
    env.env_params["objective"] = jnp.asarray([[4.0, 4.0]])
    legacy = make_environment("raw_s1", N=1, num_objectives=1, seed=3)
    legacy.state.pos_c = env.state.pos_c
    legacy.env_params["objective"] = env.env_params["objective"]
    # The default legacy approach at d=0 is .1*(.01/lambda+2*.02/lambda^3).
    baseline_peak = 0.1 * (0.01 / (1 / 3) + 2 * 0.02 / (1 / 3) ** 3)
    assert MATCHED_COSINE_COEFF == pytest.approx(baseline_peak)
    np.testing.assert_allclose(env._potential(env), [1 + baseline_peak], rtol=2e-7)
    np.testing.assert_allclose(
        env._potential(env), legacy._potential(legacy), rtol=2e-7
    )
    env = env.checkpoint(env, jnp.zeros((1, 2)))
    saved = np.asarray(env.env_params["prev_potential"])
    env.state.pos_c = env.state.pos_c.at[0, 0].add(0.5)
    np.testing.assert_array_equal(env.reward(env), env._potential(env) - saved)
    np.testing.assert_array_equal(env.env_params["prev_potential"], saved)


@pytest.mark.parametrize("variant", ["raw_s1", "raw_cosine_matched_s2", "raw_typed_s1"])
def test_nonlocal_goals_and_peers_cannot_change_raw_observation_or_reward(variant):
    env = make_environment(variant, N=2, num_objectives=2, seed=1)
    env.state.pos_c = jnp.asarray([[2.0, 2.0], [24.0, 24.0]])
    env.env_params["objective"] = jnp.asarray([[6.0, 2.0], [22.0, 22.0]])
    before_observation = np.asarray(env.observation(env))[0]
    before_potential = np.asarray(env._potential(env))[0]
    env.state.pos_c = env.state.pos_c.at[1].set(jnp.asarray([27.0, 27.0]))
    env.env_params["objective"] = (
        env.env_params["objective"].at[1].set(jnp.asarray([28.0, 28.0]))
    )
    np.testing.assert_array_equal(env.observation(env)[0], before_observation)
    np.testing.assert_array_equal(env._potential(env)[0], before_potential)
    # A genuinely visible target moving closer changes its raw return.
    env.env_params["objective"] = env.env_params["objective"].at[0, 0].add(-1)
    assert not np.array_equal(env.observation(env)[0], before_observation)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("goals", [0, 2])
@pytest.mark.parametrize("family", ["raw_cosine_matched", "raw_typed"])
def test_raw_empty_goals_and_vectorized_spatial_shapes(dim, goals, family):
    env = make_environment(
        f"{family}_s2",
        N=2,
        num_objectives=goals,
        dim=dim,
        n_lidar_rays=4,
        n_lidar_elevation=2,
        seed=10,
    )
    np.testing.assert_array_equal(env.reward(env), 0)
    peers, raw_goals, ids = SwarmNavigator._sense(env)
    if family == "raw_typed":
        peers = signed_peer_wall_lidar(peers, ids, env)
    expected = jnp.concatenate([env.state.vel, raw_goals / 16, peers / 16], axis=-1)
    np.testing.assert_array_equal(env.observation(env), expected)
    if goals == 0:
        np.testing.assert_array_equal(env._potential(env), 0)
    batch = vectorise_env(env, n=2)
    batch = batch.reset(batch, jax.random.split(jax.random.key(4), 2))
    assert batch.observation(batch).shape == (2, 2, env.observation_space_size)
    np.testing.assert_array_equal(batch.reward(batch), 0)


def test_signed_return_types_and_invalid_empty_self_returns():
    env = make_environment("raw_typed_s1", N=2, num_objectives=2, seed=4)
    lidar = jnp.asarray([[4.0, 3.0, 2.0, 0.0, 1.0], [4.0, 3.0, 2.0, 0.0, 1.0]])
    ids = jnp.asarray([[-1, 1, 0, -1, 5], [-2, 0, 1, -1, 7]])
    signed = signed_peer_wall_lidar(lidar, ids, env)
    np.testing.assert_array_equal(
        signed, [[-4.0, 3.0, 0.0, 0.0, 0.0], [-4.0, 3.0, 0.0, 0.0, 0.0]]
    )


@pytest.mark.parametrize("scale", [1, 2])
def test_typed_returns_preserve_raw_goals_and_baseline_reward_physics(scale):
    kwargs = dict(N=2, num_objectives=2, seed=1)
    typed = make_environment(f"raw_typed_s{scale}", **kwargs)
    raw = make_environment(f"raw_s{scale}", **kwargs)
    assert isinstance(typed, TypedUnmaskedSwarmNavigator)
    assert typed.observation_mode == "baseline"
    for env in (typed, raw):
        env.state.pos_c = jnp.asarray([[2.0, 2.0], [5.0, 2.0]])
        env.env_params["objective"] = jnp.asarray([[5.0, 2.0], [8.0, 2.0]])
    peers, goals, ids = SwarmNavigator._sense(typed)
    observed = typed.observation(typed)
    np.testing.assert_array_equal(observed[:, :14], raw.observation(raw)[:, :14])
    np.testing.assert_array_equal(observed[:, 2:14], goals / 16)
    np.testing.assert_array_equal(jnp.abs(observed[:, 14:]), peers / 16)
    np.testing.assert_array_equal(observed[:, 14:] < 0, (ids < 0) & (peers > 0))
    np.testing.assert_allclose(np.max(observed[0, 2:14]), 13 / 16)
    assert np.max(observed[1, 2:14]) == 1
    np.testing.assert_array_equal(typed._potential(typed), raw._potential(raw))
    action = jnp.full((2, 2), 0.1)
    typed, _, _ = advance_action(typed, action, skip_frames=2)
    raw, _, _ = advance_action(raw, action, skip_frames=2)
    np.testing.assert_array_equal(typed.state.pos, raw.state.pos)
    np.testing.assert_array_equal(typed.state.vel, raw.state.vel)
    np.testing.assert_array_equal(typed.reward(typed), raw.reward(raw))


@pytest.mark.parametrize("scale", [1, 2])
def test_low_entropy_is_exactly_raw_environment(scale):
    kwargs = dict(N=2, num_objectives=2, n_lidar_rays=4, seed=2)
    raw = make_environment(f"raw_s{scale}", **kwargs)
    low = make_environment(f"raw_low_entropy_s{scale}", **kwargs)
    assert type(low) is type(raw)
    np.testing.assert_array_equal(low.state.pos, raw.state.pos)
    np.testing.assert_array_equal(low.observation(low), raw.observation(raw))
    np.testing.assert_array_equal(low._potential(low), raw._potential(raw))
    action = jnp.full((2, 2), 0.1)
    raw, _, _ = advance_action(raw, action, skip_frames=2)
    low, _, _ = advance_action(low, action, skip_frames=2)
    np.testing.assert_array_equal(low.state.pos, raw.state.pos)
    np.testing.assert_array_equal(low.reward(low), raw.reward(raw))


@pytest.mark.parametrize(
    "variant,kwargs,parameter",
    [
        ("raw_s1", {"hide_occupied_objectives": True}, "hide_occupied_objectives"),
        (
            "raw_typed_s1",
            {"hide_occupied_objectives": True},
            "hide_occupied_objectives",
        ),
        (
            "raw_low_entropy_s2",
            {"hide_occupied_objectives": True},
            "hide_occupied_objectives",
        ),
        (
            "raw_cosine_matched_s2",
            {"hide_occupied_objectives": True},
            "hide_occupied_objectives",
        ),
        ("masked_s1", {"hide_occupied_objectives": False}, "hide_occupied_objectives"),
        ("raw_s1", {"vacancy_aware_lidar": True}, "vacancy_aware_lidar"),
        ("raw_s1", {"approach_kernel": "cosine"}, "approach_kernel"),
        ("raw_cosine_matched_s1", {"attraction_coeff": 0.1}, "attraction_coeff"),
        ("raw_s1", {"spacing_scale": 2}, "spacing_scale"),
    ],
)
def test_conflicting_overrides_cannot_silently_enable_masking(
    variant, kwargs, parameter
):
    with pytest.raises(ValueError, match=parameter):
        make_environment(variant, **kwargs)
