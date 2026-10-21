"""Semantic checks for the experimental reward/sensor changes."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.swarm_mean_shift.variants import (
    compact_scan_features,
    exact_local_features,
    make_environment,
    paper_vacancy_features,
    scan_goal_scores,
    vacant_objective_lidar,
)
from jaxdem.rl.environments import SwarmNavigator
from jaxdem.utils import advance_action


def _sensor_env(pos=None, objectives=None, n_lidar_rays=4):
    pos = jnp.asarray([[0.0, 0.0], [3.0, 0.0]] if pos is None else pos)
    objectives = jnp.asarray([[3.0, 0.0]] if objectives is None else objectives)
    return SimpleNamespace(
        max_num_agents=pos.shape[0],
        num_objectives=objectives.shape[0],
        n_lidar_rays=n_lidar_rays,
        n_lidar_elevation=4,
        state=SimpleNamespace(pos=pos, rad=jnp.ones(pos.shape[0]), dim=2),
        env_params={"lidar_range": jnp.asarray(10.0), "objective": objectives},
    )


@pytest.mark.parametrize(
    "variant,features",
    [
        ("baseline", 26),
        ("simple", 26),
        ("compact", 18),
        ("combined", 18),
        ("exact_compact", 18),
        ("exact_combined", 18),
        ("vacancy_lidar", 26),
        ("vacancy_combined", 26),
        ("paper_compact", 18),
        ("paper_combined", 18),
    ],
)
def test_variants_preserve_jax_tree_and_return_finite_features(variant, features):
    env = make_environment(variant, N=2, num_objectives=3, seed=5)
    assert env.observation_space_size == features
    assert env.observation(env).shape == (2, features)
    assert np.isfinite(env.observation(env)).all()
    assert np.isfinite(env.reward(env)).all()
    structure = jax.tree.structure(env)
    start = env.env_params["prev_potential"]
    action = jnp.zeros((2, 2))
    stepped, _, _ = advance_action(env, action, skip_frames=1)
    assert jax.tree.structure(stepped) == structure
    assert np.isfinite(stepped.reward(stepped)).all()
    if variant in {"baseline", "compact", "exact_compact", "vacancy_lidar", "paper_compact"}:
        np.testing.assert_allclose(env.reward(env), 0, atol=1e-12)
        np.testing.assert_allclose(stepped.env_params["prev_potential"], start)
    else:
        # Direct state reward is positive at reset and needs no history update.
        assert np.any(np.asarray(env.reward(env)) > 0)
        assert np.all(np.asarray(env.reward(env)) >= 0)
    reset = env.reset_if_done(stepped, jnp.asarray(True), jax.random.key(6))
    assert jax.tree.structure(reset) == structure
    assert reset.observation(reset).shape == (2, features)


def test_baseline_factory_is_the_unchanged_production_class():
    env = make_environment("baseline", N=2)
    assert type(env) is SwarmNavigator


def test_rational_reward_increases_on_approach_and_discounts_peer_claims():
    env = _sensor_env()
    empty = jnp.zeros((2, 4))
    ids = jnp.broadcast_to(jnp.arange(2)[:, None], empty.shape)
    far = empty.at[0, 0].set(4.0)  # d=6
    near = empty.at[0, 0].set(7.0)  # d=3
    _, far_score, _ = scan_goal_scores(far, empty, ids, env)
    _, near_score, _ = scan_goal_scores(near, empty, ids, env)
    assert near_score[0, 0] > far_score[0, 0] > 0
    occupied_ids = ids.at[0, 0].set(1)
    _, occupied_score, _ = scan_goal_scores(near, near, occupied_ids, env)
    np.testing.assert_allclose(occupied_score[0, 0], near_score[0, 0] / 2)
    # A wall in the same bin cannot claim the goal.
    _, wall_score, _ = scan_goal_scores(near, near, ids.at[0, 0].set(-1), env)
    np.testing.assert_allclose(wall_score, near_score)


def test_own_goal_and_balanced_vacancy_remain_distinguishable_from_empty():
    env = _sensor_env()
    empty = jnp.zeros((2, 4))
    ids = jnp.broadcast_to(jnp.arange(2)[:, None], empty.shape)
    own_goal = empty.at[0, 2].set(10)
    features = compact_scan_features(own_goal, empty, ids, env)
    np.testing.assert_allclose(features[0, :2], 0, atol=1e-7)
    assert features[0, -2] > 0
    np.testing.assert_allclose(features[0, -1], 1)
    balanced = empty.at[0, 0].set(7).at[0, 2].set(7)
    features = compact_scan_features(balanced, empty, ids, env)
    np.testing.assert_allclose(features[0, :2], 0, atol=1e-7)
    assert features[0, -2] > 0
    np.testing.assert_array_equal(compact_scan_features(empty, empty, ids, env), 0)


def test_exact_compact_features_ignore_all_unseen_goals_and_peers():
    env = _sensor_env(
        pos=[[0.0, 0.0], [4.0, 0.0], [20.0, 0.0]],
        objectives=[[5.0, 1.0], [20.0, 20.0]],
    )
    before = exact_local_features(env)[0]
    env.state.pos = env.state.pos.at[2].set(jnp.asarray([100.0, 100.0]))
    env.env_params["objective"] = env.env_params["objective"].at[1].set(
        jnp.asarray([100.0, -100.0])
    )
    np.testing.assert_array_equal(exact_local_features(env)[0], before)
    assert before[-2] > 0


def test_exact_sensor_reveals_additional_goal_behind_same_bin_return():
    env = _sensor_env(objectives=[[3.0, 0.0], [6.0, 0.0]])
    both = exact_local_features(env)[0]
    env.env_params["objective"] = env.env_params["objective"].at[1].set(
        jnp.asarray([20.0, 0.0])
    )
    front_only = exact_local_features(env)[0]
    assert both[0] > front_only[0]
    assert both[-2] > front_only[-2]
    np.testing.assert_allclose(both[-1], front_only[-1])


def test_vacancy_lidar_filters_before_reduction_and_preserves_own_goal():
    env = _sensor_env(objectives=[[3.0, 0.0], [6.0, 0.0]])
    lidar = vacant_objective_lidar(env)
    # Observer at 0 sees the free goal at 6 behind the peer covering goal 3.
    np.testing.assert_allclose(lidar[0], [0, 0, 4, 0])
    # Observer at 3 retains the goal it covers itself (distance zero).
    np.testing.assert_allclose(lidar[1, 2], 10)
    # Moving the covering peer out of range removes its claim entirely.
    env.state.pos = env.state.pos.at[1].set(jnp.asarray([20.0, 0.0]))
    np.testing.assert_allclose(vacant_objective_lidar(env)[0], [0, 0, 7, 0])


def test_paper_explores_vacancies_while_retaining_own_goal_signal():
    env = _sensor_env(
        pos=[[0.0, 0.0], [3.0, 0.0]],
        objectives=[[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
    )
    features = paper_vacancy_features(env)[0]
    # Neither the observer's own goal nor the peer's goal dilutes drift.
    np.testing.assert_allclose(features[:2], [0.6, 0], atol=1e-7)
    np.testing.assert_allclose(features[-1], 1)
    assert features[-2] > 0
    env.env_params["objective"] = env.env_params["objective"][:2]
    env.num_objectives = 2
    no_vacancies = paper_vacancy_features(env)[0]
    np.testing.assert_array_equal(no_vacancies[:3], 0)
    np.testing.assert_allclose(no_vacancies[-1], 1)


@pytest.mark.parametrize("feature_fn", [paper_vacancy_features, vacant_objective_lidar])
def test_paper_sensor_variants_ignore_unseen_geometry(feature_fn):
    env = _sensor_env(
        pos=[[0.0, 0.0], [4.0, 0.0], [20.0, 0.0]],
        objectives=[[5.0, 1.0], [20.0, 20.0]],
    )
    before = feature_fn(env)[0]
    env.state.pos = env.state.pos.at[2].set(jnp.asarray([100.0, 100.0]))
    env.env_params["objective"] = env.env_params["objective"].at[1].set(
        jnp.asarray([100.0, -100.0])
    )
    np.testing.assert_array_equal(feature_fn(env)[0], before)


@pytest.mark.parametrize(
    "variant", ["simple", "compact", "combined", "exact_combined", "vacancy_combined", "paper_combined"]
)
def test_empty_goals_produce_zero_reward_and_finite_observations(variant):
    env = make_environment(variant, N=2, num_objectives=0, seed=4, n_lidar_rays=4)
    np.testing.assert_array_equal(env.reward(env), 0)
    assert np.isfinite(env.observation(env)).all()
    if variant not in {"simple", "vacancy_combined"}:
        # velocity, mean shift, vacancy mass, own-goal signal are all zero.
        np.testing.assert_array_equal(env.observation(env)[:, :6], 0)


def test_compact_3d_feature_shape():
    env = make_environment(
        "combined", N=2, num_objectives=2, dim=3,
        n_lidar_rays=4, n_lidar_elevation=4, seed=2,
    )
    assert env.observation_space_size == 24  # 3 velocity + 5 compact + 16 LiDAR
    assert env.observation(env).shape == (2, 24)
    assert np.isfinite(env.observation(env)).all()
