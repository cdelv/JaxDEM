# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Checks for exact layout scaling and unchanged local reward/dynamics."""

import jax
import jax.numpy as jnp

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from benchmarks.swarm_mean_shift import spacing_followup
from benchmarks.swarm_mean_shift.spacing_followup import make_environment
from benchmarks.swarm_mean_shift.variants import scan_goal_scores
from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils import advance_action


@pytest.mark.parametrize("variant", spacing_followup.FOLLOWUP_NAMES)
def test_reset_doubles_exact_original_n64_layout_and_domain(variant):
    key = jax.random.key(17)
    original = SwarmNavigator.Create(N=64, num_objectives=64)
    original = original.reset(original, key)
    env = make_environment(variant, seed=17)
    np.testing.assert_array_equal(env.state.pos, 2 * original.state.pos)
    # Nested XLA fusion can change sample rounding by one float32 ULP.
    np.testing.assert_allclose(
        env.env_params["objective"],
        2 * original.env_params["objective"],
        rtol=2e-7,
        atol=1e-6,
    )
    before = np.asarray(original.env_params["objective"])
    after = np.asarray(env.env_params["objective"])
    before_dist = np.linalg.norm(before[:, None] - before[None, :], axis=-1)
    after_dist = np.linalg.norm(after[:, None] - after[None, :], axis=-1)
    np.testing.assert_allclose(after_dist, 2 * before_dist, rtol=2e-7, atol=4e-6)
    np.testing.assert_array_equal(
        env.system.domain.box_size, 2 * original.system.domain.box_size
    )
    np.testing.assert_array_equal(
        env.system.domain.anchor, 2 * original.system.domain.anchor
    )
    np.testing.assert_array_equal(
        env.system.domain.inv_box_size, 1 / env.system.domain.box_size
    )
    assert float(env.env_params["box_size"]) == 40
    assert float(env.env_params["box_padding"]) == 20
    np.testing.assert_array_equal(env.state.rad, original.state.rad)
    np.testing.assert_array_equal(env.state.mass, original.state.mass)
    np.testing.assert_array_equal(env.system.dt, original.system.dt)
    np.testing.assert_array_equal(env.system.time, original.system.time)
    np.testing.assert_array_equal(env.system.step_count, original.system.step_count)
    for name in ("friction", "lidar_range", "sharing_range", "attraction_decay"):
        np.testing.assert_array_equal(env.env_params[name], original.env_params[name])
    assert float(env.env_params["lidar_range"]) == 16
    assert float(env.env_params["max_steps"]) == 200000
    assert env.observation_space_size == 26
    assert env.observation(env).shape == (64, 26)
    np.testing.assert_array_equal(env.observation(env), SwarmNavigator.observation(env))
    np.testing.assert_allclose(env.reward(env), 0, atol=1e-7)


@pytest.mark.parametrize("variant", spacing_followup.FOLLOWUP_NAMES)
def test_repeated_reset_does_not_scale_previously_scaled_geometry(variant):
    env = make_environment(variant, N=4, num_objectives=4, seed=9)
    original_pos = np.asarray(env.state.pos)
    original_goals = np.asarray(env.env_params["objective"])
    for _ in range(3):
        env = env.reset(env, jax.random.key(9))
        np.testing.assert_array_equal(env.state.pos, original_pos)
        np.testing.assert_array_equal(env.env_params["objective"], original_goals)
        np.testing.assert_array_equal(env.system.domain.box_size, [60, 60])
        np.testing.assert_allclose(env.reward(env), 0, atol=1e-8)
    batch = vectorise_env(env, n=2)
    keys = jax.random.split(jax.random.key(20), 2)
    batch = batch.reset(batch, keys)
    batch = batch.reset(batch, keys)
    assert batch.observation(batch).shape == (2, 4, 26)
    np.testing.assert_array_equal(batch.system.domain.box_size, [[60, 60], [60, 60]])
    np.testing.assert_allclose(batch.reward(batch), 0, atol=1e-8)


def _controlled_env(variant):
    env = make_environment(variant, N=2, num_objectives=2, seed=4)
    env.state.pos_c = jnp.asarray([[3.0, 3.0], [20.0, 20.0]])
    env.env_params["objective"] = jnp.asarray([[7.0, 3.0], [25.0, 25.0]])
    return env


def test_spacing_baseline_matches_original_raw_reward_after_scaling_and_action():
    env = _controlled_env("spacing_baseline")
    action = jnp.asarray([[1.0, 0.0], [0.0, -1.0]])
    env = env.checkpoint(env, action)
    np.testing.assert_array_equal(
        env.env_params["prev_potential"], SwarmNavigator._potential(env)
    )
    env.state.pos_c = env.state.pos_c.at[0, 0].add(1)
    np.testing.assert_array_equal(env.reward(env), SwarmNavigator.reward(env))
    stepped, _, _ = advance_action(env, action, skip_frames=2)
    np.testing.assert_array_equal(
        stepped.reward(stepped), SwarmNavigator.reward(stepped)
    )
    assert float(stepped.system.dt) == float(env.system.dt)


def test_rational_delta_checkpoints_the_unscaled_1p5_radius_kernel():
    env = _controlled_env("spacing_rational_delta")
    # Agent 0 has no peer within sensor range and its sole visible target is
    # four radii away. The kernel scale remains 1.5 radii after geometry scaling.
    env = env.checkpoint(env, jnp.zeros((2, 2)))
    expected_before = 1 / (1 + (4 / 1.5) ** 2)
    np.testing.assert_allclose(env.env_params["prev_potential"][0], expected_before)
    env.state.pos_c = env.state.pos_c.at[0, 0].add(1)
    expected_after = 1 / (1 + (3 / 1.5) ** 2)
    np.testing.assert_allclose(env.reward(env)[0], expected_after - expected_before)
    peers, objectives, ids = SwarmNavigator._sense(env)
    _, scores, _ = scan_goal_scores(objectives, peers, ids, env)
    np.testing.assert_allclose(
        env.reward(env), scores.max(axis=-1) - env.env_params["prev_potential"]
    )
    np.testing.assert_allclose(env.env_params["prev_potential"][0], expected_before)
    env = env.checkpoint(env, jnp.zeros((2, 2)))
    np.testing.assert_allclose(env.reward(env), 0, atol=1e-8)


@pytest.mark.parametrize("variant", spacing_followup.FOLLOWUP_NAMES)
def test_distant_goal_and_peer_do_not_affect_local_observation_or_score(variant):
    env = _controlled_env(variant)
    # The distant agent/goal remain beyond the observing agent's range.
    before_score = np.asarray(env._score(env))[0]
    before_obs = np.asarray(env.observation(env))[0]
    env.state.pos_c = env.state.pos_c.at[1].set(jnp.asarray([35.0, 35.0]))
    env.env_params["objective"] = (
        env.env_params["objective"].at[1].set(jnp.asarray([38.0, 38.0]))
    )
    np.testing.assert_array_equal(env._score(env)[0], before_score)
    np.testing.assert_array_equal(env.observation(env)[0], before_obs)


@pytest.mark.parametrize("variant", spacing_followup.FOLLOWUP_NAMES)
def test_empty_goals_and_short_physics_rollout_are_finite(variant):
    env = make_environment(variant, N=2, num_objectives=0, seed=2)
    np.testing.assert_array_equal(env.reward(env), 0)
    assert np.isfinite(env.observation(env)).all()
    stepped, _, _ = advance_action(env, jnp.ones((2, 2)), skip_frames=2)
    np.testing.assert_array_equal(stepped.reward(stepped), 0)
    assert np.isfinite(stepped.state.pos).all()


def test_runner_factory_names_hashes_and_evaluation_are_restored():
    from benchmarks.swarm_mean_shift import run

    before = (run.make_environment, run.VARIANT_NAMES, run.source_hashes, run.evaluate)
    with spacing_followup.matched_followup_runner() as matched:
        assert matched.make_environment is make_environment
        assert matched.VARIANT_NAMES == spacing_followup.FOLLOWUP_NAMES
        hashes = matched.source_hashes(Path("/tmp/missing_source_experiment.py"))
        for module in (spacing_followup, spacing_followup.variants):
            source = Path(module.__file__).resolve()
            assert (
                hashes[str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
            )
    assert (
        run.make_environment,
        run.VARIANT_NAMES,
        run.source_hashes,
        run.evaluate,
    ) == before


def test_cli_applies_double_time_defaults_without_overriding_explicit_values(
    monkeypatch,
):
    from benchmarks.swarm_mean_shift import run

    argv = [
        "spacing_followup",
        "--epochs",
        "4",
        "--checkpoint-at-120",
        "--eval-at",
        "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    captured = {}

    def fake_main():
        captured["args"] = run.parse_args()

    monkeypatch.setattr(run, "main", fake_main)
    spacing_followup.main()
    args = captured["args"]
    assert args.variants == list(spacing_followup.FOLLOWUP_NAMES)
    assert args.epochs == 4
    assert args.schedule_epochs == 4320
    assert args.max_physics_steps == 200000
    assert args.eval_actions == 4000
    assert args.eval_at == [4]
    assert sys.argv is argv
