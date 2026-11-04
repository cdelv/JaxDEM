# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Focused reward-history and runner-isolation checks for follow-up ablations."""

import jax
import jax.numpy as jnp

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from benchmarks.swarm_mean_shift import reward_followups
from benchmarks.swarm_mean_shift.reward_followups import make_followup_environment
from benchmarks.swarm_mean_shift.variants import ExperimentalSwarmNavigator
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils import advance_action


def _controlled_env(variant):
    env = make_followup_environment(
        variant, N=2, num_objectives=1, lidar_range=8, n_lidar_rays=4, seed=3
    )
    env.state.pos_c = jnp.asarray([[0.0, 0.0], [9.0, 9.0]])
    env.env_params["objective"] = jnp.asarray([[4.0, 0.0]])
    return env


def test_delta_reset_and_checkpoint_use_rational_score():
    env = make_followup_environment("rational_delta", N=2, seed=5)
    np.testing.assert_allclose(env.reward(env), 0, atol=1e-8)
    np.testing.assert_allclose(
        env.env_params["prev_potential"], ExperimentalSwarmNavigator.reward(env)
    )
    env = _controlled_env("rational_delta")
    action = jnp.zeros((2, 2))
    before = ExperimentalSwarmNavigator.reward(env)
    env = env.checkpoint(env, action)
    env.state.pos_c = env.state.pos_c.at[0].set(jnp.asarray([1.0, 0.0]))
    after = ExperimentalSwarmNavigator.reward(env)
    np.testing.assert_allclose(env.reward(env), after - before)
    assert env.reward(env)[0] > 0
    np.testing.assert_array_equal(env.env_params["prev_potential"], before)
    env = env.checkpoint(env, action)
    np.testing.assert_allclose(env.reward(env), 0, atol=1e-8)


def test_scaled_state_score_is_exactly_one_hundredth_without_checkpoint_history():
    env = _controlled_env("rational_scaled")
    score = ExperimentalSwarmNavigator.reward(env)
    np.testing.assert_allclose(env.reward(env), 0.01 * score)
    assert env.reward(env)[0] > 0
    prior = env.env_params["prev_potential"]
    checkpoint = env.checkpoint(env, jnp.ones((2, 2)))
    np.testing.assert_array_equal(checkpoint.env_params["prev_potential"], prior)
    np.testing.assert_allclose(checkpoint.reward(checkpoint), 0.01 * score)


@pytest.mark.parametrize("variant", reward_followups.FOLLOWUP_NAMES)
def test_followups_match_baseline_observation_shape_and_support_vmap_reset(variant):
    env = make_followup_environment(variant, N=2, num_objectives=3, seed=2)
    assert env.observation_space_size == 26
    assert env.observation(env).shape == (2, 26)
    structure = jax.tree.structure(env)
    stepped, _, _ = advance_action(env, jnp.zeros((2, 2)), skip_frames=2)
    assert jax.tree.structure(stepped) == structure
    assert np.isfinite(stepped.reward(stepped)).all()
    batch = vectorise_env(env, n=2)
    batch = batch.reset(batch, jax.random.split(jax.random.key(4), 2))
    assert batch.observation(batch).shape == (2, 2, 26)
    assert np.isfinite(batch.reward(batch)).all()
    if variant == "rational_delta":
        np.testing.assert_allclose(batch.reward(batch), 0, atol=1e-8)


@pytest.mark.parametrize("variant", reward_followups.FOLLOWUP_NAMES)
def test_empty_goals_remain_zero(variant):
    env = make_followup_environment(variant, N=2, num_objectives=0, seed=2)
    np.testing.assert_array_equal(env.reward(env), 0)
    assert np.isfinite(env.observation(env)).all()


def test_runner_wrapper_changes_only_factory_choices_hashes_and_restores_them():
    from benchmarks.swarm_mean_shift import run

    before = (run.make_environment, run.VARIANT_NAMES, run.source_hashes)
    with reward_followups.matched_followup_runner() as matched:
        assert matched.make_environment is make_followup_environment
        assert matched.VARIANT_NAMES == reward_followups.FOLLOWUP_NAMES
        source = Path(reward_followups.__file__).resolve()
        hashes = matched.source_hashes(Path("/tmp/missing_source_experiment.py"))
        assert hashes[str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (run.make_environment, run.VARIANT_NAMES, run.source_hashes) == before


def test_cli_default_variants_and_argv_are_process_local(monkeypatch):
    from benchmarks.swarm_mean_shift import run

    argv = ["reward_followups", "--epochs", "120"]
    monkeypatch.setattr(sys, "argv", argv)
    captured = {}

    def fake_main():
        captured["argv"] = list(sys.argv)
        captured["factory"] = run.make_environment

    monkeypatch.setattr(run, "main", fake_main)
    reward_followups.main()
    assert captured["argv"] == [
        "reward_followups",
        "--variants",
        "rational_delta",
        "rational_scaled",
        "--epochs",
        "120",
    ]
    assert captured["factory"] is make_followup_environment
    assert sys.argv is argv
