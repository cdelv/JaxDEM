"""Check that appended vacancy guidance preserves the baseline exactly."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.swarm_mean_shift import observation_followups
from benchmarks.swarm_mean_shift.observation_followups import make_followup_environment
from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils import advance_action


def test_augmented_default_n64_shape():
    env = make_followup_environment("paper_augmented", seed=2)
    assert env.observation_space_size == 29
    observation = env.observation(env)
    assert observation.shape == (64, 29)
    assert np.isfinite(observation).all()
    np.testing.assert_array_equal(observation[:, :26], SwarmNavigator.observation(env))


def test_reward_dynamics_and_baseline_features_equal_production_after_action():
    kwargs = dict(N=2, num_objectives=3)
    augmented = make_followup_environment("paper_augmented", seed=4, **kwargs)
    baseline = SwarmNavigator.Create(**kwargs)
    baseline = baseline.reset(baseline, jax.random.key(4))
    action = jnp.asarray([[0.5, 0.25], [-0.25, 0.5]])
    augmented, _, _ = advance_action(augmented, action, skip_frames=2)
    baseline, _, _ = advance_action(baseline, action, skip_frames=2)
    np.testing.assert_array_equal(augmented.state.pos, baseline.state.pos)
    np.testing.assert_array_equal(augmented.state.vel, baseline.state.vel)
    np.testing.assert_array_equal(augmented.reward(augmented), baseline.reward(baseline))
    np.testing.assert_array_equal(
        augmented.env_params["prev_potential"], baseline.env_params["prev_potential"]
    )
    np.testing.assert_array_equal(
        augmented.observation(augmented)[:, :26], baseline.observation(baseline)
    )


def test_appended_guidance_ignores_geometry_outside_observers_range():
    env = make_followup_environment(
        "paper_augmented", N=3, num_objectives=2, lidar_range=10, seed=3
    )
    env.state.pos_c = jnp.asarray([[0.0, 0.0], [4.0, 0.0], [20.0, 0.0]])
    env.env_params["objective"] = jnp.asarray([[5.0, 1.0], [20.0, 20.0]])
    before = env.observation(env)[0]
    assert before[-1] > 0
    env.state.pos_c = env.state.pos_c.at[2].set(jnp.asarray([100.0, 100.0]))
    env.env_params["objective"] = env.env_params["objective"].at[1].set(
        jnp.asarray([100.0, -100.0])
    )
    np.testing.assert_array_equal(env.observation(env)[0], before)


@pytest.mark.parametrize("num_objectives", [0, 2])
def test_augmented_environment_supports_batched_resets(num_objectives):
    env = make_followup_environment("paper_augmented", N=2, num_objectives=num_objectives)
    batch = vectorise_env(env, n=2)
    batch = batch.reset(batch, jax.random.split(jax.random.key(6), 2))
    assert batch.observation(batch).shape == (2, 2, 29)
    assert np.isfinite(batch.observation(batch)).all()
    np.testing.assert_allclose(batch.reward(batch), 0, atol=1e-8)
    if num_objectives == 0:
        np.testing.assert_array_equal(batch.observation(batch)[..., -3:], 0)


def test_observation_followup_wrapper_restores_runner_and_hashes_its_source():
    from benchmarks.swarm_mean_shift import run

    before = (run.make_environment, run.VARIANT_NAMES, run.source_hashes)
    with observation_followups.matched_followup_runner() as matched:
        assert matched.VARIANT_NAMES == ("paper_augmented",)
        assert matched.make_environment is make_followup_environment
        source = Path(observation_followups.__file__).resolve()
        hashes = matched.source_hashes(Path("/tmp/missing_source_experiment.py"))
        assert hashes[str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (run.make_environment, run.VARIANT_NAMES, run.source_hashes) == before
