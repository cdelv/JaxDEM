"""Verify the combined ablation reuses both established mechanisms exactly."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.swarm_mean_shift import combined_followup, reward_followups
from benchmarks.swarm_mean_shift.combined_followup import make_followup_environment
from benchmarks.swarm_mean_shift.variants import ExperimentalSwarmNavigator, make_environment
from jaxdem.utils import advance_action


def test_combined_n64_shape_and_initial_rational_checkpoint():
    env = make_followup_environment("rational_delta_vacancy", seed=4)
    assert env.observation_space_size == 26
    assert env.observation(env).shape == (64, 26)
    assert np.isfinite(env.observation(env)).all()
    np.testing.assert_array_equal(env.reward(env), 0)
    np.testing.assert_array_equal(
        env.env_params["prev_potential"], ExperimentalSwarmNavigator.reward(env)
    )


def test_combined_observation_equals_vacancy_lidar_bit_for_bit():
    kwargs = dict(N=3, num_objectives=4, seed=3)
    combined = make_followup_environment("rational_delta_vacancy", **kwargs)
    vacancy = make_environment("vacancy_lidar", **kwargs)
    np.testing.assert_array_equal(
        combined.observation(combined), vacancy.observation(vacancy)
    )


def test_combined_reward_checkpoint_and_physics_equal_rational_delta():
    kwargs = dict(N=2, num_objectives=3, seed=7)
    combined = make_followup_environment("rational_delta_vacancy", **kwargs)
    delta = reward_followups.make_followup_environment("rational_delta", **kwargs)
    action = jnp.asarray([[0.5, -0.25], [0.25, 0.5]])
    combined, _, _ = advance_action(combined, action, skip_frames=2)
    delta, _, _ = advance_action(delta, action, skip_frames=2)
    np.testing.assert_array_equal(combined.state.pos, delta.state.pos)
    np.testing.assert_array_equal(combined.state.vel, delta.state.vel)
    np.testing.assert_array_equal(
        combined.env_params["prev_potential"], delta.env_params["prev_potential"]
    )
    np.testing.assert_array_equal(combined.reward(combined), delta.reward(delta))


def test_combined_observation_and_raw_score_ignore_unseen_geometry():
    env = make_followup_environment(
        "rational_delta_vacancy", N=3, num_objectives=2, lidar_range=10, seed=2
    )
    env.state.pos_c = jnp.asarray([[0.0, 0.0], [4.0, 0.0], [20.0, 0.0]])
    env.env_params["objective"] = jnp.asarray([[5.0, 1.0], [20.0, 20.0]])
    before_observation = env.observation(env)[0]
    before_score = ExperimentalSwarmNavigator.reward(env)[0]
    env.state.pos_c = env.state.pos_c.at[2].set(jnp.asarray([100.0, 100.0]))
    env.env_params["objective"] = env.env_params["objective"].at[1].set(
        jnp.asarray([100.0, -100.0])
    )
    np.testing.assert_array_equal(env.observation(env)[0], before_observation)
    np.testing.assert_array_equal(ExperimentalSwarmNavigator.reward(env)[0], before_score)


def test_combined_runner_wrapper_restores_globals_and_records_dependency_hash():
    from benchmarks.swarm_mean_shift import run

    before = (run.make_environment, run.VARIANT_NAMES, run.source_hashes)
    with combined_followup.matched_followup_runner() as matched:
        assert matched.make_environment is make_followup_environment
        assert matched.VARIANT_NAMES == ("rational_delta_vacancy",)
        hashes = matched.source_hashes(Path("/tmp/missing_source_experiment.py"))
        for module in (combined_followup, reward_followups):
            source = Path(module.__file__).resolve()
            assert hashes[str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (run.make_environment, run.VARIANT_NAMES, run.source_hashes) == before
