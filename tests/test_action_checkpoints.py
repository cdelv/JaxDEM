"""Policy checkpoints span repeated physics steps and stop at episode boundaries."""

from dataclasses import dataclass, replace

import distrax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import clip_action_env, vectorise_env
from jaxdem.rl.models import Model
from jaxdem.rl.trainers import Trainer
from jaxdem.state import State
from jaxdem.system import System
from jaxdem.utils.environment import advance_action, env_step


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CheckpointCounter(Environment):
    @classmethod
    def Create(cls):
        state = State.create(pos=jnp.zeros((1, 2)))
        return cls(state, System.create(state.shape), {
            'count': jnp.asarray(0), 'limit': jnp.asarray(100),
            'terminal': jnp.asarray(False), 'checkpoints': jnp.asarray(0),
            'previous': jnp.asarray(0.),
        })

    @staticmethod
    def reset(env, key):
        return replace(env, env_params={**env.env_params,
            'count': jnp.asarray(0), 'checkpoints': jnp.asarray(0),
            'previous': jnp.asarray(0.),
        })

    @staticmethod
    def step(env, action):
        return replace(env, env_params={**env.env_params,
                                        'count': env.env_params['count']+1})

    @staticmethod
    def checkpoint(env, action):
        p = env.env_params
        return replace(env, env_params={**p,
            'previous': p['count'].astype(float)**2,
            'checkpoints': p['checkpoints']+1})

    @staticmethod
    def observation(env):
        return env.env_params['count'].astype(float).reshape(1, 1)

    @staticmethod
    def reward(env):
        return (env.env_params['count'].astype(float)**2-env.env_params['previous'])[None]

    @staticmethod
    def terminated(env):
        p = env.env_params
        return (p['count'] >= p['limit']) & p['terminal']

    @staticmethod
    def truncated(env):
        p = env.env_params
        return (p['count'] >= p['limit']) & ~p['terminal']


class EndpointValuePolicy(Model):
    def __call__(self, x, sequence=False):
        return distrax.MultivariateNormalDiag(jnp.zeros_like(x), jnp.ones_like(x)), x


@pytest.mark.parametrize('skip_frames', [0, 1, 50])
def test_checkpoint_runs_once_per_action_and_identical_actions_have_new_baselines(skip_frames):
    env = CheckpointCounter.Create()
    action = jnp.zeros((1, 1))
    env, _, _ = advance_action(env, action, skip_frames=skip_frames)
    n = 1+skip_frames
    assert env.env_params['checkpoints'] == 1
    np.testing.assert_array_equal(env.reward(env), [n*n])
    # Extend the episode so even the 51-frame case includes both full intervals.
    env = replace(env, env_params={**env.env_params, 'limit': jnp.asarray(1000)})
    env, _, _ = advance_action(env, action, skip_frames=skip_frames)
    assert env.env_params['checkpoints'] == 2
    assert env.env_params['previous'] == n*n
    np.testing.assert_array_equal(env.reward(env), [3*n*n])
    np.testing.assert_array_equal(env.observation(env), [[2*n]])


def _mixed_batch():
    env = vectorise_env(CheckpointCounter.Create(), n=3)
    return replace(env, env_params={**env.env_params,
        'limit': jnp.array([2, 4, 100]), 'terminal': jnp.array([False, True, False])})


def test_start_baseline_is_preserved_until_each_actual_terminal_state():
    env = clip_action_env(_mixed_batch())
    env, terminated, truncated = advance_action(env, jnp.ones((3, 1, 1)), skip_frames=4)
    np.testing.assert_array_equal(env.env_params['count'], [2, 4, 5])
    np.testing.assert_array_equal(env.env_params['checkpoints'], [1, 1, 1])
    np.testing.assert_array_equal(env.reward(env)[:, 0], [4, 16, 25])
    np.testing.assert_array_equal(terminated, [False, True, False])
    np.testing.assert_array_equal(truncated, [True, False, False])
    env, _, _ = advance_action(env, jnp.ones((3, 1, 1)), skip_frames=4,
                              terminated=terminated, truncated=truncated)
    np.testing.assert_array_equal(env.env_params['count'], [2, 4, 10])
    np.testing.assert_array_equal(env.reward(env)[:, 0], [0, 0, 75])


def test_trajectory_checkpoint_precedes_bootstrap_and_reset():
    env = _mixed_batch()
    graphdef, graphstate = nnx.split((EndpointValuePolicy(),))
    env, _, _, td = Trainer.trajectory_rollout(
        env, graphdef, graphstate, jax.random.key(5), num_steps_epoch=2, skip_frames=4,
    )
    np.testing.assert_array_equal(td.reward[..., 0], [[4, 16, 25], [4, 16, 75]])
    np.testing.assert_array_equal(td.obs[..., 0, 0], [[0, 0, 0], [0, 0, 5]])
    # The value of the truncated endpoint is 2; its reset observation is 0.
    np.testing.assert_array_equal(td.bootstrap_value[:, 0, 0], [2, 2])
    np.testing.assert_array_equal(env.observation(env)[:, 0, 0], [0, 0, 10])
    np.testing.assert_array_equal(env.env_params['previous'], [0, 0, 25])


def test_visualization_uses_the_same_checkpoint_and_boundary_rules():
    def policy(obs, key, state):
        return jnp.zeros_like(obs), state

    env, _, _ = env_step(_mixed_batch(), policy, jax.random.key(6), (), n=2, skip_frames=4)
    np.testing.assert_array_equal(env.observation(env)[:, 0, 0], [2, 4, 10])
    np.testing.assert_array_equal(env.reward(env)[:, 0], [0, 0, 75])




@pytest.mark.parametrize('skip_frames', [-1, True, 1.5])
def test_invalid_repeat_counts(skip_frames):
    with pytest.raises(ValueError, match='skip_frames'):
        advance_action(CheckpointCounter.Create(), jnp.zeros((1, 1)), skip_frames=skip_frames)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NoHistoryCounter(CheckpointCounter):
    checkpoint = staticmethod(Environment.checkpoint)

    @staticmethod
    def reward(env):
        return env.env_params['count'].astype(float)[None]


def test_environment_without_history_uses_noop_checkpoint():
    env = NoHistoryCounter.Create()
    env, _, _ = advance_action(env, jnp.zeros((1, 1)), skip_frames=4)
    assert env.env_params['checkpoints'] == 0
    np.testing.assert_array_equal(env.reward(env), [5.])
    np.testing.assert_array_equal(env.observation(env), [[5.]])


def test_zero_length_evaluation_does_not_checkpoint():
    def policy(obs, key, state):
        return jnp.zeros_like(obs), state

    env = CheckpointCounter.Create()
    result, _, _ = env_step(env, policy, jax.random.key(0), (), n=0)
    assert result.env_params['checkpoints'] == 0
    assert result.env_params['count'] == 0


def test_start_checkpoint_reads_live_state_instead_of_stale_history():
    env = CheckpointCounter.Create()
    env = replace(env, env_params={**env.env_params,
        'count': jnp.asarray(10), 'previous': jnp.asarray(-100.)})
    env, _, _ = advance_action(env, jnp.zeros((1, 1)), skip_frames=2)
    assert env.env_params['checkpoints'] == 1
    assert env.env_params['previous'] == 100
    np.testing.assert_array_equal(env.reward(env), [69.])


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ActionCounter(CheckpointCounter):
    @classmethod
    def Create(cls):
        env = CheckpointCounter.Create()
        return cls(env.state, env.system, {**env.env_params,
            'checkpoint_action': jnp.zeros((1, 1)),
            'applied_action': jnp.zeros((1, 1))})

    @staticmethod
    def checkpoint(env, action):
        env = CheckpointCounter.checkpoint(env, action)
        return replace(env, env_params={**env.env_params, 'checkpoint_action': action})

    @staticmethod
    def step(env, action):
        env = CheckpointCounter.step(env, action)
        return replace(env, env_params={**env.env_params, 'applied_action': action})


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('clip_first', [False, True])
def test_checkpoint_receives_the_same_clipped_action_as_every_step(batch_size, clip_first):
    env = ActionCounter.Create()
    if clip_first:
        env = vectorise_env(clip_action_env(env, -.2, .7), n=batch_size)
    else:
        env = clip_action_env(vectorise_env(env, n=batch_size), -.2, .7)
    for value, expected in [(1.5, .7), (-3., -.2)]:
        action = jnp.full((batch_size, 1, 1), value)
        env, _, _ = advance_action(env, action, skip_frames=2)
        np.testing.assert_array_equal(env.env_params['checkpoint_action'], expected)
        np.testing.assert_array_equal(env.env_params['applied_action'], expected)
    np.testing.assert_array_equal(env.env_params['checkpoints'], 2)


def test_trainer_passes_its_sampled_action_to_checkpoint():
    env = vectorise_env(ActionCounter.Create(), n=2)
    graphdef, graphstate = nnx.split((EndpointValuePolicy(),))
    final, _, _, td = Trainer.trajectory_rollout(
        env, graphdef, graphstate, jax.random.key(7), num_steps_epoch=1, skip_frames=2,
    )
    np.testing.assert_array_equal(final.env_params['checkpoint_action'], td.action[0])
    np.testing.assert_array_equal(final.env_params['applied_action'], td.action[0])
    np.testing.assert_array_equal(final.env_params['checkpoints'], 1)
