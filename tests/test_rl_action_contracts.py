"""PPO action collection and latent ratios using a synthetic training fixture."""
from dataclasses import dataclass, replace

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxdem.rl.action_spaces import BoxSpace, MaxNormSpace, Transformed
from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.state import State
from jaxdem.system import System
from jaxdem.rl.models import SharedActorCritic, MinGRUActorCritic, Model
from jaxdem.rl.trainers import Trainer
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from tests.test_ppo_alignment import _loss_args


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


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ActionLookupEnv(CheckpointCounter):
    """Two slots, one inactive; discrete actions must really be valid indices."""
    @classmethod
    def Create(cls):
        env = CheckpointCounter.Create()
        state = env.state.add(env.state, pos=jnp.ones((1, 2)), rad=jnp.ones(1))
        return cls(state, env.system, {**env.env_params,
            'limit': jnp.asarray(2), 'total': jnp.asarray(0.0),
            'inactive_action': jnp.asarray(0.0)})

    @staticmethod
    def agent_mask(env):
        return jnp.array([True, False])

    @staticmethod
    def observation(env):
        return jnp.broadcast_to(env.env_params['count'].astype(float), (2, 1))

    @staticmethod
    def reward(env):
        return jnp.array([env.env_params['total'], 0.0])

    @staticmethod
    def step(env, action):
        if jnp.issubdtype(action.dtype, jnp.integer):
            amount = jnp.array([0., 1., 4.])[action[0]]
            inactive = action[1]
        else:
            amount, inactive = action[0, 0], action[1, 0]
        return replace(env, env_params={**env.env_params,
            'count': env.env_params['count']+1,
            'total': env.env_params['total']+amount,
            'inactive_action': inactive.astype(float)})


@pytest.mark.parametrize('discrete', [False, True])
@pytest.mark.parametrize('model_type', [SharedActorCritic, MinGRUActorCritic])
def test_tiny_ppo_epoch_accepts_both_action_types_with_padding(discrete, model_type):
    kwargs = {}
    if model_type is MinGRUActorCritic:
        kwargs.update(hidden_features=4, gru_features=4, num_layers=2)
    model = model_type(observation_space_size=1, action_space_size=3 if discrete else 1,
                       key=nnx.Rngs(8), discrete=discrete, **kwargs)
    tr = PPOTrainer.Create(ActionLookupEnv.Create(), model, num_envs=2,
        num_steps_epoch=2, num_minibatches=2, skip_frames=1)
    tr, td, metrics = tr.epoch(tr, jnp.asarray(0))
    assert jnp.issubdtype(td.action.dtype, jnp.integer) == discrete
    assert td.action.shape == ((2, 4) if discrete else (2, 4, 1))
    assert jnp.all(td.action[:, 1::2] == 0)
    assert jnp.all(tr.env.env_params['inactive_action'] == 0)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(metrics))
    assert (td.latent_action is None) == discrete


class SaturatedPolicy(Model):
    def __init__(self, space, dtype):
        self.mean = nnx.Param(jnp.asarray([30.0], dtype=dtype))
        self.bij = nnx.data(space if isinstance(space, MaxNormSpace) else distrax.Block(space, 1))

    def __call__(self, obs, sequence=False, **kwargs):
        shape = (*obs.shape[:-1], 1)
        return Transformed(distrax.MultivariateNormalDiag(
            jnp.broadcast_to(self.mean[...], shape),
            jnp.full(shape, .1, dtype=self.mean[...].dtype)), self.bij), jnp.zeros(shape)


@pytest.mark.parametrize('space', [BoxSpace(-1., 1.), MaxNormSpace()])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_saturated_actions_use_exact_latent_ratios_and_gradients(space, dtype):
    env = vectorise_env(CheckpointCounter.Create(), n=1)
    model = SaturatedPolicy(space, dtype)
    graphdef, graphstate = nnx.split((model,))
    _, _, _, td = Trainer.trajectory_rollout(env, graphdef, graphstate,
        jax.random.key(7), num_steps_epoch=2)
    td = jax.tree.map(lambda x: x[:, 0], td)
    args = _loss_args()
    (_, aux), grads = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(model, td, **args)
    np.testing.assert_allclose(aux['ratio'], 1., rtol=2e-6, atol=2e-6)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
    # A changed policy must use the original latent, not just force ratios to one.
    model.mean[...] += .02
    (_, aux), grads = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(model, td, **args)
    expected_ratio = jnp.exp(distrax.MultivariateNormalDiag(
        model.mean[...], jnp.array([.1], dtype=dtype)).log_prob(td.latent_action) - td.latent_log_prob)
    np.testing.assert_allclose(aux['ratio'], expected_ratio, rtol=2e-5)
    expected_grad = -jnp.mean(aux['advantages'] * expected_ratio *
                             (td.latent_action[..., 0] - model.mean[0]) / .1**2)
    np.testing.assert_allclose(grads.mean[0], expected_grad, rtol=2e-4, atol=2e-4)
