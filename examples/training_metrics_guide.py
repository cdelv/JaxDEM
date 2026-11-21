# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Log a custom training metric
===============================

Train a recurrent :py:class:`~jaxdem.rl.models.mingru.MinGRUActorCritic`
policy in :py:class:`~jaxdem.rl.environments.multi_navigator.MultiNavigator`,
attach a custom ``metrics_fn`` to
:py:meth:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer.train`, and evaluate on
fresh episodes. See :doc:`intro_to_rl` for core RL concepts.
"""

import tempfile

import jax
import jax.numpy as jnp
from flax import nnx

import jaxdem as jdem
import jaxdem.rl as rl

# %%
# Create the Environment, Recurrent Policy, and Trainer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Keep sizes compact for a quick local run; scale ``N``, ``num_envs``, and
# ``epochs`` up for full multi-agent training:

seed = 6
n_agents = 8
phi = 0.45
box = jnp.sqrt(n_agents * jnp.pi / phi)
skip_frames = 19
num_envs = 8
num_steps_epoch = 16
epochs = 8

env = rl.Environment.create(
    "MultiNavigator",
    N=n_agents,
    min_box_size=box,
    max_box_size=box,
    max_steps=800,
    friction=2.0,
)
model_key, trainer_key = jax.random.split(jax.random.key(seed))
policy = rl.Model.create(
    "MinGRUActorCritic",
    key=nnx.Rngs(model_key),
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
    action_space=rl.ActionSpace.create("maxNorm", max_norm=2.0),
    actor_sigma_head=False,
    hidden_features=32,
    gru_features=32,
    num_layers=2,
    activation=nnx.identity,
)
trainer = rl.Trainer.create(
    "PPO",
    env=env,
    model=policy,
    key=trainer_key,
    num_envs=num_envs,
    num_steps_epoch=num_steps_epoch,
    skip_frames=skip_frames,
    num_minibatches=4,
    learning_rate=1e-3,
    num_epochs=epochs,
)

# %%
# Define ``metrics_fn`` and Train
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``metrics_fn(trainer, trajectory)`` runs **inside JIT** at the end of every
# epoch and must return a ``dict[str, jax.Array]`` of scalar values. Its keys
# are merged with built-in PPO metrics (such as ``loss``, ``score``,
# ``entropy``, ``steps_per_sec``), written to TensorBoard under ``directory``
# when ``log=True``, and formatted in the progress bar via ``progress_metrics``:


def measure_accuracy(current_env, cutoff_radii: float = 1.0) -> jax.Array:
    """Fraction of agents within ``cutoff_radii`` radii of their target."""
    dist = jnp.linalg.norm(
        current_env.state.pos_c - current_env.env_params["objective"], axis=-1
    )
    return (dist <= cutoff_radii * current_env.state.rad).mean(axis=-1)


def training_metrics(current_trainer, trajectory) -> dict[str, jax.Array]:
    del trajectory
    return {"accuracy": jax.vmap(measure_accuracy)(current_trainer.env).mean()}


with tempfile.TemporaryDirectory(prefix="jaxdem-rl-metrics-") as log_dir:
    trainer = trainer.train(
        trainer,
        directory=log_dir,
        metrics_fn=training_metrics,
        progress_metrics={
            "accuracy": ".1%",
            "score": ".3f",
            "steps_per_sec": ".2e",
        },
        verbose=True,
        log=True,
    )

# %%
# Evaluate the Recurrent Policy on Fresh Episodes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Recurrent models carry hidden state that must be reset to match the
# evaluation batch shape ``(num_envs, max_num_agents, 1)`` before rolling out:


@jax.jit
def policy_model(obs, key, graphstate, graphdef):
    m = nnx.merge(graphdef, graphstate)
    pi, _val = m(obs, sequence=False)
    action = m.policy_output(pi)
    _, new_graphstate = nnx.split(m)
    return action, new_graphstate


eval_model = trainer.model
eval_model.eval()
eval_model.reset(shape=(num_envs, trainer.env.max_num_agents, 1), mask=None)
graphdef, graphstate = nnx.split(eval_model)

eval_key, reset_key = jax.random.split(trainer.key)
eval_env = trainer.env.reset(trainer.env, jax.random.split(reset_key, num_envs))

steps_per_action = 1 + skip_frames
episode_actions = (int(env.env_params["max_steps"]) + steps_per_action - 1) // steps_per_action
eval_env, eval_key, graphstate = jdem.utils.env_step(
    eval_env,
    policy_model,
    eval_key,
    graphstate,
    graphdef=graphdef,
    n=episode_actions,
    skip_frames=skip_frames,
)

accuracy = jax.vmap(measure_accuracy)(eval_env)
print(f"Evaluation accuracy: {float(accuracy.mean()):.1%} +/- {float(accuracy.std()):.1%}")
