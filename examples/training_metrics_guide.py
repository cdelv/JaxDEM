# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Log a custom training metric
===============================

This example trains a MinGRU policy in MultiNavigator and adds an accuracy
metric to PPO reports. It then measures accuracy on fresh episodes.
See :doc:`intro_to_rl` for the environment, model, and trainer APIs.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

import jaxdem as jdem
import jaxdem.rl as rl

# %%
# Create the environment and policy
# ---------------------------------
seed = 6
horizon = 160
phi = 0.62
box = jnp.sqrt(64 * jnp.pi / phi)
skip_frames = 49
num_envs = 64
num_steps_epoch = 64
num_minibatches = 8
learning_rate = 1e-3
epochs = 64  # Increase epochs to 2000 for a longer training run.

env = rl.Environment.create(
    "MultiNavigator",
    N=64,
    min_box_size=box,
    max_box_size=box,
    friction=2.0,
)
env.env_params["max_steps"] = jnp.asarray(
    round(horizon / float(env.system.dt)), dtype=int
)
model_key, trainer_key = jax.random.split(jax.random.key(seed))
policy = rl.Model.create(
    "MinGRUActorCritic",
    key=nnx.Rngs(model_key),
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
    action_space=rl.ActionSpace.create("maxNorm", max_norm=2.0),
    actor_sigma_head=False,
    hidden_features=64,
    gru_features=64,
    num_layers=4,
    activation=nnx.identity,
)

# %%
# Create the PPO trainer
# -----------------------
trainer = rl.Trainer.create(
    "PPO",
    env=env,
    model=policy,
    key=trainer_key,
    num_envs=num_envs,
    num_steps_epoch=num_steps_epoch,
    skip_frames=skip_frames,
    num_minibatches=num_minibatches,
    learning_rate=learning_rate,
    anneal_learning_rate=False,
    num_epochs=epochs,
    stop_at_epoch=epochs,
    advantage_gamma=0.99,
    advantage_lambda=0.95,
    ppo_entropy_coeff=0.001,
    ppo_clip_eps=0.2,
    ppo_value_coeff=2.0,
    max_grad_norm=1.5,
    caps_temporal_coeff=1e-4,
    caps_spatial_coeff=1e-3,
    caps_noise_std=0.01,
    vtrace=False,
)


# %%
# Define the metric and train
# ---------------------------
# Accuracy is the fraction of agents within one radius of their target.
# The metric function reduces the environment batch to one JAX scalar.
# It measures the current training state, which can include automatic resets.
def measure_accuracy(env, cutoff_radii=1.0):
    distance = jnp.linalg.norm(env.state.pos_c - env.env_params["objective"], axis=-1)
    return (distance <= cutoff_radii * env.state.rad).mean(axis=-1)


def training_metrics(current, trajectory):
    return {"accuracy": jax.vmap(measure_accuracy)(current.env).mean()}


# The progress mapping selects fields and their Python number formats.
log_directory = Path(tempfile.mkdtemp(prefix="jaxdem-rl-metrics-"))
trainer = trainer.train(
    trainer,
    directory=log_directory,
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
# Evaluate on fresh episodes
# --------------------------
# The evaluation resets each environment and the model's recurrent state.
# It measures accuracy at episode endpoints, without automatic resets.
@jax.jit
def policy_model(obs, key, graphstate, graphdef):
    model, *rest = nnx.merge(graphdef, graphstate)
    pi, _value = model(obs, sequence=False)
    action = pi.sample(seed=key)
    _, new_graphstate = nnx.split((model, *rest))
    return action, new_graphstate


model, *rest = nnx.merge(trainer.graphdef, trainer.graphstate)
model.reset(shape=(num_envs, trainer.env.max_num_agents, 1), mask=None)
trainer.graphdef, trainer.graphstate = nnx.split((model, *rest))
trainer.key, subkey = jax.random.split(trainer.key)
subkeys = jax.random.split(subkey, num_envs)
trainer.env = trainer.env.reset(trainer.env, subkeys)

physics_steps_per_action = 1 + skip_frames
episode_actions = (
    int(env.env_params["max_steps"]) + physics_steps_per_action - 1
) // physics_steps_per_action
trainer.env, trainer.key, trainer.graphstate = jdem.utils.env_step(
    trainer.env,
    policy_model,
    trainer.key,
    trainer.graphstate,
    graphdef=trainer.graphdef,
    n=episode_actions,
    skip_frames=skip_frames,
)

# Report the mean and standard deviation across environments.
accuracy = measure_accuracy(trainer.env)
print(f"Accuracy: {accuracy.mean():.1%} +/- {accuracy.std():.1%}")
