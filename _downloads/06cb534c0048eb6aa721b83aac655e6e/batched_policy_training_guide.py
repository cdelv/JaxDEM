# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Batch train eight policies
=============================

Train eight independent PPO policies in parallel with ``jax.vmap`` (each with
its own parameters, optimizer state, environments, and PRNG key), then compare
them from identical initial conditions.

See :doc:`intro_to_rl` for the navigation task and PPO configuration. Install
``JaxDEM[rl]`` and ``matplotlib``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

import jaxdem.rl as rl
from jaxdem import utils
from jaxdem.rl.trainers import PPOTrainer

# %%
# Configure One Template Trainer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build a single ``template`` trainer outside ``vmap``. Keeping the static
# ``graphdef`` and Optax transform ``tx`` on ``template`` lets ``vmap`` batch
# only the per-policy dynamic leaves (``graphstate``, ``env``, ``key``, and any
# swept scalar hyperparameters such as ``ppo_entropy_coeff``):

num_policies = 8
num_envs = 32
num_epochs = 64
skip_frames = 49
evaluation_steps = 40

env = rl.Environment.create(
    "single_navigator",
    min_box_size=4.0,
    max_box_size=4.0,
    max_steps=2000,
    friction=2.0,
    action_alpha=1.0,
)
action_space = rl.ActionSpace.create("max_norm", max_norm=3.0)


def make_model(key):
    return rl.Model.create(
        "SharedActorCritic",
        key=nnx.Rngs(key),
        architecture=[32, 32],
        observation_space_size=env.observation_space_size,
        action_space_size=env.action_space_size,
        action_space=action_space,
    )


template = rl.Trainer.create(
    "PPO",
    env=env,
    model=make_model(jax.random.key(0)),
    key=jax.random.key(1),
    optimizer=optax.adam,
    learning_rate=3e-3,
    num_envs=num_envs,
    num_steps_epoch=32,
    num_epochs=num_epochs,
    skip_frames=skip_frames,
    num_minibatches=16,
    minibatch_size=256,
    ppo_entropy_coeff=1e-4,
    ppo_value_coeff=0.5,
)
_, template_optimizer = nnx.merge(template.graphdef, template.graphstate)

# %%
# Initialize and Train Eight Policies via ``jax.vmap``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each policy gets independent model weights, optimizer state, reset
# environments, and PRNG keys (observation shape ``(8, 32, 1, 6)``). To sweep a
# dynamic hyperparameter such as entropy regularization across the batch, pass
# an array into ``vmap`` and include ``ppo_entropy_coeff=coeff`` in ``replace``:


def initialize_policy(key):
    model_key, reset_key, training_key = jax.random.split(key, 3)
    model = make_model(model_key)
    optimizer = nnx.Optimizer(model, template_optimizer.tx, wrt=nnx.Param)
    _graphdef, graphstate = nnx.split((model, optimizer))
    training_env = template.env.reset(
        template.env, jax.random.split(reset_key, num_envs)
    )
    return replace(
        template, graphstate=graphstate, env=training_env, key=training_key
    )


keys = jax.random.split(jax.random.key(42), num_policies)
batch = jax.jit(jax.vmap(initialize_policy))(keys)


@jax.jit
def train_epoch(current, epoch):
    updated, _trajectory, metrics = jax.vmap(PPOTrainer.epoch, in_axes=(0, None))(
        current, epoch
    )
    return updated, metrics


for epoch in range(num_epochs):
    batch, metrics = train_epoch(batch, jnp.asarray(epoch))
    if epoch == 0 or (epoch + 1) % 16 == 0:
        losses = np.asarray(metrics["loss"])
        print(f"Epoch {epoch + 1:2d}: mean loss={losses.mean():.5f}")

batch.env.system.check_overflow()

# %%
# Evaluate All Eight Policies from an Identical Start
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reset one environment, broadcast it to ``num_policies`` copies, and roll out
# every policy deterministically with ``jax.vmap``:

common_start = env.reset(env, jax.random.key(202))
eval_envs = jax.vmap(lambda _: common_start)(jnp.arange(num_policies))


def evaluate_policy(trainer, start):
    model = trainer.model
    model.eval()

    def distance(current):
        return jnp.linalg.norm(
            current.state.pos_c[0] - current.env_params["objective"][0]
        )

    def step(current, _):
        pi, _val = model(current.observation(current), sequence=False)
        action = model.policy_output(pi)
        current, _terminated, _truncated = utils.advance_action(
            current, action, skip_frames=skip_frames
        )
        return current, distance(current)

    final, dists = jax.lax.scan(step, start, None, length=evaluation_steps)
    return jnp.concatenate((distance(start)[None], dists)), final


distances, _final_envs = jax.jit(jax.vmap(evaluate_policy))(batch, eval_envs)
distances = np.asarray(distances)
print(f"Initial distance: {distances[0, 0]:.4f}")
print("Final distances:", np.round(distances[:, -1], 4))

# %%
# Plot the Eight Trajectories
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

decision_interval = float(common_start.system.dt) * (1 + skip_frames)
time = np.arange(evaluation_steps + 1) * decision_interval
low, q25, q75, high = np.percentile(distances, [0, 25, 75, 100], axis=0)

fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
axis.fill_between(time, low, high, color="#326b9e", alpha=0.12, label="Full range")
axis.fill_between(time, q25, q75, color="#326b9e", alpha=0.24, label="Middle 50%")
axis.plot(time, distances.T, linewidth=1, alpha=0.55)
axis.plot(time, distances.mean(axis=0), color="#173f62", linewidth=2.5, label="Mean")
axis.set(
    xlabel="Physical time",
    ylabel="Distance to target",
    title="Eight trained policies from identical initial conditions",
)
axis.grid(alpha=0.2)
axis.legend()
plt.show()
