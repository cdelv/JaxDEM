"""
Intro to JaxDEM Reinforcement Learning
--------------------------------------

In this example, we’ll train a simple agent using JaxDEM’s reinforcement learning tools.

The agent is a humble sphere that moves inside a box with reflective boundaries; the objective is
to reach a target location. We train it with Proximal Policy Optimization (PPO)
(:py:class:`jaxdem.rl.trainer.PPOTrainer`) and a shared-parameters actor–critic MLP
(:py:class:`jaxdem.rl.model.SharedActorCritic`).
"""
# %%
# Imports
# ~~~~~~~
import jax
import jax.numpy as jnp

import jaxdem as jdem
import jaxdem.rl as rl
from dataclasses import replace

from flax import nnx

Save = False

# %%
# Environment
# ~~~~~~~~~~~
# First, we create a single-agent navigation environment with reflective boundaries
# (uses sensible defaults for domain/time step internally). Check :py:class:`jaxdem.rl.environment.SingleNavigator`
# for details.
env = rl.Environment.create("singleNavigator")

# %%
# Model
# ~~~~~
# Next, we build a shared-parameters actor–critic MLP.
key = jax.random.key(1)
key, subkey = jax.random.split(key)
model = rl.Model.create(
    "SharedActorCritic",
    key=nnx.Rngs(subkey),
    observation_space=env.observation_space_size,
    action_space=env.action_space_size,
    architecture=[24, 24],
)

# %%
# Trainer (PPO)
# ~~~~~~~~~~~~~
# Then, we construct the PPO trainer; feel free to tweak learning rate, num_epochs, etc. (:py:class:`jaxdem.rl.trainer.PPOTrainer`)
# This parameters are chosen for the training to run very fast. Not really for quality.
key, subkey = jax.random.split(key)
tr = rl.Trainer.create(
    "PPO",
    env=env,
    model=model,
    key=subkey,
    num_epochs=220,
    num_envs=256,
    num_steps_epoch=64,
    num_minibatches=3,
    minibatch_size=86,
    learning_rate=1e-1,
)

# %%
# Training
# ~~~~~~~~
# Train the policy. Returns the updated trainer with learned parameters. This method is just a convenience
# training loop. If desired, one can iterate manually :py:meth:`jaxdem.rl.trainer.epoch`
tr, _ = tr.train(tr, verbose=False)

# %%
# Testing the New Policy
# ~~~~~~~~~~~~~~~~~~~~~
# Now that we have a trained agent, lets play arround with it.
#
# We spawn the agent and periodically change the target it needs to go to. This way,
# we will have the agent chasing arround the objective. When saving the simulation state,
# we add a small sphere to the state to visualize where the agent needs to go.
key, subkey = jax.random.split(key)
tr = replace(tr, env=env.reset(env, subkey))

if Save:
    writer = jdem.VTKWriter()
    state = tr.env.state.add(
        tr.env.state, pos=tr.env.env_params["objective"], rad=tr.env.state.rad / 5
    )
    writer.save(state, tr.env.system)

for i in range(1, 2000):
    tr, _ = tr.step(tr)

    if i % 10 == 0 and Save:
        state = tr.env.state.add(
            tr.env.state, pos=tr.env.env_params["objective"], rad=tr.env.state.rad / 5
        )
        writer.save(state, tr.env.system)

    # Change the objective without moving the agent
    if i % 200 == 0:
        key, subkey = jax.random.split(key)
        min_pos = tr.env.state.rad * jnp.ones_like(tr.env.system.domain.box_size)
        objective = jax.random.uniform(
            subkey,
            (1, tr.env.state.dim),
            minval=min_pos,
            maxval=tr.env.system.domain.box_size - min_pos,
            dtype=float,
        )
        tr.env.env_params["objective"] = objective
