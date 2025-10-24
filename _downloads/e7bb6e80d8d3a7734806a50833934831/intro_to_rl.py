"""
Intro to JaxDEM Reinforcement Learning
--------------------------------------

In this example, we'll train a simple agent using JaxDEM's reinforcement learning tools.

The agent is a humble sphere that moves inside a box with reflective boundaries; the objective is
to reach a target location. We train it with Proximal Policy Optimization (PPO)
(:py:class:`~jaxdem.rl.trainers.PPOTrainer`) and a shared-parameters actor–critic MLP
(:py:class:`~jaxdem.rl.models.SharedActorCritic`).
"""
# %%
# Imports
# ~~~~~~~
import distrax
import jax
import jax.numpy as jnp

import jaxdem as jdem
import jaxdem.rl as rl

from flax import nnx
import optax

# %%
# Environment
# ~~~~~~~~~~~
# First, we create a single-agent navigation environment with reflective boundaries
# (uses sensible defaults for domain/time step internally). Check :py:class:`~jaxdem.rl.environments.SingleNavigator`
# for details.
env = rl.Environment.create("singleNavigator")

# %%
# Model
# ~~~~~
# Next, we build a shared-parameters actor–critic MLP. We can use a bijector to constrain the action space.

key = jax.random.key(1)
key, subkey = jax.random.split(key)
model = rl.Model.create(
    "SharedActorCritic",
    key=nnx.Rngs(subkey),
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
    action_space=rl.ActionSpace.create("maxNorm", max_norm=6.0),
)

# %%
# Trainer (PPO)
# ~~~~~~~~~~~~~
# Then, we construct the PPO trainer; feel free to tweak learning rate, num_epochs, etc. (:py:class:`~jaxdem.rl.trainers.PPOTrainer`)
# These parameters are chosen for the training to run very fast. Not really for quality. Using a bijector, we don't need to clip actions.
# However, if we wanted to, we could pass that option to the trainer.
key, subkey = jax.random.split(key)
tr = rl.Trainer.create(
    "PPO",
    env=env,
    model=model,
    key=subkey,
    num_epochs=100,  # 300
    num_envs=512,
    num_steps_epoch=32,
    num_minibatches=4,
    learning_rate=1e-1,  # 1e-1
)

# %%
# Training
# ~~~~~~~~
# Train the policy. Returns the updated trainer with learned parameters. This method is just a convenience
# training loop. If desired, one can iterate manually :py:meth:`~jaxdem.rl.trainers.trainer.epoch`
tr = tr.train(tr, directory="/tmp/runs", verbose=False)

# %%
# Testing the New Policy
# ~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have a trained agent, let's play around with it.
#
# We spawn the agent and periodically change the target it needs to go to. This way,
# we will have the agent chasing around the objective. When saving the simulation state,
# we add a small sphere to visualize where the agent needs to go.
tr.key, subkey = jax.random.split(tr.key)
tr.env = env.reset(env, subkey)  # replace the vectorized env with the serial one

writer = jdem.VTKWriter(directory="/tmp/frames")
state = tr.env.state.add(
    tr.env.state, pos=tr.env.env_params["objective"], rad=tr.env.state.rad / 5
)
state.ID = state.ID.at[..., state.N // 2 :].set(state.ID[..., : state.N // 2])

writer.save(state, tr.env.system)

for i in range(1, 2000):
    tr, _ = tr.step(tr)

    if i % 10 == 0:
        state = tr.env.state.add(
            tr.env.state,
            pos=tr.env.env_params["objective"],
            rad=tr.env.state.rad / 5,
        )
        state.ID = state.ID.at[..., state.N // 2 :].set(state.ID[..., : state.N // 2])

        writer.save(state, tr.env.system)

    # Change the objective without moving the agent
    if i % 500 == 0:
        key, subkey = jax.random.split(key)
        min_pos = tr.env.state.rad[0] * jnp.ones_like(tr.env.system.domain.box_size)
        objective = jax.random.uniform(
            subkey,
            (tr.env.max_num_agents, tr.env.state.dim),
            minval=min_pos,
            maxval=tr.env.system.domain.box_size - min_pos,
            dtype=float,
        )
        tr.env.env_params["objective"] = objective
