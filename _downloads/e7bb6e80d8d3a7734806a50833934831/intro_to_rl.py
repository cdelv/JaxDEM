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
import jax
import jax.numpy as jnp

import jaxdem as jdem
import jaxdem.rl as rl
from jaxdem import utils

from flax import nnx

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
)

# %%
# Training
# ~~~~~~~~
# Train the policy. Returns the updated trainer with learned parameters. This method is just a convenience
# training loop. If desired, one can iterate manually :py:meth:`~jaxdem.rl.trainers.trainer.epoch`
tr = tr.train(tr, directory="/tmp/runs", verbose=False, log=False)

# %%
# Testing the New Policy
# ~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have a trained agent, let's play around with it.
#
# We spawn the agent and periodically change the target it needs to go to. This way,
# we will have the agent chasing around the objective. When saving the simulation state,
# we add a small sphere to visualize where the agent needs to go.
tr.key, subkey = jax.random.split(tr.key)
env = env.reset(env, subkey)  # replace the vectorized env with the serial one

writer = jdem.VTKWriter(directory="/tmp/frames")
state = env.state.add(env.state, pos=env.env_params["objective"], rad=env.state.rad / 5)
state.clump_ID = state.clump_ID.at[..., state.N // 2 :].set(
    state.clump_ID[..., : state.N // 2]
)
writer.save(state, env.system)


# %%
# We have some utilities that will help drive the environment more efficiently. But to use them, we need to create a
# policy function:
@jax.jit
def policy_model(obs, key, graphdef, graphstate):
    base_model = nnx.merge(graphdef, graphstate)
    pi, value = base_model(obs, sequence=False)
    action = pi.sample(seed=key)
    return action


base_model = tr.model
graphdef, graphstate = nnx.split(base_model)


for i in range(1, 1000):
    tr.key, subkey = jax.random.split(tr.key)
    env = utils.env_step(
        env,
        policy_model,
        subkey,
        graphdef=graphdef,
        graphstate=graphstate,
        n=1,
    )

    if i % 10 == 0:
        state = env.state.add(
            env.state,
            pos=env.env_params["objective"],
            rad=env.state.rad / 5,
        )
        state.clump_ID = state.clump_ID.at[..., state.N // 2 :].set(
            state.clump_ID[..., : state.N // 2]
        )

        writer.save(state, env.system)

    # Change the objective without moving the agent
    if i % 200 == 0:
        key, subkey = jax.random.split(key)
        min_pos = env.state.rad[0] * jnp.ones_like(env.system.domain.box_size)
        objective = jax.random.uniform(
            subkey,
            (env.max_num_agents, env.state.dim),
            minval=min_pos,
            maxval=env.system.domain.box_size - min_pos,
            dtype=float,
        )
        env.env_params["objective"] = objective
