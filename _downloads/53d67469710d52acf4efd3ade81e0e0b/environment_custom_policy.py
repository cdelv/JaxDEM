"""
Driving Environments with a Custom Policy
-----------------------------------------

In this example, we create an environment instance and show how to drive it
efficiently using a custom policy. This approach removes the need to create a
trainer object, making evaluation much more efficient.
"""
# %%
# Imports
# ~~~~~~~~
import jax
import jax.numpy as jnp
from flax import nnx

import jaxdem as jdem
import jaxdem.rl as rl
from jaxdem import utils

from pathlib import Path

# %%
# Variables
# ~~~~~~~~~
# First, we define all the variables needed for the example.
frames_dir = Path("/tmp/frames")
key = jax.random.key(1)
N = 24
save_every = 40
T = 4000
batches = T // save_every
num_envs = 40


# %%
# The Policy
# ~~~~~~~~~~
# Next, we define a callable that takes the observations and some keyword
# arguments, and returns the corresponding actions. For more information,
# see :py:func:`~jaxdem.utils.env_step`.
#
# In this example, we drive the environment with a model from JaxDEM using
# ``nnx``. However, `model` can be any JIT-compatible function.
def model(obs, graphdef, graphstate):
    base_model = nnx.merge(graphdef, graphstate)
    pi, value = base_model(obs, sequence=False)
    action = pi.sample(seed=1)
    return action


# %%
# Model and Environment
# ~~~~~~~~~~~~~~~~~~~~~
# Now we create a model and an environment to use in the example.
# We will not perform any training here, since the goal is to show
# how to drive the environment directly.
#
# A trained model could be loaded in the same way using
# :py:class:`~jaxdem.writers.CheckpointModelLoader`.
env = rl.Environment.create("MultiNavigator", N=N)

key, subkey = jax.random.split(key)
base_model = rl.Model.create(
    "SharedActorCritic",
    key=nnx.Rngs(subkey),
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
)
base_model.eval()
graphdef, graphstate = nnx.split(base_model)

# %%
# Environment Vectorization
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM supports vectorized environments, allowing multiple simulations to
# run in parallel for significant speedups. This is usefull for gathering statistics about the environmentt.
subkeys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(num_envs))
env = jax.vmap(lambda _: env)(jnp.arange(num_envs))
env = rl.vectorise_env(env)
env = env.reset(env, subkeys)

# %%
# Driving the Environment
# ~~~~~~~~~~~~~~~~~~~~~~~
# There are two main ways to drive an environment. The first is by stepping
# it manually for a fixed number of steps:
env = utils.env_step(
    env,
    model,
    graphdef=graphdef,
    graphstate=graphstate,
    n=save_every,
)

# %%
# The second approach is to roll out a trajectory, collecting data every
# `stride` steps:
env, env_traj = utils.env_trajectory_rollout(
    env,
    model,
    graphdef=graphdef,
    graphstate=graphstate,
    n=batches,
    stride=save_every,
)

# %%
# Saving Data
# ~~~~~~~~~~~
# Finally, we can use JaxDEM’s :py:class:`~jaxdem.writers.VTKWriter` to save
# the full rollout to disk in a single call:
writer = jdem.VTKWriter(directory=frames_dir)
writer.save(env_traj.state, env_traj.system, trajectory=True)
