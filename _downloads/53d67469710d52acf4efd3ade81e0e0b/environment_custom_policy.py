"""Driving Environments with a Custom Policy.
---------------------------------------------

In this example, we create an environment instance and drive it with a
custom policy.
"""

# %%
# Imports
# ~~~~~~~~
import tempfile
from pathlib import Path

import jax
from flax import nnx

import jaxdem as jdem
import jaxdem.rl as rl
from jaxdem import utils

# %%
# Variables
# ~~~~~~~~~
# First, we define all the variables needed for the example.
frames_dir = Path(tempfile.gettempdir()) / "frames"
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
def model(obs, key, graphstate, graphdef):
    base_model = nnx.merge(graphdef, graphstate)
    pi, _value = base_model(obs, sequence=False)
    action = pi.sample(seed=key)
    _, new_graphstate = nnx.split(base_model)
    return action, new_graphstate


# %%
# Model and Environment
# ~~~~~~~~~~~~~~~~~~~~~
# Now we create a model and an environment to use in the example.
# We do not train the model here. The goal is to show
# how to drive the environment directly.
#
# You can load a trained model in the same way with
# :py:class:`~jaxdem.writers.CheckpointModelLoader`.
env = rl.Environment.create("multi_navigator", N=N)

key, subkey = jax.random.split(key)
base_model = rl.Model.create(
    "SharedActorCritic",
    key=nnx.Rngs(subkey),
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
)
base_model.eval()

# %%
# NOTE: For a recurrent model (like LSTMActorCritic or MinGRUActorCritic), reset
# its internal memory before running the policy. Always call reset.
# Non-recurrent models ignore it.

base_model.reset(
    shape=(num_envs, env.max_num_agents, 1),
    mask=None,
)
graphdef, graphstate = nnx.split(base_model)

# %%
# Environment Vectorization
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM supports vectorized environments, so multiple simulations can
# run in parallel. Use this to gather statistics about the environment.
# Passing ``n`` to :py:func:`~jaxdem.rl.vectorise_env` broadcasts the scalar
# environment to a batch of ``n`` copies. We then reset each copy with its own key.
key, subkey = jax.random.split(key)
subkeys = jax.random.split(subkey, num_envs)
env = rl.vectorise_env(env, n=num_envs)
env = env.reset(env, subkeys)

# %%
# Driving the Environment
# ~~~~~~~~~~~~~~~~~~~~~~~
# There are two main ways to drive an environment. The first is by stepping
# it manually for a fixed number of steps. By default each logical step
# advances exactly one physics frame (``1 + skip_frames`` in general), so the
# call below runs ``save_every`` physics frames:
env, key, graphstate = utils.env_step(
    env,
    model,
    key,
    graphstate,
    graphdef=graphdef,
    n=save_every,
)

# %%
# The second approach is to roll out a trajectory, collecting data every
# `stride` steps:
env, key, graphstate, env_traj = utils.env_trajectory_rollout(
    env,
    model,
    key,
    graphstate,
    graphdef=graphdef,
    n=batches - 1,
    stride=save_every,
)

# %%
# Saving Data
# ~~~~~~~~~~~
# Finally, we use JaxDEM’s :py:class:`~jaxdem.writers.VTKWriter` to save
# the full rollout to disk in a single call:
writer = jdem.VTKWriter(directory=frames_dir)
writer.save(env_traj.state, env_traj.system, trajectory=True)
