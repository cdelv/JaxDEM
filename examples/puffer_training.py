"""
Training with PufferLib
-----------------------

This example demonstrates how to train a JaxDEM RL environment using
PufferLib's PPO implementation and then visualise the learned behaviour
by saving VTK frames.

Requirements::

    pip install pufferlib torch
"""

# %%
# Imports
# ~~~~~~~
import jax
import jax.numpy as jnp
import numpy as np
import torch

import jaxdem as jdem
import jaxdem.rl as rl
import pufferlib.pufferl
import pufferlib.pytorch

from jaxdem.rl.puffer import JaxDEMPufferEnv, ContinuousPolicy, default_config
from pathlib import Path

# %%
# Variables
# ~~~~~~~~~
N = 8
num_envs = 4
frames_dir = Path("/tmp/puffer_frames")
save_every = 10
T = 2000
device = "cuda"

# %%
# Environment and Policy
# ~~~~~~~~~~~~~~~~~~~~~~
# Create a vectorised JaxDEM environment wrapped as a PufferLib env.
# ``num_envs`` controls how many parallel copies run via ``jax.vmap``.
vecenv = JaxDEMPufferEnv(
    "MultiNavigator",
    num_envs=num_envs,
    seed=42,
    N=N,
)

# The policy is a simple two-layer MLP with continuous (Normal) outputs.
# Move it to the training device before creating the trainer.
policy = ContinuousPolicy(vecenv, hidden_size=128).to(device)

# %%
# Configuration
# ~~~~~~~~~~~~~
# :func:`~jaxdem.rl.puffer.default_config` provides sensible PPO
# hyper-parameters.  Any key can be overridden.
config = default_config(
    env_name="MultiNavigator",
    total_timesteps=500_000,
    device=device,
    learning_rate=3e-4,
)

# %%
# Training Loop
# ~~~~~~~~~~~~~
# The ``PuffeRL`` trainer alternates between collecting experience
# (``evaluate``) and updating the policy (``train``).
trainer = pufferlib.pufferl.PuffeRL(config, vecenv, policy)

while trainer.global_step < config["total_timesteps"]:
    trainer.evaluate()
    trainer.train()

trainer.close()

# %%
# Testing the Learned Policy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# After training we create a *single* (non-vectorised) copy of the
# environment and drive it with the learned policy, saving VTK frames
# for visualisation.
key = jax.random.key(123)
env = rl.Environment.create("MultiNavigator", N=N)
env = rl.clip_action_env(env)

key, subkey = jax.random.split(key)
env = env.reset(env, subkey)

writer = jdem.VTKWriter(directory=frames_dir)
policy.eval()

for i in range(T):
    obs = env.observation(env)
    obs_torch = torch.as_tensor(np.array(obs), device=device).float()

    with torch.no_grad():
        dist, _ = policy.forward_eval(obs_torch)
        action = dist.sample()

    action_np = action.cpu().numpy()
    env = env.step(env, jnp.asarray(action_np))

    done = env.done(env)
    if done:
        key, subkey = jax.random.split(key)
        env = env.reset(env, subkey)

    if i % save_every == 0:
        writer.save(env.state, env.system)
