"""Intro to JaxDEM Reinforcement Learning.
------------------------------------------

In this example, we train a simple agent with JaxDEM's reinforcement learning tools.

The agent is a sphere that moves in a box with reflective boundaries. The objective is
to reach a target location. We train it with Proximal Policy Optimization (PPO)
(:py:class:`~jaxdem.rl.trainers.PPOTrainer`) and a shared-parameters actor–critic MLP
(:py:class:`~jaxdem.rl.models.SharedActorCritic`).
"""

import tempfile
from pathlib import Path

# %%
# Imports
# ~~~~~~~
import jax
import jax.numpy as jnp
from flax import nnx
from jax._src.ad_util import stop_gradient_p

import jaxdem as jdem
import jaxdem.rl as rl
from jaxdem import utils

num_steps_epoch = 100
reset_every = 40
skip_frames = 50
num_envs = 32

# %%
# Environment
# ~~~~~~~~~~~
# First, we create a single-agent navigation environment with reflective boundaries.
# It uses default values for the domain and time step. See
# :py:class:`~jaxdem.rl.environments.SingleNavigator` for details.

env = rl.Environment.create(
    "single_navigator",
    max_steps=num_steps_epoch * reset_every * skip_frames,
)

# %%
# Model
# ~~~~~
# Next, we build a shared-parameters actor–critic MLP. We can use a bijector to constrain the action space.
# Registry keys are case- and underscore-insensitive, so ``"max_norm"`` and ``"MaxNorm"`` are equivalent.

model = rl.Model.create(
    "SharedActorCritic",
    key=nnx.Rngs(jax.random.key(1)),
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
    action_space=rl.ActionSpace.create("max_norm", max_norm=1.0),
)

# %%
# Trainer (PPO)
# ~~~~~~~~~~~~~
# Then, we create the PPO trainer. You can change the learning rate, num_epochs, and other options
# (:py:class:`~jaxdem.rl.trainers.PPOTrainer`).
# We choose these parameters so training runs fast, not for quality. With a bijector, we do not need to clip actions.
# To clip actions anyway, pass that option to the trainer.

key = jax.random.key(6)
tr = rl.Trainer.create(
    "PPO",
    env=env,
    model=model,
    key=key,
    num_steps_epoch=num_steps_epoch,
    num_envs=num_envs,
    num_epochs=1080,  # We anneal the learning rate
    stop_at_epoch=reset_every * 6,
    skip_frames=skip_frames,
    learning_rate=2e-3,
)

# %%
# Training
# ~~~~~~~~
# Train the policy. This returns the updated trainer with the learned parameters. This method is a
# convenience training loop. To control the loop yourself, call
# :py:meth:`~jaxdem.rl.trainers.Trainer.epoch` directly.
tmp_runs = Path(tempfile.gettempdir()) / "runs"
tr = tr.train(tr, directory=tmp_runs, verbose=False, log=False)

# %%
# Testing the New Policy
# ~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have a trained agent, we test it.
#
# We reset the agent and move the target periodically, so the agent chases the objective.
# When we save the simulation state, we add a small sphere at the target to show
# where the agent must go.
tr.key, subkey = jax.random.split(tr.key)
env = env.reset(
    env, subkey
)  # re-seed and reset the serial env. The trainer used its own vectorized copy.

tmp_frames = Path(tempfile.gettempdir()) / "frames"
writer = jdem.VTKWriter(directory=tmp_frames)
state = env.state.add(env.state, pos=env.env_params["objective"], rad=env.state.rad / 5)
writer.save(state, env.system)


# %%
# JaxDEM has utilities that drive the environment. To use them, we create a policy function.
# Each :py:func:`~jaxdem.utils.env_step` call advances ``n`` logical steps. Each logical step runs
# ``1 + skip_frames`` physics frames. The loop below calls ``env_step`` with ``n=10``, so each saved
# frame covers 10 logical steps, and the objective moves every 20 calls (200 logical steps).
@jax.jit
def policy_model(obs, key, graphstate, graphdef):
    model = nnx.merge(graphdef, graphstate)
    pi, _value = model(obs, sequence=False)
    action = pi.sample(seed=key)
    _, graphstate = nnx.split(model)
    return action, graphstate


# %%
# NOTE: With a recurrent model (like LSTMActorCritic or MinGRUActorCritic), we must
# reset its internal memory before we run the policy. It is good practice to always
# call reset, because non-recurrent models ignore it.

base_model = tr.model
base_model.reset(
    shape=(env.max_num_agents, 1),
    mask=None,
)
graphdef, graphstate = nnx.split(base_model)

for _ in range(5):  # 1000 total steps / 200 steps per objective change
    for _ in range(200 // 10):
        env, tr.key, graphstate = utils.env_step(
            env,
            policy_model,
            tr.key,
            graphstate,
            graphdef=graphdef,
            n=10,
            skip_frames=skip_frames,
        )

        state = env.state.add(
            env.state,
            pos=env.env_params["objective"],
            rad=env.state.rad / 5,
        )
        writer.save(state, env.system)

    tr.key, subkey = jax.random.split(tr.key)
    min_pos = env.state.rad[0] * jnp.ones_like(env.system.domain.box_size)
    objective = jax.random.uniform(
        subkey,
        (env.max_num_agents, env.state.dim),
        minval=min_pos,
        maxval=env.system.domain.box_size - min_pos,
        dtype=float,
    )
    env.env_params["objective"] = objective
