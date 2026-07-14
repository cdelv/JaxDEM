"""Intro to JaxDEM Reinforcement Learning.
------------------------------------------

In this example, we'll train a simple agent using JaxDEM's reinforcement learning tools.

The agent is a humble sphere that moves inside a box with reflective boundaries; the objective is
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
# First, we create a single-agent navigation environment with reflective boundaries
# (uses sensible defaults for domain/time step internally). Check :py:class:`~jaxdem.rl.environments.SingleNavigator`
# for details.

env = rl.Environment.create(
    "single_navigator",
    max_steps=num_steps_epoch * reset_every * skip_frames,
)

# %%
# Model
# ~~~~~
# Next, we build a shared-parameters actor–critic MLP. We can use a bijector to constrain the action space.
# Registry keys are case- and underscore-insensitive, so ``"max_norm"`` and ``"MaxNorm"`` are equivalent;

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
# Then, we construct the PPO trainer; feel free to tweak learning rate, num_epochs, etc. (:py:class:`~jaxdem.rl.trainers.PPOTrainer`)
# These parameters are chosen for the training to run very fast. Not really for quality. Using a bijector, we don't need to clip actions.
# However, if we wanted to, we could pass that option to the trainer.

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
# Train the policy. Returns the updated trainer with learned parameters. This method is just a convenience
# training loop. If desired, one can iterate manually :py:meth:`~jaxdem.rl.trainers.Trainer.epoch`
tmp_runs = Path(tempfile.gettempdir()) / "runs"
tr = tr.train(tr, directory=tmp_runs, verbose=False, log=False)

# %%
# Testing the New Policy
# ~~~~~~~~~~~~~~~~~~~~~~~
# Now that we have a trained agent, let's play around with it.
#
# We spawn the agent and periodically change the target it needs to go to. This way,
# we will have the agent chasing around the objective. When saving the simulation state,
# we add a small sphere to visualize where the agent needs to go.
tr.key, subkey = jax.random.split(tr.key)
env = env.reset(
    env, subkey
)  # re-seed and reset the serial env; the trainer used its own vectorized copy

tmp_frames = Path(tempfile.gettempdir()) / "frames"
writer = jdem.VTKWriter(directory=tmp_frames)
state = env.state.add(env.state, pos=env.env_params["objective"], rad=env.state.rad / 5)
writer.save(state, env.system)


# %%
# We have some utilities that will help drive the environment more efficiently. But to use them, we need to create a
# policy function. Each :py:func:`~jaxdem.utils.env_step` call advances ``n`` logical steps, and by default each
# logical step runs exactly one physics frame (``1 + skip_frames`` in general). So with ``n=1``, the loop below saves
# a frame every 10 physics steps and moves the objective every 200.
@jax.jit
def policy_model(obs, key, graphstate, graphdef):
    model = nnx.merge(graphdef, graphstate)
    pi, _value = model(obs, sequence=False)
    action = pi.sample(seed=key)
    _, graphstate = nnx.split(model)
    return action, graphstate


# %%
# NOTE: If using a recurrent model (like LSTMActorCritic or MinGRUActorCritic), we must
# reset its internal memory before running the policy. It is good practice to always
# call reset, as non-recurrent models will simply ignore it.

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
