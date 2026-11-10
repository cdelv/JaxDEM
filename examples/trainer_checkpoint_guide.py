# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Checkpoint and resume training
=================================

This example saves a PPO trainer while it trains. It then loads a checkpoint,
continues the run, and loads only the policy. A trainer checkpoint holds the
environment, the model, the optimizer state, the random key, and the
hyperparameters. See :doc:`checkpoint_guide` for simulation checkpoints and
:doc:`intro_to_rl` for the trainer. Install ``JaxDEM[rl,io]``.
"""

import tempfile

import jax
import numpy as np
from flax import nnx

import jaxdem as jdem
import jaxdem.rl as rl

# %%
# Create the trainer
# ------------------
# The run is small, so the example is fast. Use your own settings.
env = rl.Environment.create("MultiNavigator", N=4, min_box_size=8.0, max_box_size=8.0)
policy = rl.Model.create(
    "MinGRUActorCritic",
    key=nnx.Rngs(0),
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
    action_space=rl.ActionSpace.create("maxNorm", max_norm=2.0),
    hidden_features=16,
    gru_features=16,
    num_layers=1,
)
trainer = rl.Trainer.create(
    "PPO",
    env=env,
    model=policy,
    key=jax.random.key(1),
    num_envs=8,
    num_steps_epoch=8,
    num_minibatches=2,
    num_epochs=8,
    skip_frames=4,
)

# %%
# Save while the trainer trains
# -----------------------------
# Pass a :py:class:`~jaxdem.writers.checkpoints.CheckpointTrainerWriter` to
# :py:meth:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer.train`. The step of a
# checkpoint is the number of completed epochs. ``save_every`` sets the
# interval, and the last epoch is always saved. ``max_to_keep`` limits disk use.
tmp_root = tempfile.TemporaryDirectory(prefix="jaxdem-trainer-checkpoint-guide-")
directory = tmp_root.name
with jdem.CheckpointTrainerWriter(directory, save_every=4, max_to_keep=3) as writer:
    trainer = trainer.train(
        trainer, verbose=False, log=False, checkpoint_writer=writer
    )

# %%
# Load a checkpoint and continue
# ------------------------------
# The loader builds the environment, model, optimizer, and trainer from the
# checkpoint. You do not build a trainer first. Pass the checkpoint step as
# ``start_epoch`` to continue the run.
with jdem.CheckpointTrainerLoader(directory) as loader:
    print("Saved epochs:", loader.checkpointer.all_steps())
    print("Latest epoch:", loader.latest_step())
    restored = loader.load(step=4)

resumed = restored.train(restored, start_epoch=4, verbose=False, log=False)

# The continued run gives the same policy as the run without an interruption.
for expected, actual in zip(
    jax.tree.leaves(trainer.graphstate), jax.tree.leaves(resumed.graphstate)
):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
print("The continued run and the first run agree.")

# %%
# Load only the policy
# --------------------
# :py:class:`~jaxdem.writers.checkpoints.CheckpointModelLoader` reads the model
# from the same directory. It does not read the environment or the optimizer.
# :py:class:`~jaxdem.writers.checkpoints.CheckpointLoader` reads the state and
# system of the training environment.
with jdem.CheckpointModelLoader(directory) as loader:
    model = loader.load()
model.eval()
print("Loaded policy:", type(model).__name__)

with jdem.CheckpointLoader(directory) as loader:
    state, system = loader.load()
print("Environment batch and particles:", state.pos.shape[:2])

# %%
# Notes
# -----
# - With your own epoch loop, call ``writer.save(trainer, epoch + 1)`` after
#   each epoch. The writer skips the steps that ``save_every`` excludes.
# - Trainers stacked with ``jax.vmap`` are saved and loaded as one batch. See
#   :doc:`batched_policy_training_guide`.
# - A custom environment, model, or trainer class must be registered. Import
#   its module before you load the checkpoint.
# - Metadata cannot describe an optimizer that is a ``lambda`` or a
#   ``functools.partial``. For such a checkpoint, build the trainer with your
#   code and pass it: ``loader.load(target=trainer)``.
# - Load only trusted checkpoints, because their metadata names Python
#   classes and functions to import.

tmp_root.cleanup()
