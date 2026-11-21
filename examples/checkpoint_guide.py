# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Checkpoint Save and Load
---------------------------

Save and restore full simulation state using ``JaxDEM[io]``:

- :py:class:`~jaxdem.writers.checkpoints.CheckpointWriter` /
  :py:class:`~jaxdem.writers.checkpoints.CheckpointLoader` — Orbax-backed
  checkpoints for long runs, retention policies (``max_to_keep``), and exact
  restarts (including batched ``State``/``System`` pytrees).
- :py:class:`~jaxdem.writers.h5_writer.H5Writer` / :py:func:`~jaxdem.utils.h5.load`
  — self-contained single-file HDF5 snapshots for archiving or sharing across
  tools.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp

import jaxdem as jdem

# %%
# Saving Simulation Checkpoints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.writers.checkpoints.CheckpointWriter.save` numbers each
# checkpoint automatically using ``system.step_count`` and writes
# asynchronously in the background. Use a ``with`` block so pending writes
# flush cleanly on exit. By default ``clean=False`` preserves existing
# checkpoints in ``directory``; pass ``clean=True`` to start fresh.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    rad=jnp.full(2, 0.2),
    vel=jnp.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
)
system = jdem.System.create(state.shape, dt=1e-3)
state, system = jdem.System.initialize(state, system)

tmp_root = tempfile.TemporaryDirectory(prefix="jaxdem-checkpoint-guide-")
sim_dir = Path(tmp_root.name) / "simulation"

with jdem.CheckpointWriter(directory=sim_dir, max_to_keep=2) as writer:
    writer.save(state, system)  # step 0
    state, system = jdem.System.step(state, system, n=5)
    writer.save(state, system)  # step 5
    state, system = jdem.System.step(state, system, n=5)
    writer.save(state, system)  # step 10

# %%
# Loading and Resuming Deterministically
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.writers.checkpoints.CheckpointLoader.load` returns
# ``(state, system)`` with collider caches, pair history, integrator state,
# callbacks, and PRNG key intact. Continue stepping directly — do **not** call
# :py:meth:`~jaxdem.system.System.initialize` again on a restored initialized
# checkpoint:

with jdem.CheckpointLoader(directory=sim_dir) as loader:
    print("Available steps:", loader.checkpointer.all_steps())
    print("Latest step:", loader.latest_step())
    state_latest, system_latest = loader.load()
    state_step_5, system_step_5 = loader.load(step=5)

resumed_state, resumed_system = jdem.System.step(state_step_5, system_step_5, n=5)
print(
    "Resumed step_count:",
    int(resumed_system.step_count),
    "| matches uninterrupted run:",
    bool(jnp.array_equal(resumed_state.pos, state.pos)),
)

# %%
# If you manage a writer manually without ``with``, call
# :py:meth:`~jaxdem.writers.checkpoints.BaseCheckpointManager.block_until_ready`
# before :py:meth:`~jaxdem.writers.checkpoints.BaseCheckpointManager.close`:
#
# .. code-block:: python
#
#     writer = jdem.CheckpointWriter(directory=sim_dir, max_to_keep=2)
#     writer.save(state, system)
#     writer.block_until_ready()
#     writer.close()
#
# Load only trusted checkpoints because restoration resolves recorded Python
# classes and callables.

# %%
# Bonded-Force Model Checkpointing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.bonded_forces.BondedForceModel` instances (such as
# :py:class:`~jaxdem.bonded_forces.deformable_particle.DeformableParticleModel`)
# are reconstructed automatically from their saved type and parameters — you do
# not need any extra serialization setup for their internal force callbacks:

vertices_2d = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
elements_2d = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
bonded_model = jdem.BondedForceModel.create(
    "deformable_particle_model",
    vertices=vertices_2d,
    elements=elements_2d,
    edges=elements_2d,
    element_adjacency=elements_2d,
    em=1.0,
)
state_bonded = jdem.State.create(pos=vertices_2d, rad=jnp.full(4, 0.1))
system_bonded = jdem.System.create(state_bonded.shape, bonded_force_model=bonded_model)
state_bonded, system_bonded = jdem.System.initialize(state_bonded, system_bonded)

bonded_dir = Path(tmp_root.name) / "simulation_bonded"
with jdem.CheckpointWriter(directory=bonded_dir) as writer:
    writer.save(state_bonded, system_bonded)

with jdem.CheckpointLoader(directory=bonded_dir) as loader:
    state_restored, system_restored = loader.load()
    print("Restored bonded model:", system_restored.bonded_force_model.type_name)

# %%
# Custom Force Functions and Callbacks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Custom force/energy functions, minimizer constructors, ``target_fn``, and
# pre/post-step callbacks are serialized by their fully-qualified import path
# (e.g. ``my_forces.harmonic_trap``). Closures, lambdas, and functions defined
# in ``__main__`` are rejected on save. Define reusable callbacks in an
# importable module so checkpoints reload from any script:
#
# .. code-block:: python
#
#     # my_forces.py
#     import jax.numpy as jnp
#
#     def harmonic_trap(pos, state, system):
#         return -pos, jnp.zeros_like(state.torque)
#
#     # main.py
#     from my_forces import harmonic_trap
#     system = jdem.System.create(
#         state.shape, force_manager_kw={"force_functions": [harmonic_trap]}
#     )

# %%
# Single-File HDF5 Snapshots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Choose :py:class:`~jaxdem.writers.checkpoints.CheckpointWriter` (Orbax) for rotating
# training/simulation checkpoints, and :py:class:`~jaxdem.writers.h5_writer.H5Writer`
# when you want portable single-file archives. Each ``save`` call writes
# ``snapshot_00000000.h5``, ``snapshot_00000001.h5``, … preserving batch and
# trajectory shapes:

h5_dir = Path(tmp_root.name) / "hdf5"
with jdem.H5Writer(directory=h5_dir) as writer:
    writer.save(state, system)

state_h5, system_h5 = jdem.utils.h5.load(str(h5_dir / "snapshot_00000000.h5"))
state_h5, system_h5 = jdem.System.step(state_h5, system_h5)
print("HDF5 restored & stepped to:", int(system_h5.step_count))

tmp_root.cleanup()
