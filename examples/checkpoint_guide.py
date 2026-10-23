"""Checkpoint Save and Load
----------------------------

This guide introduces JaxDEM checkpointing utilities:

- :py:class:`~jaxdem.writers.checkpoints.CheckpointWriter`
- :py:class:`~jaxdem.writers.checkpoints.CheckpointLoader`
- :py:class:`~jaxdem.writers.h5_writer.H5Writer`

Checkpoints are useful for long simulations, reproducibility, and restarting
from intermediate steps. Install ``JaxDEM[io]`` for Orbax and HDF5 support.
The walkthrough writes only three tiny dynamic snapshots and one small bonded
model; all output lives in an isolated temporary directory.
"""

import tempfile
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import jaxdem as jdem

# %%
# Saving simulation checkpoints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We create a small simulation, run it in chunks with
# :py:meth:`~jaxdem.system.System.step`, and save snapshots with
# :py:meth:`~jaxdem.writers.checkpoints.CheckpointWriter.save`.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    rad=jnp.full(2, 0.2),
    vel=jnp.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
)
system = jdem.System.create(state.shape, dt=1e-3)
state, system = jdem.System.initialize(state, system)

tmp_root = tempfile.TemporaryDirectory(prefix="jaxdem-checkpoint-guide-")
tmp_dir = Path(tmp_root.name) / "simulation"
with jdem.CheckpointWriter(directory=tmp_dir, max_to_keep=2) as writer:
    # By default (clean=False) the writer preserves any checkpoints already
    # present in the directory, so re-opening a writer on the same directory
    # is safe. Pass clean=True to erase the directory and start fresh.
    writer.save(state, system)  # step 0

    state, system = system.step(state, system, n=5)
    writer.save(state, system)  # step 5

    state, system = system.step(state, system, n=5)
    writer.save(state, system)  # step 10


# %%
# Loading latest and specific checkpoints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`load() <jaxdem.writers.checkpoints.CheckpointLoader.load>` returns ``(state, system)``. Query the
# current latest step with
# :py:meth:`~jaxdem.writers.checkpoints.CheckpointLoader.latest_step`.

with jdem.CheckpointLoader(directory=tmp_dir) as loader:
    # ``loader.checkpointer`` is the underlying Orbax CheckpointManager.
    # ``all_steps()`` has no public CheckpointLoader equivalent yet.
    print("Available steps:", loader.checkpointer.all_steps())
    print("Latest step:", loader.latest_step())

    state_latest, system_latest = loader.load()
    print("Loaded latest step_count:", int(system_latest.step_count))

    state_step_5, system_step_5 = loader.load(step=5)
    print("Loaded step_count=5:", int(system_step_5.step_count))
    print("State shape at step 5:", state_step_5.pos.shape)

# Restart from step 5 without reinitializing. The loaded integration state
# must reproduce the uninterrupted run at step 10, not just its array shapes.
resumed, resumed_system = jdem.System.step(state_step_5, system_step_5, n=5)
np.testing.assert_array_equal(resumed.pos, state.pos)
np.testing.assert_array_equal(resumed.vel, state.vel)
assert int(resumed_system.step_count) == int(system.step_count) == 10
np.testing.assert_array_equal(resumed_system.time, system.time)
print("Restarted and uninterrupted positions/velocities match exactly.")

# %%
# Resource management note:
# Use :py:class:`~jaxdem.writers.checkpoints.CheckpointWriter` and
# :py:class:`~jaxdem.writers.checkpoints.CheckpointLoader`
# with a context manager (``with ... as ...``). The context manager waits for
# pending async writes and closes resources on exit. You can also manage both
# classes manually without ``with``.
#
# If you use them manually, remember:
#
# .. code-block:: python
#
#     writer = jdem.CheckpointWriter(directory=sim_checkpoint_dir, max_to_keep=2)
#     writer.save(state, system)   # async
#     writer.block_until_ready()   # ensure writes are finished
#     writer.close()               # release resources
#
# Checkpoint saving is asynchronous. Call
# :py:meth:`block_until_ready() <jaxdem.writers.checkpoints.BaseCheckpointManager.block_until_ready>` before
# program exit. When you manage the writer manually, also call it before
# :py:meth:`close() <jaxdem.writers.checkpoints.BaseCheckpointManager.close>`, so all files reach the disk.
# Continue a loaded initialized checkpoint directly; do not call
# :py:meth:`~jaxdem.system.System.initialize` again. Checkpoints include
# collider/contact history, integrator state, callbacks, and PRNG state needed
# for deterministic continuation. Load only trusted checkpoints because their
# metadata resolves Python classes and callables.

# %%
# Bonded-force model checkpointing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Checkpointing also supports systems with
# :py:class:`~jaxdem.bonded_forces.BondedForceModel` instances, such as
# :py:class:`~jaxdem.bonded_forces.deformable_particle.DeformableParticleModel`.
#
# .. note::
#
#    Bonded-force models register their own force and energy functions with
#    the :py:class:`~jaxdem.forces.force_manager.ForceManager` automatically
#    during :py:meth:`~jaxdem.system.System.create`. Checkpointing does **not**
#    serialize these internal functions like user-supplied custom forces.
#    The loader rebuilds them from the bonded model type and parameters.
#    You only need serialization for the custom force functions that
#    **you** add via ``force_manager_kw``.

vertices_2d = jnp.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)
elements_2d = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
adjacency_2d = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

bonded_model = jdem.BondedForceModel.create(
    "deformable_particle_model",
    vertices=vertices_2d,
    elements=elements_2d,
    edges=elements_2d,
    element_adjacency=adjacency_2d,
    em=1.0,
)

state_bonded = jdem.State.create(pos=vertices_2d, rad=jnp.full(4, 0.1))
system_bonded = jdem.System.create(
    state_bonded.shape,
    bonded_force_model=bonded_model,
)

# Initialize before saving so this snapshot can also continue directly.
state_bonded, system_bonded = jdem.System.initialize(state_bonded, system_bonded)

# Use a separate directory to avoid mixing the bonded snapshot with the
# earlier ones.
tmp_dir_bonded = Path(tmp_root.name) / "simulation_bonded"
with jdem.CheckpointWriter(directory=tmp_dir_bonded) as writer:
    writer.save(state_bonded, system_bonded)
    writer.block_until_ready()

with jdem.CheckpointLoader(directory=tmp_dir_bonded) as loader:
    state_restored, system_restored = loader.load()
    np.testing.assert_array_equal(state_restored.pos, state_bonded.pos)
    print(
        "Restored bonded model:",
        (
            None
            if system_restored.bonded_force_model is None
            else system_restored.bonded_force_model.type_name
        ),
    )

# %%
# Custom force functions and checkpointing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The writer serializes custom force functions passed via ``force_manager_kw``
# by their fully-qualified module path (e.g. ``mypackage.forces.harmonic_trap``).
#
# .. warning::
#
#    Checkpointed custom forces, energy functions, targets, and pre/post-step
#    callbacks must be module-level objects in an importable module. The writer
#    resolves every recorded path and verifies object identity before saving;
#    closures, local functions, and functions in ``__main__`` are rejected.
#    If the loader later cannot resolve a required physics function,
#    :py:meth:`~jaxdem.writers.checkpoints.CheckpointLoader.load` raises a
#    `RuntimeError <https://docs.python.org/3/library/exceptions.html#RuntimeError>`_
#    by default (``strict=True``). ``strict=False`` permits
#    explicitly degraded loading of legacy custom force/energy entries, while
#    required callbacks in current checkpoints still fail rather than silently
#    changing the protocol.
#
# To make checkpoints load from any script, define your custom force and
# energy functions, and any custom minimizer constructors, in a separate **importable module**:
#
# .. code-block:: python
#
#     # my_forces.py  <-- importable module
#     import jax
#     import jax.numpy as jnp
#
#     def harmonic_trap(pos, state, system):
#         k = 1.0
#         return -k * pos, jnp.zeros_like(state.torque)
#
#     def harmonic_trap_energy(pos, state, system):
#         k = 1.0
#         return 0.5 * k * jnp.sum(pos ** 2, axis=-1)
#
# Then use them in your simulation:
#
# .. code-block:: python
#
#     from my_forces import harmonic_trap, harmonic_trap_energy
#
#     system = jdem.System.create(
#         state.shape,
#         force_manager_kw=dict(
#             force_functions=[(harmonic_trap, harmonic_trap_energy)],
#         ),
#     )
#
# You can load checkpoints saved this way from **any** script that has
# ``my_forces`` on its Python path.

# %%
# HDF5 snapshots
# ~~~~~~~~~~~~~~
# :py:class:`~jaxdem.writers.h5_writer.H5Writer` uses the shared asynchronous writer queue and the existing
# HDF5
# encoding. Its :py:meth:`~jaxdem.writers.h5_writer.H5Writer.save` method stores each
# :py:class:`~jaxdem.state.State`/:py:class:`~jaxdem.system.System` pair in a self-contained file.
# Batches and trajectories keep their original array shapes in that file.
# Numbering follows saved calls and continues after existing snapshots.
# Read a saved snapshot with :py:func:`~jaxdem.utils.h5.load`.

h5_directory = Path(tmp_root.name) / "hdf5"
with jdem.H5Writer(directory=h5_directory) as writer:
    writer.save(state, system)

state_h5, system_h5 = jdem.utils.h5.load(str(h5_directory / "snapshot_00000000.h5"))
state_h5, system_h5 = jdem.System.step(state_h5, system_h5)
reference_state, reference_system = jdem.System.step(state, system)
np.testing.assert_array_equal(state_h5.pos, reference_state.pos)
assert int(system_h5.step_count) == int(reference_system.step_count)
print("HDF5 continuation step:", int(system_h5.step_count))

tmp_root.cleanup()
