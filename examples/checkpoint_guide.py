"""
Checkpoint Save and Load
------------------------

This guide introduces JaxDEM checkpointing utilities:

- :py:class:`~jaxdem.writers.CheckpointWriter`
- :py:class:`~jaxdem.writers.CheckpointLoader`

Checkpoints are useful for long simulations, reproducibility, and restarting
from intermediate steps.
"""

from pathlib import Path

import jax.numpy as jnp
import jaxdem as jdem

base_dir = Path("/tmp/jaxdem_docs_checkpoints")
sim_checkpoint_dir = base_dir / "simulation"
bonded_checkpoint_dir = base_dir / "bonded"
base_dir.mkdir(parents=True, exist_ok=True)

print("Simulation checkpoints:", sim_checkpoint_dir)
print("Bonded checkpoints:", bonded_checkpoint_dir)

# %%
# Saving simulation checkpoints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We create a small simulation, run it in chunks, and save snapshots.

state = jdem.State.create(pos=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
system = jdem.System.create(state.shape, dt=1e-3)

with jdem.CheckpointWriter(directory=sim_checkpoint_dir, max_to_keep=2) as writer:
    writer.save(state, system)  # step 0

    state, system = system.step(state, system, n=5)
    writer.save(state, system)  # step 5

    state, system = system.step(state, system, n=5)
    writer.save(state, system)  # step 10


# %%
# Loading latest and specific checkpoints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``load()`` returns ``(state, system)``. The current latest step can be queried
# using :py:meth:`~jaxdem.writers.CheckpointLoader.latest_step`.

with jdem.CheckpointLoader(directory=sim_checkpoint_dir) as loader:
    print("Available steps:", loader.checkpointer.all_steps())
    print("Latest step:", loader.latest_step())

    state_latest, system_latest = loader.load()
    print("Loaded latest step_count:", int(system_latest.step_count))

    state_step_5, system_step_5 = loader.load(step=5)
    print("Loaded step_count=5:", int(system_step_5.step_count))
    print("State shape at step 5:", state_step_5.pos.shape)

# %%
# Resource management note:
# :py:class:`~jaxdem.writers.CheckpointWriter` and :py:class:`~jaxdem.writers.CheckpointLoader`
# can be used either with a context manager (``with ... as ...``) or manually without ``with``.
# Using ``with`` is recommended because it automatically waits for pending
# async writes and closes resources on exit.
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
# Checkpoint saving is asynchronous, so call ``block_until_ready()`` before
# program exit (and before ``close()`` when managing manually) to guarantee
# files are fully written.

# %%
# Bonded-force model checkpointing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Checkpointing also supports systems with bonded force models, such as
# deformable particles.

vertices_2d = jnp.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)
elements_2d = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
adjacency_2d = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

bonded_model = jdem.BondedForceModel.create(
    "deformableparticlemodel",
    vertices=vertices_2d,
    elements=elements_2d,
    edges=elements_2d,
    element_adjacency=adjacency_2d,
    em=1.0,
)

state_bonded = jdem.State.create(pos=vertices_2d)
system_bonded = jdem.System.create(
    state_bonded.shape,
    bonded_force_model=bonded_model,
)

with jdem.CheckpointWriter(directory=bonded_checkpoint_dir) as writer:
    writer.save(state_bonded, system_bonded)
    writer.block_until_ready()

with jdem.CheckpointLoader(directory=bonded_checkpoint_dir) as loader:
    _, system_restored = loader.load()
    print(
        "Restored bonded model:",
        (
            None
            if system_restored.bonded_force_model is None
            else system_restored.bonded_force_model.type_name
        ),
    )
