# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK Post-Processing and Visualization
-----------------------------------------

This guide introduces the JaxDEM VTK writing utilities:

- :py:class:`~jaxdem.writers.VTKWriter`
- The registered concrete writers: ``"spheres"``, ``"domain"``, and ``"deformable_particles"``

VTKWriter is used to export particle positions, boundaries, meshes, and state fields
to standard VTK XML files (.vtp) and ParaView manifest collection files (.pvd) for 3D visualization.
"""

import tempfile
import shutil
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxdem as jdem

# %%
# Basic Usage of VTKWriter
# ~~~~~~~~~~~~~~~~~~~~~~~~
# VTKWriter inherits from BaseAsyncWriter, which manages a pool of background
# threads for parallel, non-blocking disk I/O. Using it within a context manager
# (``with ... as ...:``) ensures all writes finish before exiting.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
)
system = jdem.System.create(state.shape, dt=1e-2)

# Create a temporary directory for the VTK frames
tmp_dir = Path(tempfile.gettempdir()) / "vtk_output"

# We configure the writer to save to our directory.
# By default, all registered writers ("spheres", "domain", "deformable_particles") are active.
with jdem.VTKWriter(directory=tmp_dir, clean=True) as writer:
    # Save step 0
    writer.save(state, system)
    print("Saved step 0")

    # Step the simulation and save again
    state, system = system.step(state, system, n=5)
    writer.save(state, system)
    print("Saved step 5")

# %%
# Opening the data in ParaView
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The writer creates:
#
# - Individual ``.vtp`` (XML PolyData) files containing coordinates and fields under a batch subdirectory.
# - A ``.pvd`` manifest file (e.g. ``batch_00000000_spheres.pvd``) in the root output folder.
#
# To visualize the simulation time series in ParaView, simply open the ``.pvd`` file.
# ParaView will automatically resolve the timesteps and load the corresponding ``.vtp`` files.

# Check the generated files in our directory
pvd_files = list(tmp_dir.glob("*.pvd"))
print("Generated PVD collections:")
for f in pvd_files:
    print(f" - {f.name}")

vtp_files = list((tmp_dir / "batch_00000000").glob("*.vtp"))
print("\nGenerated VTP frames (first few):")
for f in sorted(vtp_files)[:4]:
    print(f" - batch_00000000/{f.name}")

# %%
# Saving Trajectories directly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you have accumulated a trajectory of states and systems in memory (e.g., via
# `jax.lax.scan` or batching), you can write the entire sequence in one call
# by setting ``trajectory=True`` and specifying the ``trajectory_axis``.

# Let's mock a trajectory of shape (T, N, dim)
T_steps = 3
pos_trajectory = jnp.stack([state.pos + i * 0.1 for i in range(T_steps)], axis=0)

# Create a batched/trajectory State using jax.vmap to ensure all internal
# state fields (like unique_id, fixed, etc.) are correctly batched.
state_trajectory = jax.vmap(lambda p: jdem.State.create(pos=p, rad=state.rad))(
    pos_trajectory
)

# Broadcast the system to match the trajectory shape
system_trajectory = jax.tree.map(
    lambda x: jnp.broadcast_to(x, (T_steps, *x.shape)), system
)

# Save the entire trajectory
with jdem.VTKWriter(directory=tmp_dir / "trajectory", clean=True) as writer:
    writer.save(state_trajectory, system_trajectory, trajectory=True, trajectory_axis=0)
    print("\nSaved a trajectory with shape:", state_trajectory.pos.shape)

# Clean up
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
