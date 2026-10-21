# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK Post-Processing and Visualization
-----------------------------------------

This guide introduces the JaxDEM VTK writing utilities:

- :py:class:`~jaxdem.writers.VTKWriter`
- The registered concrete writers: ``"spheres"``, ``"facet_spheres"``, ``"domain"``,
  ``"facets"``, ``"deformable_elements"``, ``"deformable_edge_adjacencies"``, and
  ``"deformable_edges"``

VTKWriter exports particle positions, boundaries, meshes, and state fields
to standard VTK XML files (.vtp) and ParaView manifest collection files (.pvd).
Install ``JaxDEM[io]``. This small 2D scene is also viewable in ParaView; the
output directory is kept after execution so you can open the files.
"""

import tempfile
from pathlib import Path
import jax.numpy as jnp
import jaxdem as jdem

# %%
# Basic Usage of VTKWriter
# ~~~~~~~~~~~~~~~~~~~~~~~~
# VTKWriter inherits from BaseAsyncWriter, which manages a pool of background
# threads for parallel, non-blocking disk I/O. Use it in a context manager
# (``with ... as ...:``) to make sure all writes finish before exiting.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
    rad=jnp.array([0.2, 0.2]),
    vel=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
)
system = jdem.System.create(state.shape, dt=1e-2)
state, system = jdem.System.initialize(state, system)

# Create a unique directory, preserving its few frames for inspection.
# Replace this with Path("my_output") when choosing a permanent destination.
tmp_dir = Path(tempfile.mkdtemp(prefix="jaxdem-vtk-guide-"))

# We configure the writer to save to our directory. By default, all registered
# writers are active (each one skips itself when its data is not present).
# VTKWriter defaults to clean=False. Here clean=True explicitly erases the frames directory
# on construction, with safety checks on the path. The checkpoint writers
# default to clean=False and preserve existing data.
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
# To visualize the simulation time series in ParaView, open the ``.pvd`` file.
# ParaView resolves the timesteps and loads the corresponding ``.vtp`` files.

# Check the generated files in our directory
pvd_files = list(tmp_dir.glob("*.pvd"))
assert pvd_files
print("Generated PVD collections (open one in ParaView):")
for f in pvd_files:
    print(f" - {f}")

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

# Collect matching State/System frames of shape (T, N, dim). Each System
# frame carries its own time and step count for the VTK files and manifest.
state, system, (state_trajectory, system_trajectory) = system.trajectory_rollout(
    state, system, n=3
)

# Save the entire trajectory
with jdem.VTKWriter(directory=tmp_dir / "trajectory", clean=True) as writer:
    writer.save(state_trajectory, system_trajectory, trajectory=True, trajectory_axis=0)
    print("\nSaved a trajectory with shape:", state_trajectory.pos.shape)

# The directory is intentionally retained. Remove it once you finish
# inspecting the PVD collections and their associated frame files.
print("Output retained at:", tmp_dir)
