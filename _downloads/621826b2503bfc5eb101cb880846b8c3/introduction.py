# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Introduction
---------------

Create, initialize, and advance a single particle, then optionally export it
for ParaView. The dynamics use only the core installation; the final VTK
export requires ``JaxDEM[io]``.
"""

import tempfile
from pathlib import Path

import jaxdem as jdem

# %%
# Initialize the Simulation State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First, create a :py:class:`jaxdem.state.State` holding all per-particle data.
# Here we place one particle at the origin with velocity ``(1, 0, 0)``.
# :py:meth:`jaxdem.state.State.create` fills every unspecified attribute
# (radius = 1, mass = 1, etc.) with defaults.

state = jdem.State.create(pos=[[0.0, 0.0, 0.0]], vel=[[1.0, 0.0, 0.0]])

# %%
# Notice ``pos=[[coords]]``, not ``pos=[coords]``:
# :py:meth:`jaxdem.state.State.create` expects shape ``(N, dim)``, so even a
# single particle uses a 2-D array.

# %%
# Initialize the Simulation System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, create a :py:class:`jaxdem.system.System` for the global configuration
# (domain, force model, collider, integrator, and time step). Only the spatial
# dimension needs to match the state.
#
# Always call :py:meth:`jaxdem.system.System.initialize` once before stepping
# so initial forces and spatial-search caches are prepared for the first
# integration kick:

system = jdem.System.create(state.shape, dt=0.01)
state, system = jdem.System.initialize(state, system)

# %%
# Run the Simulation
# ~~~~~~~~~~~~~~~~~~
# Advance the simulation with :py:meth:`jaxdem.system.System.step`. Because
# JaxDEM follows JAX's functional style, every update returns the new
# ``(state, system)`` pair rather than mutating compiled arrays in place:

n_steps = 10
state, system = jdem.System.step(state, system, n=n_steps)
print(f"Advanced {n_steps} steps to time {float(system.time):.3g}")
print("Particle position:", state.pos)


# %%
# Saving the Simulation
# ~~~~~~~~~~~~~~~~~~~~~
# With ``JaxDEM[io]`` installed, export frames to VTK for ParaView using
# :py:class:`~jaxdem.writers.vtk_writer.VTKWriter`.
# :py:meth:`~jaxdem.writers.vtk_writer.VTKWriter.save` writes asynchronously in
# background threads; using the writer as a context manager (or calling
# :py:meth:`writer.block_until_ready() <jaxdem.writers.async_base.BaseAsyncWriter.block_until_ready>`)
# waits for all files to finish:

tmp_dir = Path(tempfile.mkdtemp(prefix="jaxdem-introduction-"))
try:
    writer = jdem.VTKWriter(directory=tmp_dir)
except ImportError:
    print("Install JaxDEM[io] to run the optional VTK export section.")
else:
    with writer:
        writer.save(state, system)
    print("VTK output for ParaView:", tmp_dir)

# %%
# Where to Go Next
# ~~~~~~~~~~~~~~~~
# Explore the rest of the user guide to learn each component in depth:
#
# - :doc:`../auto_examples/state_guide` — particle data, fixed particles, identifiers.
# - :doc:`../auto_examples/system_guide` — system configuration, deactivating modules, batched simulations.
# - :doc:`../auto_examples/multi_gpu_guide` — sharding batches across multiple GPUs.
# - :doc:`../auto_examples/domain_guide` — free, periodic, shear-periodic, and reflective boundaries.
# - :doc:`../auto_examples/integrator_guide` — time integration and energy minimization.
# - :doc:`../auto_examples/materials_guide` — material definitions and matchmakers.
# - :doc:`../auto_examples/force_model_guide` — pairwise force laws and species-wise routing.
# - :doc:`../auto_examples/force_manager_guide` — gravity, external forces, custom functions.
# - :doc:`../auto_examples/collider_guide` — contact detection algorithms.
# - :doc:`../auto_examples/clump_guide` — rigid bodies from multiple spheres.
# - :doc:`../auto_examples/facets_guide` — flat boundaries from line segments and triangles.
# - :doc:`../auto_examples/deformable_particle_guide` — elastic, deformable bodies from bonded meshes.
# - :doc:`../auto_examples/vtk_writer_guide` — exporting frames for ParaView visualization.
# - :doc:`../auto_examples/checkpoint_guide` — saving and restoring simulations.
# - :doc:`../auto_examples/custom_modules_guide` — registering your own force models, domains, and
#   integrators.
