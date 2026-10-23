"""Introduction
---------------------

Create, initialize, and advance a single particle, then optionally export it.
The dynamics use only the core installation; the final VTK section requires
``JaxDEM[io]``. The ten-step calculation is deliberately small.
"""

import tempfile
from pathlib import Path
import numpy as np
import jaxdem as jdem

# %%
# Initialize the Simulation State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First, we create a :py:class:`jaxdem.state.State` object. It holds all
# per-particle data in the simulation. Here we create a single particle at the
# origin with velocity ``(1, 0, 0)``. :py:meth:`jaxdem.state.State.create` fills every unspecified
# attribute with a default value.

state = jdem.State.create(pos=[[0.0, 0.0, 0.0]], vel=[[1.0, 0.0, 0.0]])

# %%
# Note that we wrote ``pos=[[coords]]``, not ``pos=[coords]``.
# :py:meth:`jaxdem.state.State.create` expects a 2-D array of shape
# ``(N, dim)``, so even a single particle needs a list of lists.

# %%
# Initialize the Simulation System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we create a :py:class:`jaxdem.system.System`. It holds the global
# configuration of the simulation (domain, force model, integrator, …).
# Like the state, :py:meth:`jaxdem.system.System.create` fills in
# defaults for anything we do not specify. The only requirement is that the
# system dimension matches the state dimension:

system = jdem.System.create(state.shape, dt=0.01)
state, system = jdem.System.initialize(state, system)

# %%
# Run the Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# After explicit initialization has prepared the initial acceleration, we
# advance the simulation by calling
# :py:meth:`jaxdem.system.System.step`. Initial forces are prepared by
# :py:meth:`~jaxdem.system.System.initialize`:

n_steps = 10
state, system = system.step(state, system, n=n_steps)
print(f"Advanced {n_steps} steps to time {float(system.time):.3g}")
print("Particle position:", state.pos)
# With no other particle or external force, x(t) = x(0) + v_x * t.
np.testing.assert_allclose(state.pos, [[0.1, 0.0, 0.0]], atol=2e-7)


# %%
# Saving the Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# With the optional ``[io]`` extra installed, save the simulation to VTK so we can
# visualize it in ParaView with :py:class:`~jaxdem.writers.vtk_writer.VTKWriter`.
# Its :py:meth:`~jaxdem.writers.vtk_writer.VTKWriter.save` method writes asynchronously in background
# threads. Use the writer as a context manager (or call
# :py:meth:`writer.block_until_ready() <jaxdem.writers.async_base.BaseAsyncWriter.block_until_ready>` before
# exiting) to make sure the writer
# finishes all files:

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
# ~~~~~~~~~~~~~~~~~~
# Now that you have seen the simplest simulation, explore the rest of the
# user guide to learn about each component in depth:
#
# - :doc:`../auto_examples/state_guide` — particle data, fixed particles, identifiers.
# - :doc:`../auto_examples/system_guide` — system configuration, deactivating modules, batched simulations.
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
