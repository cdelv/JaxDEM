# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK Post-Processing and Visualization
-----------------------------------------

This guide introduces the JaxDEM VTK writing utilities:

- :py:class:`~jaxdem.writers.vtk_writer.VTKWriter`
- ``"spheres"``: :py:class:`~jaxdem.writers.vtk_spheres_writer.VTKSpheresWriter`
- ``"facet_spheres"``: :py:class:`~jaxdem.writers.vtk_spheres_writer.VTKFacetSpheresWriter`
- ``"domain"``: :py:class:`~jaxdem.writers.vtk_domain_writer.VTKDomainWriter`
- ``"facets"``: :py:class:`~jaxdem.writers.vtk_facets_writer.VTKFacetsWriter`
- ``"deformable_elements"``: :py:class:`~jaxdem.writers.vtk_deformable_particle_writer.VTKDeformableElementsWriter`
- ``"deformable_edge_adjacencies"``: :py:class:`~jaxdem.writers.vtk_deformable_particle_writer.VTKDeformableEdgeAdjacenciesWriter`
- ``"deformable_edges"``: :py:class:`~jaxdem.writers.vtk_deformable_particle_writer.VTKDeformableEdgesWriter`

:py:class:`~jaxdem.writers.vtk_writer.VTKWriter` exports particle positions, boundaries, meshes, and state fields
to standard VTK XML files (.vtp) and ParaView manifest collection files (.pvd).
Install ``JaxDEM[io]``. This small 2D scene is also viewable in ParaView; the
output directory is kept after execution so you can open the files.
"""

import tempfile
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import jaxdem as jdem

# %%
# Basic Usage of :py:class:`~jaxdem.writers.vtk_writer.VTKWriter`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.writers.vtk_writer.VTKWriter` inherits from
# :py:class:`~jaxdem.writers.async_base.BaseAsyncWriter`, which manages a pool of background
# threads for parallel, non-blocking disk I/O. Use it in a context manager
# (``with ... as ...:``) to make sure all writes finish before exiting.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
    rad=jnp.array([0.2, 0.2]),
    vel=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
    mass=jnp.ones(2),
)
system = jdem.System.create(
    state=state,
    dt=1e-2,
    rotation_integrator_type=None,
    collider_type="naive",
    mat_table=jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=20.0, poisson=0.25, density=1.0)]
    ),
)
state, system = jdem.System.initialize(state, system)

# Create a unique directory, preserving its few frames for inspection.
# Replace this with Path("my_output") when choosing a permanent destination.
tmp_dir = Path(tempfile.mkdtemp(prefix="jaxdem-vtk-guide-"))

# %%
# We configure the writer to save to our directory. By default, all registered
# writers are active (each one skips itself when its data is not present).
# :py:meth:`~jaxdem.writers.vtk_writer.VTKWriter.save` submits each snapshot to the queue.
# :py:class:`~jaxdem.writers.vtk_writer.VTKWriter` defaults to ``clean=False``. Here ``clean=True``
# explicitly erases the frames directory
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
# `jax.lax.scan <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>`_ or batching), you can write
# the entire sequence in one call
# through :py:meth:`~jaxdem.writers.vtk_writer.VTKWriter.save` by setting
# ``trajectory=True`` and specifying the ``trajectory_axis``. Collect trajectories
# with :py:meth:`~jaxdem.system.System.trajectory_rollout`.

# Collect matching State/System frames of shape (T, N, dim). Each System
# frame carries its own time and step count for the VTK files and manifest.
# Forty intervals of five steps cover two reduced time units: the grains approach,
# compress their contact springs, and separate with reversed velocities.
# A leading zero stride retains the current initialized snapshot as well.
# The modest stiffness and dt resolve the impact while keeping this run tiny.
state, system, (state_trajectory, system_trajectory) = system.trajectory_rollout(
    state, system, strides=jnp.asarray([0] + [5] * 40)
)
scene_states, scene_systems = state_trajectory, system_trajectory
assert state_trajectory.pos.shape[0] == 41
assert float(state_trajectory.vel[-1, 0, 0]) < -0.4
assert float(state_trajectory.vel[-1, 1, 0]) > 0.4
assert np.ptp(np.asarray(state_trajectory.pos[:, 0, 0])) > 0.25
system.check_overflow()

# Save the entire trajectory
with jdem.VTKWriter(directory=tmp_dir / "trajectory", clean=True) as writer:
    writer.save(state_trajectory, system_trajectory, trajectory=True, trajectory_axis=0)
    print("\nSaved a trajectory with shape:", state_trajectory.pos.shape)

# The directory is intentionally retained. Remove it once you finish
# inspecting the PVD collections and their associated frame files.
print("Output retained at:", tmp_dir)

# %%
# Render the saved output with ParaView
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The ``spheres`` collection contains particle centers and a ``rad`` field,
# not triangulated spheres. In ParaView, apply a **Glyph** filter with a sphere
# source, select ``rad`` for scaling, set the scale factor to **2**, and show
# all points. The default glyph sphere has radius 0.5, so this reproduces each
# particle's physical radius. Load the other collections to display the domain,
# facets and deformable elements alongside the grains.
# Bonded deformable meshes retain their coherent, unwrapped coordinates;
# they can extend beyond a periodic cell without their bonds being torn by
# independently wrapping each node. Ordinary grains are wrapped for display.
#
# The same operations can run headlessly in ParaView's ``pvpython`` interpreter.
# This is a separate program from the Python interpreter running JaxDEM.
# Run it from the saved output directory, or use an absolute PVD filename:
#
# .. code-block:: python
#
#    from paraview.simple import (
#        PVDReader, Glyph, CreateRenderView, Show, ColorBy,
#        Render, SaveScreenshot, ExportView,
#    )
#
#    particles = PVDReader(FileName="batch_00000000_spheres.pvd")
#    spheres = Glyph(Input=particles, GlyphType="Sphere")
#    spheres.ScaleArray = ["POINTS", "rad"]
#    spheres.ScaleFactor = 2.0
#    spheres.GlyphMode = "All Points"
#    view = CreateRenderView()
#    display = Show(spheres, view)
#    ColorBy(display, ("POINTS", "vel", "Magnitude"))
#    view.ResetCamera()
#    Render(view)
#    SaveScreenshot("particles.png", view, ImageResolution=[960, 600])
#    ExportView(
#        "particles.vtkjs", view=view,
#        ExportToGlance=1, AutomaticGlanceHTML=1, DisableNetwork=1,
#    )
#
# ParaView can save a sequence of rendered frames for a short movie. The
# documentation's media pipeline reads this run's PVD collections and selects
# saved timesteps; it does not redraw particles with Matplotlib or invent
# intermediate configurations. Matplotlib remains useful for diagnostics such
# as energy, pressure and convergence.
#
# `ParaView Glance <https://kitware.github.io/glance/doc/loading_files.html>`_
# embeds the exported snapshot in a browser viewer, so readers can rotate,
# zoom, inspect fields and change representations without a ParaView server.
# A movie shows the time evolution, while an interactive snapshot supports
# examining the geometry. See ParaView's `saving results documentation
# <https://docs.paraview.org/en/v6.2.0/UsersGuide/savingResults.html>`_.
