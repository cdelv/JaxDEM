# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK Post-Processing and Visualization
----------------------------------------

Export particle positions, domain boxes, facets, and deformable meshes to VTK
XML (``.vtp``) and ParaView collection (``.pvd``) files using
:py:class:`~jaxdem.writers.vtk_writer.VTKWriter` (requires ``JaxDEM[io]``).

Available sub-writers (all enabled by default; each skips itself when its data
is absent):

- ``"spheres"`` (:py:class:`~jaxdem.writers.vtk_spheres_writer.VTKSpheresWriter`)
- ``"facet_spheres"`` (:py:class:`~jaxdem.writers.vtk_spheres_writer.VTKFacetSpheresWriter`)
- ``"domain"`` (:py:class:`~jaxdem.writers.vtk_domain_writer.VTKDomainWriter`)
- ``"facets"`` (:py:class:`~jaxdem.writers.vtk_facets_writer.VTKFacetsWriter`)
- ``"deformable_elements"`` (:py:class:`~jaxdem.writers.vtk_deformable_particle_writer.VTKDeformableElementsWriter`)
- ``"deformable_edges"`` (:py:class:`~jaxdem.writers.vtk_deformable_particle_writer.VTKDeformableEdgesWriter`)
- ``"deformable_edge_adjacencies"`` (:py:class:`~jaxdem.writers.vtk_deformable_particle_writer.VTKDeformableEdgeAdjacenciesWriter`)
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp

import jaxdem as jdem

# %%
# Saving Individual Snapshots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.writers.vtk_writer.VTKWriter` writes asynchronously in
# background threads. Use it as a context manager so all frames flush before
# exit. Pass ``writers=["spheres", "domain"]`` to restrict which collections
# are written, and ``clean=True`` to clear an existing output folder first:

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

tmp_dir = Path(tempfile.mkdtemp(prefix="jaxdem-vtk-guide-"))
with jdem.VTKWriter(directory=tmp_dir, writers=["spheres", "domain"], clean=True) as writer:
    writer.save(state, system)  # step 0
    state, system = jdem.System.step(state, system, n=5)
    writer.save(state, system)  # step 5

# %%
# Output Layout and Batches
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Each batch element ``b`` gets its own subdirectory ``batch_0000000b/`` of
# ``.vtp`` frames and a top-level ``batch_0000000b_<writer>.pvd`` manifest
# (single simulations use ``batch_00000000``). Open the ``.pvd`` file in
# ParaView to load the full time series. Sphere frames automatically export
# point arrays such as ``rad``, ``vel``, ``force``, ``mass``, ``clump_id``,
# ``mat_id``, and ``species_id``:

print("PVD collections:", [p.name for p in sorted(tmp_dir.glob("*.pvd"))])
print("VTP frames:", [p.name for p in sorted((tmp_dir / "batch_00000000").glob("*.vtp"))[:4]])

# %%
# Saving In-Memory Trajectories
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect a trajectory with :py:meth:`~jaxdem.system.System.trajectory_rollout`
# and write all frames in one call using ``trajectory=True`` (``trajectory_axis``
# defaults to ``0``; any remaining leading axes are written as separate
# ``batch_XXXXXXXX`` series):

state, system, (state_trajectory, system_trajectory) = jdem.System.trajectory_rollout(
    state, system, strides=jnp.asarray([0] + [5] * 40)
)
system.check_overflow()

with jdem.VTKWriter(directory=tmp_dir / "trajectory", clean=True) as writer:
    writer.save(state_trajectory, system_trajectory, trajectory=True, trajectory_axis=0)
print("Saved trajectory shape:", state_trajectory.pos.shape, "| dir:", tmp_dir)

# %%
# Rendering in ParaView and ``pvpython``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``spheres`` files store particle centers and a ``rad`` point array (not
# triangulated spheres). In ParaView, add a **Glyph** filter with a Sphere
# source, scale by ``rad``, set **Scale Factor = 2.0** (ParaView's default
# sphere has radius 0.5), and set **Glyph Mode = All Points**. Ordinary grains
# are wrapped into periodic boxes for display, while bonded deformable meshes
# stay unwrapped so elements are never torn across boundaries.
#
# To render headlessly with ParaView's ``pvpython``:
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
# Exported ``.vtkjs`` scenes can be opened directly in `ParaView Glance
# <https://kitware.github.io/glance/doc/loading_files.html>`_ for interactive
# 3D inspection in a browser.
