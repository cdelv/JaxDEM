"""Introduction.
---------------------

Let's look at the simplest simulation you can run with JaxDEM.
"""

import jaxdem as jdem

# %%
# Initialize the Simulation State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First, we create a :py:class:`jaxdem.state.State` object. It holds all
# per-particle data in the simulation. Let's create a single particle at the
# origin. By default, :py:meth:`jaxdem.state.State.create` fills every
# unspecified attribute with a sensible default value.

state = jdem.State.create(pos=[[0.0, 0.0, 0.0]])

# %%
# Note that we wrote ``pos=[[coords]]``, not ``pos=[coords]``.
# :py:meth:`jaxdem.state.State.create` expects a 2-D array of shape
# ``(N, dim)``, so even a single particle needs a list of lists.

# %%
# Initialize the Simulation System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we create a :py:class:`jaxdem.system.System`. It holds the global
# configuration of the simulation (domain, force model, integrator, …).
# Just like the state, :py:meth:`jaxdem.system.System.create` fills in
# defaults for anything we don't specify. The only requirement is that the
# system dimension matches the state dimension:

system = jdem.System.create(state.shape)

# %%
# Run the Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, we advance the simulation by calling
# :py:meth:`jaxdem.system.System.step`:

n_steps = 10
state, system = system.step(state, system, n=n_steps)


# %%
# Saving the Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The last thing left to do is to save the simulation to VTK files so we can
# visualize it in ParaView:

writer = jdem.VTKWriter(directory="/tmp/data")
writer.save(state, system)

# %%
# Where to Go Next
# ~~~~~~~~~~~~~~~~~~
# Now that you have seen the simplest simulation, explore the rest of the
# user guide to learn about each component in depth:
#
# - :doc:`../auto_examples/state_guide` — particle data, fixed particles, identifiers.
# - :doc:`../auto_examples/system_guide` — system configuration, deactivating modules, batched simulations.
# - :doc:`../auto_examples/domain_guide` — boundary conditions (free, periodic, reflective).
# - :doc:`../auto_examples/integrator_guide` — time integration and energy minimisation.
# - :doc:`../auto_examples/materials_guide` — material definitions and matchmakers.
# - :doc:`../auto_examples/force_model_guide` — pairwise force laws and species-wise routing.
# - :doc:`../auto_examples/force_manager_guide` — gravity, external forces, custom functions.
# - :doc:`../auto_examples/collider_guide` — contact detection algorithms.
# - :doc:`../auto_examples/clump_guide` — rigid bodies from multiple spheres.
