"""
Introduction
---------------------

Lets look at the absolute simplest simulation you can run using JaxDEM.

Let's get started!
"""

import jaxdem as jdem

# %%
# Initialize the Simulation State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First, we create a :py:class:`jaxdem.state.State` object. The state object holds all the information of the objects inside the simulation.
# Lets create a single particle at the origin (0, 0, 0). By default, :py:meth:`jaxdem.state.State.create` will set default values for all non-specified attributes.

state = jdem.State.create(pos=[[0.0, 0.0, 0.0]])

# %%
# Note that we used `pos=[[coords]]` not `pos=[coords]`. This is because :py:meth:`jaxdem.state.State.create` expects a list of coordinates.
# Therefore, we must pass a list of lists, even for a single particle. Internally, the coordinates list will be converted to a Jax array.

# %%
# Initialize the Simulation System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we define our :py:class:`jaxdem.system.System`. This object holds all the global parameters
# and configuration of our simulation. In the same way as `state`, :py:meth:`jaxdem.system.System.create` will
# use default values for anything we don't specify. The only requirement is that the dimension of the
# simulation matches the dimension of the state:

system = jdem.System.create(state.shape)

# %%
# Run the Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, we drive the simulation by calling :py:class:`jaxdem.system.System.step`. We'll advance
# the simulation for `n_steps` time steps:

n_steps = 10
state, system = system.step(state, system, n=n_steps)


# %%
# Saving the Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The last thing left to do is to save the simulation to VTK files so we can
# visualize it in ParaView:

writer = jdem.VTKWriter(directory="/tmp/data")
writer.save(state, system)
