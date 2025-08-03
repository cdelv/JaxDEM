"""
Introduction
====================================

This example showcases the absolute simplest simulation you can run using JaxDEM.

Let's get started!
"""

import jaxdem as jdem

# %%
# Initialize the Simulation State
# -------------------------------
# First, we create a :py:class:`jaxdem.State` object. For this example, we define
# a single particle. By default, :py:meth:`jaxdem.State.create` will set default values for all non-specified attributes.
# We set the initial position at the origin (0, 0, 0).

state = jdem.State.create(pos=[[0.0, 0.0, 0.0]])

# %%
# Note that `pos=[[coords]]` not `pos=[coords]`. This is because :py:meth:`jaxdem.State.create` expects a list of coordinates.
# Therefore, we must pass a list of lists, even for a single particle.

# %%
# Initialize the Simulation System
# -------------------------------
# Next, we define our :py:class:`jaxdem.System`. This object holds all the global parameters
# and configuration of our simulation. In the same way as statem, :py:meth:`jaxdem.System.create` will
# use default values for anything we don't specify. The only requirement is that the dimension of the
# simulation matches the dimension of the state:

system = jdem.System.create(dim=state.dim)

# %%
# Run the Simulation
# ------------------
# Finally, we simulate by calling :py:class:`jdem.System.step`. We'll advance
# the simulation for `n_steps` time steps:

n_steps = 10
state, system = system.step(state, system, n=n_steps)
