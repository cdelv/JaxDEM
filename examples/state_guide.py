"""
The Simulation State
----------------------------------------

This example focuses on the :py:class:`jaxdem.state.State` object,
a core component of JaxDEM that holds all information about the particles
in a simulation.

JaxDEM stores particle data using a Structure-of-Arrays (`SoA <https://en.wikipedia.org/wiki/AoS_and_SoA>`_) architecture,
making it efficient for JAX's vectorized and parallel computations.
This approach also simplifies handling trajectories and batched simulations
without requiring complex code modifications.

Let's explore how to create, modify, and extend the simulation state effectively.
"""

# %%
# State Creation
# ~~~~~~~~~~~~~~~~~~~~~
# We'll start by creating a simple 2D state representing a single particle
# located at the origin. By default, :py:meth:`jaxdem.state.State.create`
# initializes non-specified attributes (like velocity, radius, mass) with
# sensible default values.

import jax
import jaxdem as jdem
import jax.numpy as jnp

state = jdem.State.create(pos=[[0.0, 0.0]])
print(f"Dimension of state: {state.dim}")
print(f"Initial position: {state.pos}")

# %%
# To create a 3D state, simply pass a 3D coordinate list. JaxDEM
# automatically infers the simulation dimension from the position data.
# The library is designed for flexibility across dimensions, but a check
# ensures the state is explicitly 2D or 3D.

state = jdem.State.create(pos=[[0.0, 0.0, 0.0]])
print(f"Dimension of state: {state.dim}")
print(f"Initial position: {state.pos}")

# %%
# Modifying State Attributes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We have two primary ways to set or modify particle attributes:
#
# 1.  **Direct assignment:** You can assign new JAX arrays
#     to attributes like `state.vel` via the replace interface. This is flexible but requires you
#     to ensure shape consistency.

from dataclasses import replace

state = replace(state, vel=jnp.ones_like(state.pos))
print(state.vel)

# %%
# Note that because we are dealing with JAX arrays, doing something like
#
# .. code-block:: python
#
#     state.vel[i] = jnp.asarray([1, 2, 3], dtype=float)
#
# will result in an error. The correct way of doing this is

i = 0
state = replace(state, vel=state.vel.at[i].set(jnp.asarray([1, 2, 3], dtype=float)))
print(state.vel)

# %%
# However, this is not efficient and is not recommended. Always try to use vectorized operations.

# %%
# 2.  **Constructor arguments:** This is generally the
#     safer approach, as the :py:meth:`jaxdem.state.State.create`
#     constructor automatically validates shapes and types, ensuring
#     consistency across all attributes.

state = jdem.State.create(pos=jnp.zeros((1, 2)), vel=jnp.ones((1, 2)))
print(state.vel)

# %%
# Extending the State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Working directly with `SoA <https://en.wikipedia.org/wiki/AoS_and_SoA>`_ structures can sometimes feel less intuitive
# than Array-of-Structures (AoS) for adding and modifying individual particles. To simplify
# this, JaxDEM provides utility methods like :py:meth:`jaxdem.state.State.add`.
#
# :py:meth:`jaxdem.state.State.add` allows you to append new particles to an
# existing state, automatically assigning unique IDs and checking for dimension
# consistency.

state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([0.5]))
print(f"Initial state (N={state.N}, IDs={state.ID}):\npos={state.pos}")

state = jdem.State.add(
    state,
    pos=jnp.array([[1.0, 1.0]]),
    vel=2 * jnp.ones((1, 2)),
    rad=10 * jnp.ones((1)),
)
print(f"\nState after addition (N={state.N}, IDs={state.ID}):\npos={state.pos}")
print(f"New particle velocity: {state.vel[-1]}")
print(f"New particle radius: {state.rad[-1]}")


# %%
# You can also add multiple particles at once by providing arrays of the
# appropriate shape. :py:meth:`jaxdem.state.State.add` will ensure the dimensions
# of the new particles match the existing state.

state = jdem.State.add(
    state,
    pos=jnp.array([[2.0, 0.0], [0.0, 2.0]]),
    vel=jnp.zeros((2, 2)),
    rad=jnp.array([0.8, 0.3]),
    ID=jnp.array([2, 3]),
)
print(
    f"\nState after adding multiple particles (N={state.N}, IDs={state.ID}):\n{state.pos}"
)

# %%
# Note that we provided particle IDs here. :py:meth:`jaxdem.state.State.add` will add `jnp.max(state.ID)` to the
# provided IDs to ensure no overlaps. However, we don't ensure the IDs are a continuous sequence.
# However, the default is to use sequential IDs in the constructor.

# %%
# Merging Two States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For combining two `State` objects,
# you can use :py:meth:`jaxdem.state.State.merge`. This method concatenates
# the particles from the second state onto the first. This is useful for
# assembling complex initial configurations from smaller parts.

state_a = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
)
state_b = jdem.State.create(
    jnp.array([[2.0, 2.0], [3.0, 3.0], [5.0, 2.0]]),
)
state = jdem.State.merge(state_a, state_b)

print(f"State A (N={state_a.N}, IDs={state_a.ID}):\npos={state_a.pos}")
print(f"State B (N={state_b.N}, IDs={state_b.ID}):\npos={state_b.pos}")
print(f"Merged state (N={state.N}, IDs={state.ID}):\npos={state.pos}")


# %%
# Stacking States for Trajectories or Batches
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One of the features that makes JaxDEM special is its ability
# to handle **batched states**. Batches can be interpreted as trajectories
# (multiple snapshots over time) or as independent simulations
# (multiple distinct initial conditions).
#
# This capability is handy for performance. JaxDEM is optimized
# for **throughput**, meaning that if your GPU is not saturated, you're
# leaving performance on the table. A common task in DEM simulations is to
# perform parameter sweeps. JaxDEM provides the tools to run many independent
# simulations in parallel, potentially completing many small simulations in
# at the same time it would take for just one until your GPU is fully utilized.
#
# Furthermore, JaxDEM's ability to handle trajectories means you don't have
# to interrupt the GPU to perform I/O operations (for example, saving the
# simulation state). You can accumulate an entire trajectory in memory and then
# save everything at the end. This often results in much better
# performance at the cost of a bit more memory usage.
#
# To manage simulation trajectories or perform batched simulations,
# :py:meth:`jaxdem.state.State.stack` is available. It takes a sequence of
# :py:class:`jaxdem.state.State` snapshots and concatenates them along a new
# leading axis. This creates a multi-dimensional state where the first axis
# can represent time steps, batch elements, or other high-level groupings.
# Note that stacking does *not* shift particle IDs, as it assumes the
# particles are the same entities across the stacked dimension.
# :py:meth:`jaxdem.state.State.stack` makes sure shapes are consistent.

snapshot1 = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=[2.0])
snapshot2 = jdem.State.create(pos=jnp.array([[0.1, 0.0]]), vel=jnp.array([[0.1, 0.0]]))
snapshot3 = jdem.State.create(pos=jnp.array([[0.2, 0.0]]), mass=[3.3])

batched_state = jdem.State.stack([snapshot1, snapshot2, snapshot3])

print(f"Shape of stacked positions (B, N, dim): {batched_state.pos.shape}")
print(f"Batch size: {batched_state.batch_size}")

# %%
# Another way of creating batch states is using Jax's vmap:

batched_state = jax.vmap(
    lambda i: jdem.State.create(
        i
        * jnp.ones(
            (1, 2),
        )
    )
)(jnp.arange(4))
print(f"Shape of stacked positions (B, N, dim): {batched_state.pos.shape}")
print(f"Batch size: {batched_state.batch_size}")
print(f"Position at batch 0: {batched_state.pos[0]}")
print(f"Position at batch 1: {batched_state.pos[1]}")
print(f"Position at batch 2: {batched_state.pos[2]}")


# %%
# A more realistic way in which you could encounter a batched state is the following:


def initialize(i):
    state = jdem.State.create(i * jnp.ones((4, 2)))
    system = jdem.System.create(state.dim)
    return state, system


N_batches = 10
state, system = jax.vmap(initialize)(jnp.arange(N_batches))

# %%
# Then, to tun this simulation:

state, system = jax.vmap(system.step, in_axes=(0, 0, None))(state, system, 10)
print(f"Shape of positions (B, N, dim): {state.pos.shape}")


# %%
# Its possible to run the simulation using a single system instance for all bathes:

system = jdem.System.create(state.dim, domain_type="reflect")
state, system = jax.vmap(system.step, in_axes=(0, None, None))(state, system, 10)
print(f"Shape of positions (B, N, dim): {state.pos.shape}")

# %%
# However note that system can change over time. If this is the case, each state should have its own system.


# %%
# Trajectories of Batches
# ~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM's state handling capabilities extend beyond just batches or single trajectories.
# We can also accumulate **trajectories of batched states**.
#
# This feature is handy for scenarios like **parameter sweeps**,
# where you're running multiple independent simulations (a batch) and want to
# capture their full time evolution (a trajectory) without frequent I/O operations.
# It allows for highly efficient data collection.
#
# Moreover, :py:meth:`jaxdem.writer.VTKWriter.save` is designed to intelligently
# handle these multi-dimensional states. It understands the structure
# of states with multiple leading dimensions.
#
# By convention, when dealing with `State.pos` of shape `(..., N, dim)`:
#
# *   The **first leading dimension** (index=0) is typically interpreted as a **batch** dimension.
# *   Any **subsequent leading dimensions** are interpreted as **trajectory** dimensions.
#
# For instance, a `State.pos` of shape `(B, T, N, dim)` would represent `B`
# independent batches containing a `T`-step trajectory of `N` particles.
#
# If a `State` object with more than four dimensions (`pos.ndim > 4`) is passed to
# :py:meth:`jaxdem.writer.VTKWriter.save`, all leading dimensions from index 0
# up to `pos.ndim - 2` are flattened and treated as a trajectory of batched simulation.
# (B, t_1, ..., t_k, N, dim) -> (B, T, N, dim).
# If trajectory = True, B is also trated as a trajectory dimension and also flattened.

batched_state = jdem.State.stack([batched_state, batched_state, batched_state])
print(f"Shape of stacked positions (T, B, N, dim): {batched_state.pos.shape}")
print(f"Batch size: {batched_state.batch_size}")

# %%
# Following the example of the previos section. You might encounter a trajectory of batches in the following way:

N_batches = 9
state, system = jax.vmap(initialize)(jnp.arange(N_batches))

state, system, (state_traj, system_traj) = jax.vmap(
    system.trajectory_rollout, in_axes=(0, 0, None)
)(state, system, 10)

print(f"Shape of positions (B, T, N, dim): {state_traj.pos.shape}")


# %%
# Utilities
# ~~~~~~~~~~
# To improve the ease of setting up simulations, JaxDEM includes some
# utility methods in :py:mod:`jaxdem.utils` to initialize states and more. For example,
# we can create a state of N particles with all their attributes random:

from jaxdem import utils as utils

state = utils.random_state(dim=3, N=10)
print(state)
