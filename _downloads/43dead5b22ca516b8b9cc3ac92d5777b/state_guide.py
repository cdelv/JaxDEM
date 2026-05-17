"""The Simulation State.
----------------------------------------

This example focuses on the :py:class:`jaxdem.state.State` object,
a core component of JaxDEM that holds all information about the particles
in a simulation.

JaxDEM stores particle data using a Structure-of-Arrays (`SoA <https://en.wikipedia.org/wiki/AoS_and_SoA>`_)
architecture, making it efficient for JAX's vectorised and parallel
computations. This layout also simplifies handling trajectories and
batched simulations without complex code changes.

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

state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
print(f"Dimension of state: {state.dim}")
print(f"Initial position: {state.pos}")

# %%
# To create a 3D state, simply pass 3D coordinates. JaxDEM infers the
# dimension from the position data (only 2D and 3D are supported).

state = jdem.State.create(pos=jnp.array([[0.0, 0.0, 0.0]]))
print(f"Dimension of state: {state.dim}")
print(f"Initial position: {state.pos}")

# %%
# Understanding Positions: ``pos``, ``pos_c``, and ``pos_p``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# An important detail: ``state.pos`` is **not** a stored field. It is a
# computed property defined as ``pos = pos_c + R(q) @ pos_p``, where
# ``R(q)`` is the rotation given by the particle's quaternion orientation.
#
# The stored fields are:
#
# *   ``pos_c`` — the center-of-mass position of each particle (or clump).
# *   ``pos_p`` — the offset from the center of mass in the **principal
#     (body) frame**. For simple spheres ``pos_p`` is zero, so ``pos == pos_c``.
#
# For **clumps** (rigid bodies made of multiple spheres), every sphere in the
# same clump shares the *same* ``pos_c``, orientation ``q``, velocity ``vel``,
# angular velocity ``ang_vel``, mass, and inertia. The only per-sphere fields
# that differ within a clump are ``pos_p`` (offset in the body frame) and
# ``rad`` (sphere radius). This deliberate duplication allows vectorised
# operations over all spheres without branching on clump membership.

# %%
# Modifying State Attributes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We have two primary ways to set or modify particle attributes:
#
# 1.  **Direct assignment:** You can assign new JAX arrays
#     to attributes like `state.vel`. This is flexible but requires you
#     to ensure shape consistency.

state.vel = jnp.ones_like(state.pos)
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
state.vel = state.vel.at[i].set(jnp.asarray([1, 2, 3], dtype=float))
print(state.vel)

# %%
# However, this is inefficient and not recommended. Always prefer vectorised operations.

# %%
# 2.  **Constructor arguments:** This is generally the
#     safer approach, as the :py:meth:`jaxdem.state.State.create`
#     constructor automatically validates shapes and types, ensuring
#     consistency across all attributes.

state = jdem.State.create(pos=jnp.zeros((1, 2)), vel=jnp.ones((1, 2)))
print(state.vel)

# %%
# Fixed (Immobile) Particles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The boolean field ``state.fixed`` marks particles that should not move.
# The integrator multiplies velocity updates by ``(1 - fixed)``, so
# fixed particles keep zero velocity regardless of the forces acting on
# them. This is useful for walls, obstacles, or boundary particles.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [2.0, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
    fixed=jnp.array([True, False]),
)
print("Fixed mask:", state.fixed)

# %%
# Identifier Fields
# ~~~~~~~~~~~~~~~~~~~
# Each particle carries several integer identifiers:
#
# - ``clump_id`` — groups particles into rigid bodies (see
#   :doc:`../auto_examples/clump_guide`). Particles with the same
#   ``clump_id`` never interact via contact forces and move as one
#   body. By default every particle has a unique ``clump_id``.
# - ``bond_id`` — used for deformable particles (see
#   :doc:`../auto_examples/deformable_particle_guide`). Particles with
#   the same ``bond_id`` belong to the same deformable body. By default
#   interactions between particles sharing a ``bond_id`` are **disabled**
#   unless ``interact_same_bond_id=True`` is set on the system.
# - ``mat_id`` — indexes into the :py:class:`~jaxdem.materials.MaterialTable`
#   to look up material properties (density, Young's modulus, …).
# - ``species_id`` — selects which force law applies to a pair when using
#   a :py:class:`~jaxdem.forces.router.ForceRouter` (see
#   :doc:`../auto_examples/force_model_guide`).
# - ``unique_id`` — a per-particle unique identifier (never repeated).

print("clump_id :", state.clump_id)
print("bond_id  :", state.bond_id)
print("mat_id   :", state.mat_id)
print("species_id:", state.species_id)

# %%
# Extending the State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Working directly with `SoA <https://en.wikipedia.org/wiki/AoS_and_SoA>`_ structures can sometimes feel less intuitive
# than Array-of-Structures (AoS) for adding and modifying individual particles. To simplify
# this, JaxDEM provides utility methods like :py:meth:`jaxdem.state.State.add`.
#
# :py:meth:`jaxdem.state.State.add` allows you to append new particles to an
# existing state, automatically assigning unique clump_ids and checking for dimension
# consistency.

state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([0.5]))
print(f"Initial state (N={state.N}, clump_ids={state.clump_id}):\npos={state.pos}")

state = jdem.State.add(
    state,
    pos=jnp.array([[1.0, 1.0]]),
    vel=2 * jnp.ones((1, 2)),
    rad=10 * jnp.ones(1),
)
print(
    f"\nState after addition (N={state.N}, clump_ids={state.clump_id}):\npos={state.pos}"
)
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
    clump_id=jnp.array([2, 3]),
)
print(
    f"\nState after adding multiple particles (N={state.N}, clump_ids={state.clump_id}):\n{state.pos}"
)

# %%
# Note that we provided explicit ``clump_id`` values here.
# :py:meth:`jaxdem.state.State.add` adds ``jnp.max(state.clump_id)`` to
# the provided IDs to avoid overlaps. The resulting sequence is not
# guaranteed to be contiguous, but this is perfectly valid.

# %%
# Merging Two States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To combine two :py:class:`~jaxdem.state.State` objects, use
# :py:meth:`jaxdem.state.State.merge`. It concatenates the particles from
# the second state onto the first — useful for assembling complex initial
# configurations from smaller parts.

state_a = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
)
state_b = jdem.State.create(
    jnp.array([[2.0, 2.0], [3.0, 3.0], [5.0, 2.0]]),
)
state = jdem.State.merge(state_a, state_b)

print(f"State A (N={state_a.N}, clump_ids={state_a.clump_id}):\npos={state_a.pos}")
print(f"State B (N={state_b.N}, clump_ids={state_b.clump_id}):\npos={state_b.pos}")
print(f"Merged state (N={state.N}, clump_ids={state.clump_id}):\npos={state.pos}")


# %%
# Stacking States for Trajectories or Batches
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One of the features that makes JaxDEM special is its ability
# to handle **batched states**. Batches can be interpreted as trajectories
# (multiple snapshots over time) or as independent simulations
# (multiple distinct initial conditions).
#
# This is useful for performance. JaxDEM is optimised for
# **throughput**: if your GPU is not saturated, you are leaving performance
# on the table. A common DEM task is running parameter sweeps. JaxDEM lets
# you run many independent simulations in parallel, potentially finishing
# all of them in the time it would take for just one, until the GPU is
# fully utilised.
#
# Furthermore, trajectory support means you don't have to interrupt the
# GPU for I/O (e.g., saving state to disk). You can accumulate a full
# trajectory in memory and save everything at the end, which often gives
# much better performance at the cost of a bit more memory.
#
# To manage simulation trajectories or perform batched simulations,
# :py:meth:`jaxdem.state.State.stack` is available. It takes a sequence of
# :py:class:`jaxdem.state.State` snapshots and concatenates them along a new
# leading axis. This creates a multi-dimensional state where the first axis
# can represent time steps, batch elements, or other high-level groupings.
# Note that stacking does *not* shift particle clump_ids, as it assumes the
# particles are the same entities across the stacked dimension.
# :py:meth:`jaxdem.state.State.stack` makes sure shapes are consistent.

snapshot1 = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([2.0]))
snapshot2 = jdem.State.create(pos=jnp.array([[0.1, 0.0]]), vel=jnp.array([[0.1, 0.0]]))
snapshot3 = jdem.State.create(pos=jnp.array([[0.2, 0.0]]), mass=jnp.array([3.3]))

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


def initialize(i: jax.Array) -> tuple[jdem.State, jdem.System]:
    state = jdem.State.create(i * jnp.ones((4, 2)))
    system = jdem.System.create(state.shape)
    return state, system


N_batches = 10
state, system = jax.vmap(initialize)(jnp.arange(N_batches))

# %%
# Then, to run this simulation:

state, system = system.step(state, system, n=10)
print(f"Shape of positions (B, N, dim): {state.pos.shape}")


# %%
# Note that system can change over time. Therefore, each state needs to have its own system.


# %%
# Trajectories of Batches
# ~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM's state handling capabilities extend beyond just batches or single trajectories.
# We can also accumulate **trajectories of batched states**.
#
# This is useful for scenarios like **parameter sweeps**, where you run
# multiple independent simulations (a batch) and want to capture their
# full time evolution (a trajectory) without frequent I/O. It allows
# highly efficient data collection.
#
# :py:meth:`jaxdem.writers.VTKWriter.save` understands these
# multi-dimensional states.
#
# By convention, when dealing with `State.pos` of shape `(..., N, dim)`:
#
# *   The **first leading dimension** (axis 0) is the **batch** dimension ``B``.
#     :py:attr:`~jaxdem.state.State.batch_size` returns this value.
# *   When collecting trajectories (via
#     :py:meth:`~jaxdem.system.System.trajectory_rollout`), each snapshot is
#     stacked along the **next** leading axis, giving shape
#     ``(B, T, N, dim)`` for batched trajectories.
#
# :py:meth:`jaxdem.writers.VTKWriter.save` understands these layouts. By
# default (``trajectory=False``) all leading axes are treated as independent
# batches. Pass ``trajectory=True`` to tell the writer which axis is time
# (``trajectory_axis``, default 0); the writer swaps that axis to the front,
# keeps it as ``T``, and flattens any remaining leading axes into a single
# batch axis ``B``, yielding ``(T, B, N, dim)`` internally.

batched_state = jdem.State.stack([batched_state, batched_state, batched_state])
print(f"Shape of stacked positions (T, B, N, dim): {batched_state.pos.shape}")
print(f"Batch size: {batched_state.batch_size}")

# %%
# Following the example of the previous section, you might encounter a trajectory of batches in the following way:

N_batches = 9
state, system = jax.vmap(initialize)(jnp.arange(N_batches))

state, system, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=10
)

print(f"Shape of positions (T, B, N, dim): {state_traj.pos.shape}")


# %%
# Utilities
# ~~~~~~~~~~
# JaxDEM includes utility functions in :py:mod:`jaxdem.utils` for
# quickly setting up simulations. For example, you can create a state
# with randomised attributes:

from jaxdem import utils as utils

state = utils.random_state(dim=3, N=10)
print(state)
