"""The Simulation State
----------------------------------------

This example covers the :py:class:`jaxdem.state.State` object,
a core component of JaxDEM that holds all particle data in a simulation.

JaxDEM stores particle data in a Structure-of-Arrays (`SoA <https://en.wikipedia.org/wiki/AoS_and_SoA>`_)
layout, which suits JAX's vectorized and parallel computations. This
layout also makes trajectories and batched simulations easy to handle.

This guide shows how to create, modify, and extend the simulation state.
"""

# %%
# State Creation
# ~~~~~~~~~~~~~~~~~~~~~
# We start with a simple 2D state with a single particle
# at the origin. By default, :py:meth:`jaxdem.state.State.create`
# fills non-specified attributes (like velocity, radius, mass) with
# default values.

import jax
import jaxdem as jdem
import jax.numpy as jnp

state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
print(f"Dimension of state: {state.dim}")
print(f"Initial position: {state.pos}")

# %%
# To create a 3D state, pass 3D coordinates. JaxDEM infers the
# dimension from the position data (JaxDEM supports only 2D and 3D).

state = jdem.State.create(pos=jnp.array([[0.0, 0.0, 0.0]]))
print(f"Dimension of state: {state.dim}")
print(f"Initial position: {state.pos}")

# %%
# Understanding Positions: ``pos``, ``pos_c``, and ``pos_p``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note: ``state.pos`` is **not** a stored field. It is a
# computed property: ``pos = pos_c + R(q) @ pos_p``, where
# ``R(q)`` is the rotation given by the particle's quaternion orientation.
#
# The stored fields are:
#
# *   ``pos_c`` — the center-of-mass position of each particle (or clump).
# *   ``pos_p`` — the offset from the center of mass in the **principal
#     (body) frame**. For simple spheres ``pos_p`` is zero, so ``pos == pos_c``.
#
# For **clumps** (rigid bodies made of multiple spheres), every sphere in the
# same clump shares the *same* center of mass position ``pos_c``, orientation ``q``,
# velocity ``vel``, angular velocity ``ang_vel``, ``force``, ``torque``, ``mass``,
# ``inertia``, ``fixed``, and ``clump_id``.
# The per-sphere fields that can vary within a rigid clump are:
#
# *   ``pos_p`` — the body-frame offset relative to the COM
# *   ``rad`` — individual sphere radius
# *   ``volume`` — stored per sphere: ``State.create`` defaults it to each
#     sphere's own hypersphere volume, while
#     :py:meth:`~jaxdem.state.State.add_clump` broadcasts a provided clump
#     volume to every member sphere. Packing-fraction utilities read one
#     value per clump (via a segment max).
# *   **Identifiers**: ``clump_id``, ``bond_id``, ``mat_id``, ``species_id``
#
# This design allows vectorized operations over all spheres without branching
# on clump membership.

# %%
# Particle Sizes: ``rad`` and ``_rad``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM distinguishes between the physical size of a particle and its search/interaction radius:
#
# *   ``rad`` — the "true" physical radius of the particle. Narrow-phase
#     contact force and energy calculations use this radius, as do
#     plotting and visualization. For facets, ``rad`` represents the physical
#     thickness of the segment or triangle.
# *   ``_rad`` — the private "search" radius of the particle. The broad-phase
#     colliders use it to build neighbor/cell lists. For standard spheres, ``_rad`` is
#     equal to ``rad``. For facet vertices, ``_rad`` is the maximum
#     distance from the vertex to the center of mass (COM) of the facet, so
#     the broad-phase candidate list covers the entire facet.
#
# JaxDEM computes the broad-phase search radius ``_rad`` internally. User-facing
# constructors (like ``State.create`` or ``State.add_facet``) do not expose it
# as a parameter.

# %%
# Modifying State Attributes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We have two primary ways to set or modify particle attributes:
#
# 1.  **Direct assignment:** You can assign new JAX arrays
#     to attributes like `state.vel`. This is flexible, but you must
#     keep the shapes consistent.

state.vel = jnp.ones_like(state.pos)
print(state.vel)

# %%
# Because these are JAX arrays, code like
#
# .. code-block:: python
#
#     state.vel[i] = jnp.asarray([1, 2, 3], dtype=float)
#
# raises an error. The correct way is

i = 0
state.vel = state.vel.at[i].set(jnp.asarray([1, 2, 3], dtype=float))
print(state.vel)

# %%
# This is inefficient. Prefer vectorized operations.

# %%
# 2.  **Constructor arguments:** This is the
#     safer approach. The :py:meth:`jaxdem.state.State.create`
#     constructor validates shapes and types and keeps all attributes
#     consistent.

state = jdem.State.create(pos=jnp.zeros((1, 2)), vel=jnp.ones((1, 2)))
print(state.vel)

# %%
# Fixed (Immobile) Particles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The boolean field ``state.fixed`` marks particles with a prescribed
# motion. Forces do not drive them. The integrator multiplies
# velocity *updates* by ``(1 - fixed)``, which masks the acceleration.
# A fixed particle keeps whatever velocity it currently has, no matter
# which forces act on it. With zero initial velocity it stays put. With a
# nonzero initial velocity it keeps moving at that prescribed
# velocity. This is useful for walls, obstacles, or driven boundary
# particles.

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
# - ``bond_id`` — connectivity masking array (see
#   :doc:`../auto_examples/deformable_particle_guide`). For each particle,
#   it stores the array indices of the neighbor particles it connects to.
#   JaxDEM disables (masks out) interactions between connected particles.
#   It has shape ``(N, max_num_neighbors)``, and JaxDEM pads it with ``-1``.
# - ``mat_id`` — indexes into the :py:class:`~jaxdem.materials.MaterialTable`
#   to look up material properties (density, Young's modulus, …).
# - ``species_id`` — selects which force law applies to a pair when using
#   a :py:class:`~jaxdem.forces.router.ForceRouter` (see
#   :doc:`../auto_examples/force_model_guide`).


print("clump_id :", state.clump_id)
print("bond_id  :", state.bond_id)
print("mat_id   :", state.mat_id)
print("species_id:", state.species_id)

# %%
# Setting Up Connections with ``bond_id``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For each particle, the ``bond_id`` array stores the array indices of other particles
# it connects to.
#
# By default, the collider uses these connections to **ignore contact/non-bonded interactions**
# (so connected particles do not collide with each other). The ``interact_same_bond_id``
# parameter in the system creation options controls this behavior (see :py:meth:`jaxdem.system.System.create`).
# Set ``interact_same_bond_id=True`` to let particles connected via ``bond_id`` still feel contact forces.
#
# This connectivity masking is useful with **bonded models** (where interactions are permanent),
# such as deformable particle models and cohesive/bonded networks. For details on bonded interactions,
# see the :doc:`../auto_examples/deformable_particle_guide` and the :doc:`../auto_examples/collider_guide`.
#
# When you call :py:meth:`jaxdem.state.State.create`, you can define connections by passing
# a list of lists (which can have uneven lengths) for the ``bond_id`` argument.
# JaxDEM symmetrizes these connections (if particle A connects to B, then
# B connects to A) and pads the array with ``-1`` up to the maximum number of connections.
# If a particle has no connections, its row contains only ``-1``.
# If you pass no connections at all, ``bond_id`` defaults to a shape of ``(N, 1)`` filled with ``-1``.

# Create a state with 4 particles:
# - Particle 0 connects to 1 and 2
# - Particle 1 connects to 0
# - Particle 2 connects to 0
# - Particle 3 has no connections
positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
state_bonded = jdem.State.create(pos=positions, bond_id=[[1, 2], [0], [0], []])
print("Bond IDs for each particle:\n", state_bonded.bond_id)

# Default behavior when bond_id is not passed
state_no_bonds = jdem.State.create(pos=positions)
print("Default bond IDs (no bonds):\n", state_no_bonds.bond_id)


# %%
# Extending the State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Working directly with `SoA <https://en.wikipedia.org/wiki/AoS_and_SoA>`_ structures can feel less intuitive
# than Array-of-Structures (AoS) when you add or modify individual particles. To help with
# this, JaxDEM provides utility methods like :py:meth:`jaxdem.state.State.add`.
#
# :py:meth:`jaxdem.state.State.add` appends new particles to an
# existing state. It assigns unique clump_ids and checks dimension
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
# You can also add several particles at once with arrays of the
# appropriate shape. :py:meth:`jaxdem.state.State.add` checks that the dimensions
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
# :py:meth:`jaxdem.state.State.add` adds ``jnp.max(state.clump_id) + 1`` to
# the provided IDs to avoid overlaps. The resulting sequence may not be
# contiguous, and that is valid.

# %%
# Merging Two States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To combine two :py:class:`~jaxdem.state.State` objects, use
# :py:meth:`jaxdem.state.State.merge`. It concatenates the particles from
# the second state onto the first. This is useful for assembling initial
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
# JaxDEM can handle **batched states**. A batch can represent
# trajectories (multiple snapshots over time) or independent simulations
# (multiple distinct initial conditions).
#
# This is useful for performance. JaxDEM targets **throughput**: if your
# GPU is not saturated, you are leaving performance on the table. A common
# DEM task is running parameter sweeps. JaxDEM lets you run many
# independent simulations in parallel. Until you saturate the GPU, all of
# them can finish in about the time of one.
#
# Also, with trajectory support you do not have to interrupt the
# GPU for I/O (e.g., saving state to disk). You can accumulate a full
# trajectory in memory and save everything at the end. This is often
# faster at the cost of more memory.
#
# To manage simulation trajectories or run batched simulations, use
# :py:meth:`jaxdem.state.State.stack`. It takes a sequence of
# :py:class:`jaxdem.state.State` snapshots and concatenates them along a new
# leading axis. This creates a multi-dimensional state where the first axis
# can represent time steps, batch elements, or other high-level groupings.
# Note that stacking does *not* shift particle clump_ids, because it assumes
# the particles are the same entities across the stacked dimension.
# :py:meth:`jaxdem.state.State.stack` makes sure shapes are consistent.

snapshot1 = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([2.0]))
snapshot2 = jdem.State.create(pos=jnp.array([[0.1, 0.0]]), vel=jnp.array([[0.1, 0.0]]))
snapshot3 = jdem.State.create(pos=jnp.array([[0.2, 0.0]]), mass=jnp.array([3.3]))

batched_state = jdem.State.stack([snapshot1, snapshot2, snapshot3])

print(f"Shape of stacked positions (B, N, dim): {batched_state.pos.shape}")
print(f"Batch size: {batched_state.batch_size}")

# %%
# Another way to create batch states is JAX's vmap:

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
# A more realistic way to get a batched state:


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
# The system can change over time, so each state needs its own system.


# %%
# Trajectories of Batches
# ~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM can also accumulate **trajectories of batched states**.
#
# This is useful for **parameter sweeps**: you run several independent
# simulations (a batch) and capture their full time evolution (a
# trajectory) without frequent I/O.
#
# :py:meth:`jaxdem.writers.VTKWriter.save` understands these
# multi-dimensional states.
#
# By convention, when dealing with `State` attributes of shape `(..., N, dim)`:
#
# *   For a **single snapshot** (no batch, no trajectory), the shape is ``(N, dim)``.
# *   For a **batched state** (a single snapshot across multiple independent simulations), the shape is ``(B, N, dim)``.
# *   For a **single trajectory** (multiple snapshots over time of a single simulation), the shape is ``(T, N, dim)``.
# *   For a **trajectory of batches** (multiple snapshots over time of multiple parallel simulations), the shape is ``(T, B, N, dim)``.
#
# In JaxDEM, the batch dimension ``B`` (if present) is always at ``shape[-3]``.
# This is the axis before the particle dimension ``N`` at ``shape[-2]``.
# So :py:attr:`~jaxdem.state.State.batch_size` returns ``shape[-3]`` when
# ``ndim >= 3``. This gives ``B`` for both ``(B, N, dim)`` and
# ``(T, B, N, dim)`` shapes.
#
# When collecting trajectories (via :py:meth:`~jaxdem.system.System.trajectory_rollout`), each snapshot is
# stacked along the first axis (axis 0), producing a state of shape
# ``(T, B, N, dim)`` for batched trajectories.
#
# :py:meth:`jaxdem.writers.VTKWriter.save` understands these layouts. By
# default (``trajectory=False``) it treats all leading axes as independent
# batches. Pass ``trajectory=True`` to tell the writer which axis is time
# (``trajectory_axis``, default 0). The writer swaps that axis to the front,
# keeps it as ``T``, and flattens any remaining leading axes into a single
# batch axis ``B``. The result is ``(T, B, N, dim)`` internally.

batched_state = jdem.State.stack([batched_state, batched_state, batched_state])
print(f"Shape of stacked positions (T, B, N, dim): {batched_state.pos.shape}")
print(f"Batch size: {batched_state.batch_size}")

# %%
# As in the previous section, you can get a trajectory of batches like this:

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
# setting up simulations. For example, you can create a state
# with randomized attributes:

from jaxdem import utils as utils

state = utils.random_state(dim=3, N=10)
print(state)
