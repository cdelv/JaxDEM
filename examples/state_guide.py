# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""The Simulation State
-----------------------

This tutorial covers :py:class:`jaxdem.state.State`, which holds all
per-particle data in a Structure-of-Arrays
(`SoA <https://en.wikipedia.org/wiki/AoS_and_SoA>`_) layout suited for JAX
vectorization, trajectories, and batched simulations.
"""

# %%
# State Creation
# ~~~~~~~~~~~~~~
# Pass an array of particle positions with shape ``(N, dim)`` (2D or 3D).
# :py:meth:`jaxdem.state.State.create` infers the spatial dimension from
# ``pos`` and fills unspecified fields with defaults. Pass ``mat_table=...`` to
# compute particle mass and inertia automatically from material density (see
# :doc:`../auto_examples/materials_guide`), or ``only_spheres=True`` to opt
# into sphere-only fast paths when your simulation has no clumps or facets.

import dataclasses

import jax
import jax.numpy as jnp

import jaxdem as jdem

state_2d = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
state_3d = jdem.State.create(pos=jnp.array([[0.0, 0.0, 0.0]]))
print(f"2D state: dim={state_2d.dim}, ang_vel shape={state_2d.ang_vel.shape}")
print(f"3D state: dim={state_3d.dim}, ang_vel shape={state_3d.ang_vel.shape}")

# %%
# Notice that rotational vectors (:py:attr:`~jaxdem.state.State.ang_vel`,
# :py:attr:`~jaxdem.state.State.torque`, :py:attr:`~jaxdem.state.State.inertia`)
# have shape ``(N, 1)`` in 2D (rotation about the z-axis) and ``(N, 3)`` in 3D.
# You can also create an empty state (``jdem.State.create(dim=2)``) as a
# starting point for :py:meth:`~jaxdem.state.State.add` or
# :py:meth:`~jaxdem.state.State.merge`.

# %%
# Modifying State Attributes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Prefer passing initial attributes directly to
# :py:meth:`~jaxdem.state.State.create`, which validates shapes and keeps all
# fields consistent:

state = jdem.State.create(pos=jnp.zeros((2, 2)), vel=jnp.ones((2, 2)))
print("Created vel:\n", state.vel)

# %%
# During host-side setup you can also reassign array attributes directly, using
# JAX's ``.at[...].set(...)`` syntax for partial updates (JAX arrays are
# immutable, so ``state.vel[0] = ...`` raises an error):

state.vel = state.vel.at[0].set(jnp.array([2.0, -1.0]))
print("Updated vel:\n", state.vel)

# %%
# Inside JAX transformations (``jit``, ``vmap``, ``scan``), treat
# :py:class:`~jaxdem.state.State` as an immutable pytree: rebind the returned
# object or use ``dataclasses.replace``. Quaternion components are
# immutable, so replace the whole quaternion:
# ``state.q = dataclasses.replace(state.q, w=new_w, xyz=new_xyz)``.
# Before a long compiled run, call :py:meth:`~jaxdem.state.State.validate`
# (outside ``jit``) to check shapes, finiteness, and invariants.

state.validate()

# %%
# Understanding Positions: :py:attr:`~jaxdem.state.State.pos`, :py:attr:`~jaxdem.state.State.pos_c`, and :py:attr:`~jaxdem.state.State.pos_p`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:attr:`state.pos <jaxdem.state.State.pos>` is a **computed property**, not
# a stored field: ``pos = pos_c + R(q) @ pos_p``, where ``R(q)`` is the
# rotation given by :py:attr:`state.q <jaxdem.state.State.q>`.
#
# * :py:attr:`pos_c <jaxdem.state.State.pos_c>` — center-of-mass position of
#   each sphere or rigid clump.
# * :py:attr:`pos_p <jaxdem.state.State.pos_p>` — body-frame offset from the
#   center of mass. For simple spheres ``pos_p == 0``, so ``pos == pos_c``.
#
# For multi-sphere rigid bodies (clumps), all member spheres share COM fields
# (:py:attr:`~jaxdem.state.State.pos_c`, :py:attr:`~jaxdem.state.State.q`,
# :py:attr:`~jaxdem.state.State.vel`, :py:attr:`~jaxdem.state.State.mass`,
# :py:attr:`~jaxdem.state.State.inertia`, :py:attr:`~jaxdem.state.State.clump_id`)
# while keeping per-sphere :py:attr:`~jaxdem.state.State.pos_p`,
# :py:attr:`~jaxdem.state.State.rad`, and material/species IDs. See
# :doc:`../auto_examples/clump_guide` for details.

# %%
# Fixed and Prescribed-Motion Particles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setting :py:attr:`state.fixed <jaxdem.state.State.fixed>` to ``True`` masks
# acceleration updates by ``(1 - fixed)``. A fixed particle keeps its current
# velocity regardless of forces — zero velocity holds it stationary, nonzero
# velocity moves it at a constant prescribed speed.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [2.0, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
    fixed=jnp.array([True, False]),
)
print("Fixed mask:", state.fixed)

# %%
# Identifier Fields
# ~~~~~~~~~~~~~~~~~
# Each particle carries four integer identifiers:
#
# - :py:attr:`clump_id <jaxdem.state.State.clump_id>` — groups spheres into
#   rigid bodies (:doc:`../auto_examples/clump_guide`). Members of the same
#   clump never collide with one another.
# - :py:attr:`bond_id <jaxdem.state.State.bond_id>` — neighbor-index array of
#   shape ``(N, max_num_neighbors)`` padded with ``-1``. Connected pairs are
#   excluded from non-bonded contacts by default (override with
#   ``interact_same_bond_id=True`` in :py:meth:`~jaxdem.system.System.create`).
#   See :doc:`../auto_examples/deformable_particle_guide`.
# - :py:attr:`mat_id <jaxdem.state.State.mat_id>` — indexes into
#   :py:class:`~jaxdem.materials.material_table.MaterialTable`.
# - :py:attr:`species_id <jaxdem.state.State.species_id>` — selects pairwise
#   force laws in :py:class:`~jaxdem.forces.router.ForceRouter`
#   (:doc:`../auto_examples/force_model_guide`).

print("clump_id  :", state.clump_id)
print("bond_id   :", state.bond_id)
print("mat_id    :", state.mat_id)
print("species_id:", state.species_id)

# %%
# Passing uneven lists of neighbor indices to ``bond_id`` automatically
# symmetrizes connections (if A connects to B, B connects to A) and pads unused
# slots with ``-1``:

positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
state_bonded = jdem.State.create(pos=positions, bond_id=[[1, 2], [0], [0], []])
print("Symmetrized bond_id:\n", state_bonded.bond_id)

# %%
# Extending and Merging States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`jaxdem.state.State.add` appends particles to an existing state and
# shifts the new group's :py:attr:`~jaxdem.state.State.clump_id` above existing
# IDs (preserving grouping, not literal input integers):

state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([0.5]))
state = jdem.State.add(
    state,
    pos=jnp.array([[1.0, 1.0], [2.0, 0.0]]),
    vel=jnp.zeros((2, 2)),
    rad=jnp.array([0.8, 0.3]),
)
print(f"After add (N={state.N}, clump_ids={state.clump_id}):\n{state.pos}")

# %%
# :py:meth:`jaxdem.state.State.merge` concatenates two existing states:

state_a = jdem.State.create(pos=jnp.array([[0.0, 0.0], [1.0, 1.0]]))
state_b = jdem.State.create(pos=jnp.array([[2.0, 2.0], [3.0, 3.0], [5.0, 2.0]]))
merged = jdem.State.merge(state_a, state_b)
print(f"Merged (N={merged.N}, clump_ids={merged.clump_id})")

# %%
# Stacking, Batches, and Trajectories
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM uses a consistent leading-axis convention for ``(..., N, dim)`` arrays:
#
# * ``(N, dim)`` — a single snapshot.
# * ``(B, N, dim)`` — a batch of ``B`` independent simulations.
# * ``(T, N, dim)`` — a trajectory of ``T`` time snapshots.
# * ``(T, B, N, dim)`` — a trajectory of ``B`` parallel simulations.
#
# :py:attr:`~jaxdem.state.State.batch_size` always reads axis ``-3`` (just
# before the particle axis ``N``). Use :py:meth:`~jaxdem.state.State.stack` and
# :py:meth:`~jaxdem.state.State.unstack` (or ``jax.vmap``) to build and split
# leading axes without shifting :py:attr:`~jaxdem.state.State.clump_id`:

s1 = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([2.0]))
s2 = jdem.State.create(pos=jnp.array([[0.1, 0.0]]), vel=jnp.array([[0.1, 0.0]]))
s3 = jdem.State.create(pos=jnp.array([[0.2, 0.0]]), mass=jnp.array([3.3]))

stacked = jdem.State.stack([s1, s2, s3])
print(f"Stacked shape (B, N, dim): {stacked.pos.shape}, batch_size={stacked.batch_size}")
unstacked = jdem.State.unstack(stacked)
print(f"Unstacked count: {len(unstacked)}, first pos: {unstacked[0].pos}")

batched = jax.vmap(lambda i: jdem.State.create(i * jnp.ones((1, 2))))(jnp.arange(4))
traj_of_batches = jdem.State.stack([batched, batched, batched])
print(f"Trajectory of batches (T, B, N, dim): {traj_of_batches.pos.shape}")

# %%
# See :doc:`../auto_examples/system_guide` for stepping and rolling out batched
# simulations, and :doc:`../auto_examples/vtk_writer_guide` for exporting
# batched trajectories to ParaView.

# %%
# State Generators
# ~~~~~~~~~~~~~~~~
# :py:mod:`jaxdem.utils` includes quick helpers for initial states, such as
# :py:func:`~jaxdem.utils.random_state` and :py:func:`~jaxdem.utils.grid_state`:

rand_state = jdem.utils.random_state(dim=3, N=4)
grid = jdem.utils.grid_state(n_per_axis=(2, 2), spacing=2.5)
print("random_state shape:", rand_state.pos.shape, "| grid_state shape:", grid.pos.shape)
