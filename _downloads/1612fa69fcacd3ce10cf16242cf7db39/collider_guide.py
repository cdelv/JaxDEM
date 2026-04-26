r"""Colliders.
----------------------------------------

A :py:class:`~jaxdem.colliders.Collider` is the component that detects
interacting particle pairs and evaluates the
:py:class:`~jaxdem.forces.ForceModel` for each pair. Different colliders
implement different spatial-search strategies, trading generality for speed.

This guide covers:

- The available collider implementations and when to use each one.
- How to configure a collider via ``collider_type`` / ``collider_kw``.
- How the collider interacts with force models and the force manager.
- Computing potential energy through the collider.
- Neighbor-list creation for diagnostics and caching.
"""

# %%
# Selecting a Collider
# ~~~~~~~~~~~~~~~~~~~~~~
# The collider is chosen via ``collider_type`` when creating a
# :py:class:`~jaxdem.system.System`. The default is ``"naive"``.

import jax.numpy as jnp
import jaxdem as jdem

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]),
    rad=jnp.array([1.0, 1.0, 1.0]),
)
system = jdem.System.create(state.shape, collider_type="naive")
print("Collider:", type(system.collider).__name__)


# %%
# Available Colliders
# ~~~~~~~~~~~~~~~~~~~~~
# JaxDEM provides several collider implementations registered in the
# :py:class:`~jaxdem.colliders.Collider` factory:
#
# .. list-table::
#    :header-rows: 1
#
#    * - ``collider_type``
#      - Class
#      - Complexity
#      - Best for
#    * - ``"naive"``
#      - :py:class:`~jaxdem.colliders.naive.NaiveSimulator`
#      - :math:`O(N^2)`
#      - Small systems (< 1k–4k particles)
#    * - ``"StaticCellList"``
#      - :py:class:`~jaxdem.colliders.cell_list.StaticCellList`
#      - :math:`O(N \log N)`
#      - Monodisperse or mildly polydisperse systems of spheres
#    * - ``"CellList"``
#      - :py:class:`~jaxdem.colliders.cell_list.DynamicCellList`
#      - :math:`O(N \log N)`
#      - Polydisperse systems and clumps
#    * - ``"NeighborList"``
#      - :py:class:`~jaxdem.colliders.neighbor_list.NeighborList`
#      - :math:`O(N)` amortised
#      - Large, cold systems with infrequent rebuilds
#
# The registered colliders are:
print("Colliders:", list(jdem.Collider._registry.keys()))

# %%
# The Naive Collider
# ~~~~~~~~~~~~~~~~~~~~
# The :py:class:`~jaxdem.colliders.naive.NaiveSimulator` evaluates the
# force model for **every** pair :math:`(i, j)`, giving :math:`O(N^2)`
# complexity. It requires no configuration and is the default.
# This is by far the fastest option for small systems because it has no
# overhead, but it becomes prohibitively expensive as :math:`N` grows.

system_naive = jdem.System.create(state.shape, collider_type="naive")
state_out, system_out = system_naive.step(state, system_naive)
print("Forces after one step:\n", state_out.force)


# %%
# The Static Cell List
# ~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.cell_list.StaticCellList` partitions space
# into a regular grid. Only particles in the same or neighboring cells
# interact. It uses an implicit infinite grid, so it works for all domain
# types (periodic, free, etc.) as long as a ``box_size`` is provided.
#
# Key parameters (all have automatic defaults):
#
# - ``cell_size`` — edge length of each grid cell.
# - ``max_occupancy`` — maximum particles expected per cell.
# - ``box_size`` — domain size (optional; used to ensure the grid fits in
#   periodic domains).
#
# Important: cell-list colliders sort/reorder the state internally for
# traversal performance. The returned state follows that sorted ordering.
#
# The static cell list is very fast for monodisperse or mildly
# polydisperse spheres, but it can break down if some cells become too
# crowded (e.g., due to large overlaps or clumps) because it only probes
# a fixed number of particles per cell. Clumps usually have many
# overlapping particles that belong to the same clump, which can
# overcrowd cells and cause missed interactions with other particles in
# the same cell.

state_p = jdem.State.create(
    pos=jnp.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]),
    rad=jnp.array([0.5, 0.5, 0.5]),
)
system_cl = jdem.System.create(
    state_p.shape,
    collider_type="StaticCellList",
    collider_kw={"state": state_p},
)
print("Cell size:", getattr(system_cl.collider, "cell_size", "n/a"))
print("Max occupancy:", getattr(system_cl.collider, "max_occupancy", "n/a"))


# %%
# The Dynamic Cell List
# ~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.cell_list.DynamicCellList` uses the same
# spatial hashing, but probes each cell with a ``jax.lax.while_loop``
# instead of a fixed ``max_occupancy`` window. This makes it robust to
# high or variable cell occupancy — ideal for **polydisperse** systems
# and **clumps**. Its overhead is higher than the static cell list, and
# operations that happen in parallel in the static cell list happen
# sequentially here, so it is slower.
#
# Like the static cell list, this collider also sorts/reorders the state,
# and neighbor indices refer to that sorted state.

system_dcl = jdem.System.create(
    state_p.shape,
    collider_type="CellList",
    collider_kw={"state": state_p},
)
print("Dynamic cell list collider:", type(system_dcl.collider).__name__)


# %%
# Neighbor-list creation for all colliders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every collider implements
# :py:meth:`~jaxdem.colliders.Collider.create_neighbor_list`. This is useful
# both for diagnostics and for algorithms that need explicit neighbors.
#
# The API returns:
#
# - ``neighbor_list`` with shape ``(N, max_neighbors)`` padded with ``-1``.
# - ``overflow`` flag, which is ``True`` if any particle had more than
#   ``max_neighbors`` neighbors within the requested cutoff.
#
# Since ``max_neighbors`` is a static, user-provided buffer size, checking
# ``overflow`` is the correct way to verify that the neighbor list was built
# properly for your chosen cutoff/density.
#
# Example with a regular collider (here: Dynamic Cell List):
_, _, nl_cl, overflow_cl = system_dcl.collider.create_neighbor_list(
    state_p, system_dcl, cutoff=2.0, max_neighbors=8
)
print("Cell-list neighbor list shape:", nl_cl.shape)
print("Cell-list overflow:", bool(overflow_cl))

# %%
# The Neighbor List collider
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.neighbor_list.NeighborList` caches a
# per-particle list of neighbors built with a secondary collider
# (by default, the dynamic cell list). The list is rebuilt only when
# some particle has moved more than ``skin / 2``. Between rebuilds, the
# cost is :math:`O(N)`.
#
# Key parameters:
#
# - ``cutoff`` — physical interaction radius.
# - ``skin`` — buffer distance (fraction of cutoff). Must be > 0 for
#   performance.
# - ``max_neighbors`` — buffer size per particle (auto-estimated if
#   omitted).
# - ``secondary_collider_type`` — any registered collider that can build
#   a neighbor list (``"naive"``, ``"StaticCellList"``, ``"CellList"``, …),
#   except another ``"NeighborList"``.
#
# This design works because every collider exposes ``create_neighbor_list``.
# A ``NeighborList`` wrapping another ``NeighborList`` is not meaningful and
# should be avoided.
#
# When a rebuild occurs, ordering may change according to the secondary
# collider's sorting behaviour.

system_nl = jdem.System.create(
    state_p.shape,
    collider_type="NeighborList",
    collider_kw={
        "state": state_p,
        "cutoff": 2.0,
        "skin": 0.1,
        "secondary_collider_type": "CellList",
        "secondary_collider_kw": {"state": state_p},
        "max_neighbors": 8,
    },
)
print("Neighbor list collider:", type(system_nl.collider).__name__)
print("Cutoff:", float(getattr(system_nl.collider, "cutoff", jnp.nan)))
print("Skin:", float(getattr(system_nl.collider, "skin", jnp.nan)))
print("Max neighbors:", getattr(system_nl.collider, "max_neighbors", "n/a"))
print("Number of builds:", getattr(system_nl.collider, "n_build_times", "n/a"))
print("Last build overflow:", bool(getattr(system_nl.collider, "overflow", False)))


# %%
# Computing Potential Energy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The collider exposes
# :py:meth:`~jaxdem.colliders.Collider.compute_potential_energy`, which
# sums all pairwise interaction energies as defined by the force model,
# per particle:

state_pe = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
)
system_pe = jdem.System.create(state_pe.shape, force_model_type="spring")

pe = system_pe.collider.compute_potential_energy(state_pe, system_pe)
print("Per particle PE energy:", pe)


# %%
# How the Collider Fits in the Step Pipeline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# During each integration step, the pipeline is:
#
# 1. **Domain** — applies boundary conditions.
# 2. **Integrator** (before force) — advances positions a half-step.
# 3. **Collider** — evaluates pairwise forces and writes ``state.force``
#    / ``state.torque``.
# 4. **Force manager** — adds gravity, external forces, custom force
#    functions, and performs rigid-body aggregation.
# 5. **Integrator** (after force) — advances velocities.
#
# The collider only writes the *pairwise contact* contributions and
# resets forces; the force manager then adds everything else on top.
