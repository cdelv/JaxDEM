r"""Colliders
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
#    * - ``"cell_list"``
#      - :py:class:`~jaxdem.colliders.cell_list.DynamicCellList`
#      - :math:`O(N \log N)`
#      - Low to moderate polydispersity systems and clumps
#    * - ``"multi_cell_list"``
#      - :py:class:`~jaxdem.colliders.multi_cell_list.DynamicMultiCellList`
#      - :math:`O(N \cdot max\_hashes \log (N \cdot max\_hashes))`
#      - Highly polydisperse systems (wide size distributions)
#    * - ``"neighbor_list"``
#      - :py:class:`~jaxdem.colliders.neighbor_list.NeighborList`
#      - :math:`O(N)` amortized
#      - Large systems with infrequent neighbor-list rebuilds
#
# Registry keys are normalized: lookups are case-insensitive and ignore
# underscores, spaces, and hyphens, so ``"cell_list"``, ``"CellList"``, and
# ``"celllist"`` all select the same class.
#
# The registered colliders are (the empty key ``""`` is a registered no-op;
# we filter it out):
print("Colliders:", sorted(k for k in jdem.Collider._registry if k))

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
# The Cell List Collider
# ~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.cell_list.DynamicCellList` (registered as ``"cell_list"``)
# partitions space into a regular grid. Only particles in the same or
# neighboring cells interact. It uses an implicit infinite grid, so it works for all domain
# types (periodic, free, etc.).
#
# It probes each cell with a ``jax.lax.while_loop``, making it robust to high or variable
# cell occupancy—ideal for polydisperse systems and clumps.
#
# Key parameters (all have automatic defaults):
#
# - ``cell_size`` — edge length of each grid cell.
# - ``box_size`` — domain size (optional; only needed when the box size is small compared with the cell size to ensure correct periodic wrap stencil dimensions).
#
# Cell-list colliders sort/reorder the state internally for traversal
# performance — this is an intentional performance feature. The returned
# state follows that sorted ordering, so track particle identity across
# steps via ``state.unique_id`` (for example,
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.add_force_at`
# addresses particles by ``unique_id``).
#
# Colliders whose ``Create`` method needs a reference state (cell lists,
# neighbor lists) receive it automatically when you pass ``state=`` to
# :py:meth:`~jaxdem.system.System.create`.

state_p = jdem.State.create(
    pos=jnp.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]),
    rad=jnp.array([0.5, 0.5, 0.5]),
)
system_cl = jdem.System.create(
    state=state_p,
    collider_type="cell_list",
)
print("Cell size:", getattr(system_cl.collider, "cell_size", "n/a"))


# %%
# The Multi-Cell List Collider
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.multi_cell_list.DynamicMultiCellList` (registered as ``"multi_cell_list"``)
# partitions space into a regular grid of size ``cell_size``. Unlike standard cell lists
# where the cell size is bounded by the largest particle diameter, the multi-cell list allows
# particles to span/register in multiple cells (up to ``max_hashes`` cells).
#
# This formulation is exceptionally well-suited for systems with extreme polydispersity,
# as it prevents a few large particles from forcing a large cell size for all the small particles.
#
# Key parameters (all have automatic defaults):
#
# - ``cell_size`` — edge length of each grid cell. If None, it defaults to the minimum particle diameter.
# - ``max_hashes`` — maximum number of cells a single particle is allowed to overlap.
#
# Like the standard cell list, it sorts/reorders the state internally for performance.

system_mcl = jdem.System.create(
    state=state_p,
    collider_type="multi_cell_list",
)
print("Multi-Cell List cell size:", getattr(system_mcl.collider, "cell_size", "n/a"))
print("Multi-Cell List max hashes:", getattr(system_mcl.collider, "max_hashes", "n/a"))


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
# .. note::
#    **Verifying Neighbor List Capacity with the Overflow Flag**
#
#    Since ``max_neighbors`` is a static, user-provided buffer size required for JAX compile-time sizing,
#    checking the returned ``overflow`` flag is the correct way to verify that your simulation is working
#    correctly. If ``overflow`` is ``True``, some particles have more neighbors than ``max_neighbors``,
#    meaning some interactions may be truncated. If this occurs, you must increase the ``max_neighbors``
#    parameter to ensure physical correctness.
#
# Example with a regular collider (here: Cell List):
_, _, nl_cl, overflow_cl = system_cl.collider.create_neighbor_list(
    state_p, system_cl, cutoff=2.0, max_neighbors=8
)
print("Cell-list neighbor list shape:", nl_cl.shape)
print("Cell-list overflow:", bool(overflow_cl))

# %%
# The Neighbor List collider
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.neighbor_list.NeighborList` caches a
# per-particle list of neighbors built with a secondary collider
# (by default, the cell list). The list is rebuilt only when
# some particle has moved more than ``skin / 2``. Between rebuilds, the
# cost is :math:`O(N)`.
#
# .. warning::
#    In a batched simulation (:py:func:`jax.vmap` over many systems), the
#    rebuild decision is a :py:func:`jax.lax.cond`, which vmap lowers to a
#    ``select`` that executes **both** branches: every batch member pays
#    the full rebuild cost at every step, whether or not its list was
#    stale. The neighbor list therefore loses its main advantage under
#    ``vmap`` and may not be the best collider choice for batched systems.
#
# Key parameters:
#
# - ``cutoff`` — physical interaction radius.
# - ``skin`` — **absolute** buffer distance added to the cutoff (the same
#   quantity the stored ``skin`` field holds). Must be > 0 for performance.
# - ``skin_fraction`` — alternative way to specify the skin as a fraction
#   of the cutoff (defaults to ``0.05`` when neither ``skin`` nor
#   ``skin_fraction`` is given). Passing both raises an error.
# - ``max_neighbors`` — buffer size per particle (auto-estimated if
#   omitted).
# - ``secondary_collider_type`` — any registered collider except another ``"neighbor_list"``.
#
# This design works because every collider exposes ``create_neighbor_list``.
# A ``NeighborList`` wrapping another ``NeighborList`` is not meaningful and
# should be avoided.
#
# When a rebuild occurs, ordering may change according to the secondary
# collider's sorting behavior.
#
# Note that the reference state is forwarded automatically — both to the
# neighbor list itself and to its secondary collider — when you pass
# ``state=`` to :py:meth:`~jaxdem.system.System.create`, so there is no
# need to repeat it inside ``collider_kw`` or ``secondary_collider_kw``.

system_nl = jdem.System.create(
    state=state_p,
    collider_type="neighbor_list",
    collider_kw={
        "cutoff": 2.0,
        "skin": 0.1,
        "secondary_collider_type": "cell_list",
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
# If you edit the state by hand (moving particles, changing radii, adding
# particles) after the system has been created, the cached neighbor list may
# become stale. Use :py:func:`jaxdem.colliders.refresh_collider` to rebuild a
# stateful collider from the edited state:
#
# .. code-block:: python
#
#    system_nl.collider = jdem.colliders.refresh_collider(edited_state, system_nl.collider)


# %%
# Computing Potential Energy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The collider exposes
# :py:meth:`~jaxdem.colliders.Collider.compute_potential_energy`, which
# sums all pairwise interaction energies as defined by the force model,
# and returns a tuple ``(state, system, potential_energy)``, where
# ``potential_energy`` is the **total** potential energy of the system.
#
# Crucially, calling ``compute_potential_energy`` ensures that any mutations
# to the state or collider (such as spatial sorting or neighbor list rebuilds)
# are preserved. For the ``"neighbor_list"`` collider, a rebuild also updates
# the ``system.collider.overflow`` flag; the naive and cell-list colliders do
# not maintain this flag during force or energy evaluation (they only report
# overflow through ``create_neighbor_list``).

state_pe = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
)
system_pe = jdem.System.create(state_pe.shape, force_model_type="spring")

state_pe, system_pe, pe = system_pe.collider.compute_potential_energy(
    state_pe, system_pe
)
print("Total potential energy:", pe)


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
