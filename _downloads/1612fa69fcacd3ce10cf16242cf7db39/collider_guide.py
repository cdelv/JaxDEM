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
# You choose the collider via ``collider_type`` when creating a
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
# The registry normalizes keys: lookups ignore case, underscores, spaces,
# and hyphens, so ``"cell_list"``, ``"CellList"``, and
# ``"celllist"`` all select the same class.
#
# The registered colliders are (the empty key ``""`` is a registered no-op,
# and we filter it out):
print("Colliders:", sorted(k for k in jdem.Collider._registry if k))

# %%
# The Naive Collider
# ~~~~~~~~~~~~~~~~~~~~
# The :py:class:`~jaxdem.colliders.naive.NaiveSimulator` evaluates the
# force model for **every** pair :math:`(i, j)`, giving :math:`O(N^2)`
# complexity. It requires no configuration and is the default.
# It has no search overhead, so it is the fastest option for small
# systems. The cost grows quickly as :math:`N` grows.

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
# It probes each cell with a ``jax.lax.while_loop``, so it handles high or
# variable cell occupancy. This suits polydisperse systems and clumps.
#
# Key parameters (all have automatic defaults):
#
# - ``cell_size`` — edge length of each grid cell.
# - ``box_size`` — domain size (optional; needed only when the box size is
#   small compared with the cell size, to size the periodic wrap stencil
#   correctly).
#
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
# partitions space into a regular grid of cells of edge length ``cell_size``
# and bins each particle into exactly one cell by its center. Unlike a
# standard cell list, each cell also carries an expandable bounding box that
# covers its members, so the collider can skip whole cells during the search.
#
# This suits systems with extreme polydispersity. A few large particles no
# longer force a large cell size on all the small particles.
#
# Key parameters (all have automatic defaults):
#
# - ``cell_size`` — edge length of each grid cell. If None, it defaults to the minimum particle diameter.


system_mcl = jdem.System.create(
    state=state_p,
    collider_type="multi_cell_list",
)
print("Multi-Cell List cell size:", getattr(system_mcl.collider, "cell_size", "n/a"))


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
#    ``max_neighbors`` is a static buffer size that JAX needs at compile time.
#    Check the returned ``overflow`` flag to verify the buffer is large enough.
#    If ``overflow`` is ``True``, some particles have more neighbors than
#    ``max_neighbors``, and some interactions are dropped. In that case,
#    increase ``max_neighbors``.
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
# (by default, the cell list). The collider rebuilds the list only when
# some particle has moved more than ``skin / 2``. Between rebuilds, the
# cost is :math:`O(N)`.
#
# .. warning::
#    In a batched simulation (:py:func:`jax.vmap` over many systems), the
#    rebuild decision is a :py:func:`jax.lax.cond`, which vmap lowers to a
#    ``select`` that executes **both** branches. Every batch member pays
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
# Do not wrap a ``NeighborList`` in another ``NeighborList``.
#
# When you pass ``state=`` to :py:meth:`~jaxdem.system.System.create`, it
# forwards the reference state to the neighbor list and to its secondary
# collider. Do not repeat it inside ``collider_kw`` or
# ``secondary_collider_kw``.

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
# If you edit the state by hand after creating the system, the cached
# neighbor list may become stale. Edits include moving particles, changing
# radii, or adding particles. Use :py:func:`jaxdem.colliders.refresh_collider`
# to rebuild a stateful collider from the edited state:
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
# Calling ``compute_potential_energy`` also preserves any mutations
# to the state or collider, such as neighbor-list rebuilds. For the
# ``"neighbor_list"`` collider, a rebuild also updates
# the ``system.collider.overflow`` flag. The naive and cell-list colliders do
# not maintain this flag during force or energy evaluation. They only report
# overflow through ``create_neighbor_list``.

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
#    functions, and aggregates rigid-body forces.
# 5. **Integrator** (after force) — advances velocities.
#
# The collider only writes the *pairwise contact* contributions and
# resets forces. The force manager then adds everything else on top.
