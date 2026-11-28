# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Colliders
-------------

A :py:class:`~jaxdem.colliders.Collider` detects interacting particle pairs and
evaluates the :py:class:`~jaxdem.forces.ForceModel` on each pair. Pass
``collider_type=None`` to disable non-bonded contacts altogether (e.g. for
bonded-only or external-force-only systems).
"""

# %%
# Comparing Built-in Colliders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Registry names are case- and separator-insensitive (``"cell_list"`` and
# ``"CellList"`` select the same class):
#
# .. list-table::
#    :header-rows: 1
#
#    * - ``collider_type``
#      - Class
#      - Complexity
#      - Pair history
#      - Reverse AD
#      - Best for
#    * - ``"naive"``
#      - :py:class:`~jaxdem.colliders.naive.NaiveSimulator`
#      - :math:`O(N^2)`
#      - Yes
#      - Yes
#      - Small systems, exact Hessians, dense history
#    * - ``"cell_list"``
#      - :py:class:`~jaxdem.colliders.cell_list.DynamicCellList`
#      - :math:`O(N \log N)`
#      - No
#      - With ``static=True``
#      - Low-to-moderate polydispersity and clumps
#    * - ``"multi_cell_list"``
#      - :py:class:`~jaxdem.colliders.multi_cell_list.DynamicMultiCellList`
#      - :math:`O(N \log N)`
#      - No
#      - With ``static=True``
#      - Wide size distributions (high polydispersity)
#    * - ``"neighbor_list"``
#      - :py:class:`~jaxdem.colliders.neighbor_list.NeighborList`
#      - :math:`O(NK)` between rebuilds
#      - Yes
#      - Yes
#      - Large systems and frictional/history simulations

import jax.numpy as jnp

import jaxdem as jdem
from jaxdem.utils.neighbor_list_sizing import (
    estimate_neighbor_capacity,
    measure_neighbor_candidates,
)

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]),
    rad=jnp.ones(3),
)

# %%
# Naive Collider
# ~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.naive.NaiveSimulator` (default) checks every
# pair :math:`(i, j)` with zero spatial-search setup or overflow risk. It is
# fast for small :math:`N` and stores pair history densely as ``history[i, j]``:

system_naive = jdem.System.create(state.shape, collider_type="naive")
state, system_naive = jdem.System.initialize(state, system_naive)
state_out, system_out = jdem.System.step(state, system_naive)
print("Naive forces:\n", state_out.force)

# %%
# Cell List and Multi-Cell List
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pass ``state=`` to :py:meth:`~jaxdem.system.System.create` so spatial
# colliders automatically infer defaults like ``cell_size`` from particle radii.
#
# * :py:class:`~jaxdem.colliders.cell_list.DynamicCellList` (``"cell_list"``)
#   bins particles into a uniform implicit grid and probes neighboring cells.
# * :py:class:`~jaxdem.colliders.multi_cell_list.DynamicMultiCellList`
#   (``"multi_cell_list"``) attaches an expandable bounding box to each cell so
#   a few large particles do not force a huge cell size on small ones.

state_p = jdem.State.create(
    pos=jnp.array([[1.0, 1.0], [2.5, 1.0], [5.0, 5.0]]),
    rad=jnp.full(3, 0.5),
)
system_cl = jdem.System.create(state=state_p, collider_type="cell_list")
system_mcl = jdem.System.create(state=state_p, collider_type="multi_cell_list")
print("CellList cell_size:", system_cl.collider.cell_size)
print("MultiCellList cell_size:", system_mcl.collider.cell_size)

# %%
# Fixed-Capacity (Static) Cell Traversal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# By default ``static=False`` uses dynamic ``while_loop`` traversal (no tile
# capacity limit, forward-mode AD only). Set ``static=True`` (or pass an
# explicit ``tile_capacity`` budget) for fixed-tile traversal that supports
# reverse-mode AD and avoids dynamic loops; always check
# :py:meth:`~jaxdem.system.System.check_overflow` afterward:

for collider_type in ("cell_list", "multi_cell_list"):
    system_static = jdem.System.create(
        state=state_p,
        collider_type=collider_type,
        collider_kw={"static": True},
    )
    _, system_static = jdem.System.initialize(state_p, system_static)
    system_static.check_overflow()
    print(collider_type, "tile_capacity:", system_static.collider.tile_capacity)

# %%
# Explicit Neighbor Queries
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Every collider exposes
# :py:meth:`~jaxdem.colliders.Collider.create_neighbor_list`, returning
# ``(state, system, neighbor_list, overflow)`` where ``neighbor_list`` has
# shape ``(N, max_neighbors)`` padded with ``-1``. Always check ``overflow``
# (if ``True``, increase ``max_neighbors``). Use
# :py:meth:`~jaxdem.colliders.Collider.create_cross_neighbor_list` when
# querying neighbors between two distinct particle sets:

state_p, system_cl, nl_cl, overflow_cl = system_cl.collider.create_neighbor_list(
    state_p, system_cl, cutoff=2.0, max_neighbors=8
)
print("Neighbor list (cutoff=2.0):\n", nl_cl)
print("Overflow:", bool(overflow_cl))

# %%
# Neighbor List Collider
# ~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.colliders.neighbor_list.NeighborList` caches candidate
# pairs built by a ``secondary_collider_type`` (default ``"cell_list"``; never
# another ``"neighbor_list"``) and rebuilds only when any particle moves more
# than ``skin / 2`` or domain geometry changes.
#
# Key parameters:
#
# * ``cutoff`` — search reach (automatically expanded if the force law needs more).
# * ``skin`` or ``skin_fraction`` (default ``0.05``) — buffer distance before
#   triggering a rebuild (pass at most one).
# * ``max_neighbors`` — average slot budget per particle (all ``N * max_neighbors``
#   slots are shared in one pool, so individual particles may exceed the average).
#
# .. warning::
#    Under ``jax.vmap``, per-replica rebuild branches can execute across the
#    batch unless caches stay synchronized; benchmark rebuild frequency for
#    your batch size.

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
_, system_nl = jdem.System.initialize(state_p, system_nl)
system_nl.check_overflow()
print(
    f"NeighborList: cutoff={float(system_nl.collider.cutoff)}, "
    f"skin={float(system_nl.collider.skin)}, "
    f"max_neighbors={system_nl.collider.max_neighbors}, "
    f"builds={int(system_nl.collider.n_build_times)}"
)

# %%
# Automatic Neighbor-List Capacity Sizing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rather than guessing ``max_neighbors`` by trial and error—especially for
# dense, polydisperse, or multi-sphere clump packings—use the host-side sizing
# helpers in :py:mod:`jaxdem.utils.neighbor_list_sizing` before launching a
# compiled run:
#
# * :py:func:`~jaxdem.utils.neighbor_list_sizing.measure_neighbor_candidates`
#   counts exact candidate degrees on the current snapshot without allocating a
#   neighbor buffer.
# * :py:func:`~jaxdem.utils.neighbor_list_sizing.estimate_neighbor_capacity`
#   scales and relaxes periodic bounding spheres to a target
#   ``packing_fraction`` (sampling clump orientations when needed) and returns
#   rounded capacities for both pooled :py:class:`~jaxdem.colliders.neighbor_list.NeighborList`
#   (``max_neighbors``) and dense :py:meth:`~jaxdem.colliders.Collider.create_neighbor_list`
#   queries (``query_max_neighbors``).

state_dense = jdem.utils.grid_state(n_per_axis=(4, 4), spacing=2.0)
state_dense.rad = jnp.where(jnp.arange(state_dense.N) % 2 == 0, 0.4, 0.6)
sys_probe = jdem.System.create(
    state=state_dense,
    domain_type="periodic",
    domain_kw={"box_size": jnp.array([8.0, 8.0])},
)

stats = measure_neighbor_candidates(state_dense, sys_probe, skin_fraction=0.1)
print(
    f"Current snapshot: mean_deg={stats.mean_degree:.2f}, "
    f"max_deg={stats.max_degree}, list_cutoff={stats.list_cutoff:.2f}"
)

est = estimate_neighbor_capacity(
    state_dense,
    sys_probe,
    packing_fraction=0.75,
    skin_fraction=0.1,
    safety_factor=1.25,
    extra_neighbors=2,
)
print(
    f"Recommended at phi=0.75: max_neighbors={est.max_neighbors} (pooled), "
    f"query_max_neighbors={est.query_max_neighbors} (dense)"
)

sys_sized = jdem.System.create(
    state=state_dense,
    domain_type="periodic",
    domain_kw={"box_size": jnp.array([8.0, 8.0])},
    collider_type="neighbor_list",
    collider_kw={
        "cutoff": est.cutoff,
        "skin": est.skin,
        "max_neighbors": est.max_neighbors,
    },
)
_, sys_sized = jdem.System.initialize(state_dense, sys_sized)
sys_sized.check_overflow()

# %%
# Overflow Checks and Editing State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Neighbor lists and static cell lists use fixed-size buffers. Call
# :py:meth:`~jaxdem.system.System.check_overflow` after initialization and
# simulation chunks to raise an error if any buffer overflowed.
#
# If you edit positions, radii, or particle counts manually on host:
#
# * Use :py:func:`jaxdem.colliders.invalidate_collider` when ``N`` and buffer
#   capacities are unchanged (forces a cache rebuild on the next query).
# * Use :py:func:`jaxdem.colliders.refresh_collider` when ``N`` or capacities
#   changed (pass ``reset_history=True`` only if pair identities changed).
#
# .. code-block:: python
#
#    system_nl.collider = jdem.colliders.refresh_collider(edited_state, system_nl.collider)

# %%
# Computing Pairwise Potential Energy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.colliders.Collider.compute_potential_energy` returns
# ``(state, system, potential_energy)`` so any cache rebuilds are preserved:

state_pe = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]), rad=jnp.ones(2)
)
system_pe = jdem.System.create(state_pe.shape, force_model_type="spring")
state_pe, system_pe, pe = system_pe.collider.compute_potential_energy(
    state_pe, system_pe
)
print("Total pairwise potential energy:", float(pe))

# %%
# For how the collider fits into the full per-step force and integration
# pipeline, see :doc:`../auto_examples/integrator_guide` and
# :doc:`../auto_examples/force_manager_guide`.
