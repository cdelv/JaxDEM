# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Find rattlers and draw a contact network
===========================================

Analyze a snapshot with :py:func:`~jaxdem.utils.post_processing.analyze_packing`
to prune rattlers, compare packing fraction and contact virial pressure, count
within- vs between-species contacts, and plot both the 2D contact-force network
and signed energy-Hessian spectrum.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import jaxdem as jdem
from jaxdem.utils import analyze_packing

# %%
# Build Three Contacting Clumps and One Isolated Rattler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each clump consists of two overlapping disks; supply their exact union area
# as ``volume`` on every member so
# :py:attr:`~jaxdem.utils.post_processing.PackingData.packing_fraction` does
# not double-count internal overlap:

centers = np.array([[1.0, 1.0], [1.8, 1.0], [1.4, 1.7], [4.0, 4.0]])
offsets = np.array([[0.0, -0.15], [0.0, 0.15]])
clump_area = 0.5 * np.pi - 0.5 * np.arccos(0.3) + 0.15 * np.sqrt(1 - 0.3**2)

state = jdem.State.create(
    pos=np.repeat(centers, 2, axis=0),
    pos_p=np.tile(offsets, (4, 1)),
    rad=jnp.full(8, 0.5),
    volume=jnp.full(8, clump_area),
    clump_id=np.repeat(np.arange(4), 2),
)
system = jdem.System.create(
    state=state,
    collider_type="naive",
    domain_type="periodic",
    domain_kw={"box_size": jnp.full(2, 6.0)},
)

# %%
# Run :py:func:`~jaxdem.utils.post_processing.analyze_packing`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:func:`~jaxdem.utils.post_processing.analyze_packing` returns matching
# :py:class:`~jaxdem.utils.post_processing.PackingData` snapshots for the full
# configuration and the pruned non-rattler network (including dense Hessian
# eigenvalues, so prefer individual contact helpers for very large packings).
# Passing ``clump_species_ids`` labels bodies for species-resolved contact
# counts without changing the force law:

analysis = analyze_packing(state, system, clump_species_ids=[0, 0, 1, 2])
full = analysis.full
remaining = analysis.non_rattlers
full.system.check_overflow()
remaining.system.check_overflow()

print("Removed rattler clump IDs:", np.asarray(analysis.rattler_ids))
print(
    f"Packing fraction: {float(full.packing_fraction):.4f} -> "
    f"{float(remaining.packing_fraction):.4f}"
)
print(
    f"Contact virial pressure: {float(full.pressure):.4f} -> "
    f"{float(remaining.pressure):.4f} (unchanged: rattler carries no load)"
)

# %%
# Count Species Contacts and Plot Force Network + Spectrum
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:attr:`~jaxdem.utils.post_processing.PackingData.clump_contacts` stores
# directed edges ``(I, J)`` and ``(J, I)``; filter ``I < J`` for unique edges.
# Draw periodic edges from ``x_I`` to ``x_I - r_IJ`` using
# :py:attr:`~jaxdem.utils.post_processing.ClumpContactNetwork.displacements`
# so minimum-image contacts never stretch across the box. Use
# :py:attr:`~jaxdem.utils.post_processing.PackingData.original_clump_ids` to
# map pruned indices back to the input state:

network = remaining.clump_contacts
pairs = np.asarray(network.pair_ids)
species = np.asarray(network.pair_species)
unique = pairs[:, 0] < pairs[:, 1]
same_species = species[:, 0] == species[:, 1]

within = int(np.sum(unique & same_species))
between = int(np.sum(unique & ~same_species))
mean_z = 2 * int(unique.sum()) / len(remaining.clump_ids)
print(
    f"Edges: {within} within-species, {between} between-species | "
    f"mean z={mean_z:.2f} | negative modes={remaining.negative_mode_count}"
)

starts = np.asarray(remaining.clump_positions)[pairs[unique, 0]]
ends = starts - np.asarray(network.displacements)[unique]
segments = np.stack((starts, ends), axis=1)
strength = np.linalg.norm(np.asarray(network.forces)[unique], axis=1)
widths = 1.5 + 3.5 * strength / max(float(strength.max(initial=0.0)), 1e-12)
edge_colors = np.where(same_species[unique], "tab:blue", "tab:orange")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")

# Left: 2D clump disks + minimum-image force network
for pos_xy, r, cid in zip(
    np.asarray(full.state.pos), np.asarray(full.state.rad), np.asarray(full.state.clump_id)
):
    color = "#b0bec5" if cid in np.asarray(analysis.rattler_ids) else "#cfd8dc"
    axes[0].add_patch(plt.Circle(pos_xy, r, fc=color, ec="#455a64", lw=1.0))
axes[0].add_collection(LineCollection(segments, linewidths=widths, colors=edge_colors, zorder=3))
axes[0].set(
    xlim=(0, 6),
    ylim=(0, 6),
    aspect="equal",
    xlabel="x",
    ylabel="y",
    title="Contact network (blue: same species, orange: cross)",
)
axes[0].grid(alpha=0.15)

# Right: signed energy-Hessian spectrum
for data, label, color in (
    (full, "Full packing", "#7b8794"),
    (remaining, "Rattler removed", "#258685"),
):
    evals = np.asarray(data.eigenvalues)
    axes[1].plot(np.arange(evals.size), evals, "o-", ms=4, label=label, color=color)
axes[1].axhline(0, color="0.4", lw=0.8)
axes[1].set(
    xlabel="Sorted eigenvalue index",
    ylabel="Signed energy curvature",
    yscale="symlog",
    title="Hessian spectrum (prescribed non-minimum)",
)
axes[1].legend(fontsize="small")
axes[1].grid(axis="y", alpha=0.2)
plt.show()
