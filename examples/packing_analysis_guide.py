"""Packing analysis and contact networks
=====================================

Analyze a small 2D clump configuration, compare full and non-rattler results,
select contacts by species, and draw the contact network. This configuration
is prescribed rather than minimized; a coordination count does not establish
mechanical stability.
"""

# %%
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import jax.numpy as jnp
import numpy as np

import jaxdem as jd
from jaxdem.utils import analyze_packing

# %%
# Four two-sphere clumps form a contacting cluster. The fifth is isolated.
centers = np.array([[1.0, 1.0], [1.8, 1.0], [1.0, 1.8], [1.8, 1.8], [4.0, 4.0]])
offsets = np.array([[0.0, -0.15], [0.0, 0.15]])
# Union area of two radius-0.5 disks with centers separated by 0.3.
clump_area = 0.5 * np.pi - 0.5 * np.arccos(0.3) + 0.15 * np.sqrt(1 - 0.3**2)
state = jd.State.create(
    pos=np.repeat(centers, 2, axis=0),
    pos_p=np.tile(offsets, (5, 1)),
    rad=jnp.full(10, 0.5),
    volume=jnp.full(10, clump_area),
    clump_id=np.repeat(np.arange(5), 2),
)
system = jd.System.create(
    state=state,
    collider_type="NeighborList",
    collider_kw={"cutoff": 1.0, "skin": 0.1, "max_neighbors": 10},
    domain_type="periodic",
    domain_kw={"box_size": jnp.full(2, 6.0)},
)

analysis = analyze_packing(state, system, clump_species_ids=[0, 0, 1, 1, 2])
full = analysis.full
non_rattlers = analysis.non_rattlers
print("Rattler clump IDs:", np.asarray(analysis.rattler_ids))
print("Full packing fraction:", float(full.packing_fraction))
print("Non-rattler packing fraction:", float(non_rattlers.packing_fraction))
print("Mean constituent contacts:", non_rattlers.mean_vertex_contacts)
print("Count criterion satisfied:", non_rattlers.satisfies_isostatic_count)
print("Negative energy modes:", non_rattlers.negative_mode_count)
assert np.array_equal(analysis.rattler_ids, [4])

# %%
# Pair statistics share one ordering. Species comparisons need no stored masks.
network = non_rattlers.clump_contacts
pairs = np.asarray(network.pair_ids)
species = np.asarray(network.pair_species)
unique = pairs[:, 0] < pairs[:, 1]
same_species = species[:, 0] == species[:, 1]
print("Within-species edges:", int(np.sum(unique & same_species)))
print("Between-species edges:", int(np.sum(unique & ~same_species)))
print("Between-species mu:", np.asarray(network.mu)[unique & ~same_species])
print("Sphere participation per edge:", np.asarray(network.sphere_counts)[unique])

# %%
# Packed rows give the contacts of one clump directly.
clump_id = int(non_rattlers.clump_ids[0])
start, stop = map(int, network.row_offsets[clump_id : clump_id + 2])
print("Neighbors of clump", clump_id, ":", pairs[start:stop, 1])

# %%
# Displacements identify the correct periodic destination image. The same
# construction works for sphere contacts using ``state.pos`` as the positions.
starts = np.asarray(non_rattlers.clump_positions)[pairs[unique, 0]]
ends = starts - np.asarray(network.displacements)[unique]
segments = np.stack((starts, ends), axis=1)
strength = np.linalg.norm(np.asarray(network.forces)[unique], axis=1)
widths = 1.0 + 3.0 * strength / strength.max()

fig, ax = plt.subplots(figsize=(5, 5))
for position, radius in zip(
    np.asarray(full.state.pos), np.asarray(full.state.rad), strict=True
):
    ax.add_patch(
        Circle(position, radius, facecolor="lightgray", edgecolor="gray", alpha=0.4)
    )
ax.add_collection(LineCollection(segments, linewidths=widths, colors="tab:blue"))
ax.scatter(centers[:, 0], centers[:, 1], c=["tab:orange"] * 4 + ["tab:red"], zorder=3)
ax.set(xlim=(0, 5), ylim=(0, 5), xlabel="x", ylabel="y", title="Clump contact network")
ax.set_aspect("equal")
fig.tight_layout()
