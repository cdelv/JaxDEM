# Packing analysis

`analyze_packing` returns the same `PackingData` structure for the full packing
and the packing after iterative rattler removal. It uses the configured collider,
preserves contact history, and performs no minimization or file writes.

```python
from jaxdem.utils import analyze_packing

analysis = analyze_packing(
    state,
    system,
    clump_species_ids=species,
)
full = analysis.full
non_rattlers = analysis.non_rattlers

print(full.pressure, non_rattlers.pressure)
print(non_rattlers.packing_fraction)
print(analysis.rattler_ids)  # clump IDs in the input state
```

Each result contains its state and system, contact networks, physical and
effective packing fractions, pair potential energy, pressure, and coordination
statistics, plus the dense Hessian, eigenvalues, eigenvectors, and mode counts.
`original_sphere_ids[i]` maps sphere `i` back to the input state;
`original_clump_ids[I]` does the same for clump `I`. These maps refer directly
to the input, including after removal. Unoccupied clump slots contain -1.
`clump_ids` lists occupied IDs, and `clump_positions[I]` supplies the COM for
network drawing. Both results share one object when nothing is removed.

## Packed contact networks

`sphere_contacts` and `clump_contacts` contain directed `pair_ids`, `forces`,
`torques`, `displacements`, `mu`, and optional `pair_species`. Each array has one
row per active contact. There are no padding rows or neighbor capacity arguments.
An edge exists when a constituent force or torque is nonzero; cancellation of
the net clump force does not remove the edge.

```python
network = non_rattlers.clump_contacts
I = int(non_rattlers.clump_ids[0])
start, stop = map(int, network.row_offsets[I:I + 2])
neighbors = network.pair_ids[start:stop, 1]
forces = network.forces[start:stop]
mu = network.mu[start:stop]
```

Sphere networks also contain signed `overlap`: the radius sum minus the current
minimum-image separation. Clump networks contain `sphere_counts` with shape
`(K, 2)`, counting distinct participating spheres on each side, and
`contact_counts` with shape `(K,)`, counting constituent directed contacts.
All quantities use the same pair ordering.

Torques in both networks are moments about the **source clump COM**, including
intrinsic moments and member lever arms. Displacements point from destination
to source: between sphere centers in the sphere network and clump COMs in the
clump network. For drawing, the destination image is
`source_position - displacement`. Reciprocal interactions have two directed
entries; select `pair_ids[:, 0] < pair_ids[:, 1]` to draw each edge once.

`mu` is the magnitude ratio of tangential to normal net force along the relevant
pair axis. Zero net force gives zero, a purely tangential nonzero force gives
infinity, and a nonzero force at coincident centers gives NaN. There is no
validity flag or denominator regularization.

## Species comparisons

Supply one integer label per occupied input clump in ascending clump-ID order.
These labels are for analysis and do not change `state.species_id` or force
routing. Both networks carry source/destination labels in `pair_species`;
sphere contacts inherit their parent clumps' labels. Omitting the input leaves
`pair_species` as None. Species masks are derived when needed:

```python
network = non_rattlers.clump_contacts
labels = network.pair_species
same_species = labels[:, 0] == labels[:, 1]
unique = network.pair_ids[:, 0] < network.pair_ids[:, 1]
mu_within_species = network.mu[same_species & unique]
inter_species_contact_count = (~same_species & unique).sum()
```

## Coordination, packing fraction, and spectra

`sphere_contact_counts`, `vertex_contact_counts`, and `clump_contact_counts`
count force-bearing contacts per sphere, constituent contacts per clump, and
distinct contacting clumps per clump. Clump arrays use clump IDs as indices.
The means include occupied clumps with zero contacts. Torque-only edges remain
available in the networks but do not increment these force-bearing counts.

`effective_packing_fraction` assigns each clump a disk in 2D or sphere in 3D,
with diameter equal to its mean contacting-clump COM distance. A zero mean
distance uses the clump's stored physical volume. This contact-distance estimate
is distinct from `packing_fraction`, which uses the stored physical volumes.
Pressure is the constituent-sphere contact virial divided by box volume and
dimension; potential energy includes the non-bonded pair law. Managed external
energies, kinetic stress, and boundary reactions are excluded from those summaries.

`coordinates="auto"` uses translations for single spheres at their COMs and
translations plus rotations for clumps. Set `coordinates="clump"` explicitly to
include single-sphere rotational coordinates. This convention selects the
rattler helper and applies to both results. `zc`, `check_contact_rank`, and
`contact_rank_tol` configure rattler pruning. Bonded models require topology
remapping and are not supported by this utility.

`satisfies_isostatic_count` compares mean vertex coordination with
`2 * (G*dof - global_modes) / G`. It does not certify mechanical stability.
`global_modes` defaults to the spatial dimension, corresponding to translations
of a periodic packing; other constraints require an explicit choice. Fixed
bodies remain in the coordinates and counts.

Both configurations include the dense `hessian`, ascending `eigenvalues`,
`eigenvectors` stored as columns, and zero/negative mode counts. All quantities
are calculated on every call; select what to save afterwards.

For clumps, `rotation_scale` supplies one positive length per occupied input
clump, in ascending ID order, for coordinates `(delta r_c, R*omega)`. Scales are
mapped to the surviving clumps. Clump Hessian blocks follow `clump_ids` order;
sphere Hessians follow the sphere array. Storage is quadratic in coordinate count.
Spectra characterize the pair potential, including its rotational terms, and
do not include dissipative or history-dependent force response.

`zero_mode_rel_gap` controls gap-based numerical-zero classification;
`zero_mode_atol` adds an absolute threshold. Negative-mode counts exclude
numerical zeros. An all-zero Hessian has every mode classified as zero.

An empty non-rattler packing returns the same result type, with empty networks,
zero pressure and packing fractions, NaN mean coordination, and a False count
criterion. Hessian and eigenvector arrays have shape `(0, 0)`, eigenvalues have
shape `(0,)`, and both mode counts are zero.

## Saving and loading

Save the complete analysis with the built-in HDF5 utilities. Loading restores
the `PackingAnalysis` object, including both states and systems, networks,
species labels, and spectra. Named attributes remain available:

```python
from jaxdem.utils import h5

h5.save(analysis, "results.h5")
loaded = h5.load("results.h5")
network = loaded.non_rattlers.clump_contacts
print(network.pair_ids, network.mu)
```

To save selected quantities, pass a dictionary instead:

```python
h5.save(
    {
        "pressure": non_rattlers.pressure,
        "contacts": non_rattlers.clump_contacts,
        "original_clump_ids": non_rattlers.original_clump_ids,
    },
    "selected_results.h5",
)
```
