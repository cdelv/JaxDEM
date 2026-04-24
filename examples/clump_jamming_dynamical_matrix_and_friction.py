# %%
r"""Jammed rigid clumps: contacts, rattlers, dynamical matrix, and friction
==========================================================================

This is the rigid-clump counterpart of
:ref:`sphx_glr_auto_examples_sphere_jamming_rattlers_dynamical_matrix.py`.
Instead of point-like spheres we build a 2D packing of rigid clumps of
mixed size and shape (a few different vertex-counts and bounding radii),
jam it, and analyze the dynamical matrix in the full rigid-body
coordinate space :math:`(\delta r, \omega)` per clump.

A rigid body in 2D has :math:`d_f = \dim + \dim_{\rm rot} = 3` degrees
of freedom (:math:`d_f = 6` in 3D), and a rattler with ``k``
force-bearing vertex contacts contributes :math:`\max(0, d_f - k)` zero
modes to the hessian — the directions perpendicular to its contact
constraints. The total zero-mode count is

.. math::

    n_{\rm zero} \;=\; \dim \;+\; \sum_{\rm rattlers} \max(0, d_f - k_i)

The leading :math:`\dim` is the rigid-translation null space of the
pair potential; in a periodic box the global rotation is **not** a
zero mode, because rotating the configuration without rotating the
box changes the minimum-image distances. A rattler with exactly
:math:`d_f` generic contacts gives no floppy modes (though the default
``zc = d_f + 1`` still flags it as a rattler, because at finite overlap
the tangential softening can make it mechanically unstable).

After removing the rattlers, only the :math:`\dim` global translational
zero modes remain.

As an encore we compute the per-clump-pair friction coefficient
:math:`\mu_{IJ}` for every contacting clump pair using
:func:`~jaxdem.utils.contacts.compute_clump_pair_friction`, which
decomposes the total contact force between two clumps along the
COM-to-COM axis and reports the ratio of its tangential to normal
magnitude.
"""

# %%
# Imports
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

from jaxdem.utils.particleCreation import build_ga_system
from jaxdem.utils.jamming import bisection_jam
from jaxdem.utils.contacts import (
    count_clump_contacts,
    count_vertex_contacts,
    compute_clump_pair_friction,
    get_clump_rattler_ids,
    remove_rattlers,
)
from jaxdem.utils.dynamicalMatrix import clump_non_bonded_hessian, zero_mode_mask


# %%
# Parameters
# -----------
# 20 bidisperse 2D clumps: 10 small (bounding radius 0.5, 3 vertices)
# and 10 large (bounding radius 0.7, 7 vertices). All vertex spheres
# share the same ``asperity_radius = 0.3`` so differences in behavior
# come from the rigid-body shape, not from disparate vertex sizes.
# ``particle_radius`` is each clump's bounding-sphere radius; the
# asperities sit on a circle of radius ``particle_radius − asperity_radius``
# and overlap internally, which is harmless because intra-clump contacts
# are skipped by the collider.
#
# We use a bidisperse mix of shapes and sizes for the same reason the
# sphere example does: to suppress the crystalline packings that a
# monodisperse system would fall into, so the zero-mode spectrum is
# cleanly "global translations + rattlers" without extra soft shear
# modes from crystal order.

rng = np.random.default_rng(seed=0)
dim = 2
n_small = 10
n_large = 10
asperity_radius = 0.3

vertex_counts = [3] * n_small + [7] * n_large
particle_radii = [0.5] * n_small + [0.7] * n_large


# %%
# Build and jam
# -------------
# :func:`~jaxdem.utils.particleCreation.build_ga_system` creates each
# clump by placing ``nv`` asperity spheres on the clump's bounding-sphere
# surface via the Thomson problem (which in 2D simply spaces the
# vertices evenly around a circle), then quasistatically compresses the
# packing to the target body-volume packing fraction.
# :func:`~jaxdem.utils.jamming.bisection_jam` drives the system from
# there to its nearest jammed state. For a clump system we need both a
# translational and a rotational FIRE minimizer, so we set
# ``linear_integrator_type="linearfire"`` and
# ``rotation_integrator_type="rotationfire"``.

state, system = build_ga_system(
    particle_radii=particle_radii,
    vertex_counts=vertex_counts,
    asperity_radius=asperity_radius,
    phi=0.4,  # loose initial target (below jamming for mixed dimers/trimers)
    dim=dim,
    particle_type="clump",
    core_type="phantom",  # treat the particles as if they were solid when calculating their properties
    domain_type="periodic",
    randomize_orientation=True,
    n_property_samples=10_000_000,  # we need at least 10m samples to get good clump properties
    compression_step=1e-2,
    max_n_min_steps_per_outer=50_000,
    dt=1e-2,
    linear_integrator_type="linearfire",
    rotation_integrator_type="rotationfire",
    collider_type="naive",
    seed=int(rng.integers(0, 2**31 - 1)),
)

state, system, phi_jam, pe_jam = bisection_jam(state, system)
print(f"Jammed: phi = {float(phi_jam):.6f}, residual PE = {float(pe_jam):.3e}")


# %%
# Count contacts: vertex vs clump
# -------------------------------
# There are two notions of "contact" for a clump system:
#
# * **Vertex contacts**: each individual sphere–sphere touch counts as
#   one contact. Two clumps may touch at two vertex pairs at once; that
#   is two vertex contacts.
# * **Clump contacts**: a contact is between two clumps, regardless of
#   how many vertex pairs are involved. Two clumps touching at two
#   vertex pairs still count as a single clump contact.
#
# Isostaticity is about constraints on the rigid-body degrees of
# freedom, and every vertex–vertex touch is one independent distance
# constraint — so the vertex count is what enters the Maxwell counting,
# and what :func:`~jaxdem.utils.contacts.get_clump_rattler_ids` uses
# internally for the default coordination threshold
# ``zc = dim + dim_rot + 1``. The clump count is the more intuitive
# "how many neighbors does this body have" quantity and is useful for
# visualization and coarse statistics.

state, system, vertex_contacts_per_clump = count_vertex_contacts(state, system)
state, system, clump_contacts_per_clump = count_clump_contacts(state, system)
N_clumps = int(state.clump_id.max()) + 1
N_vertices = int(state.N)
print(f"{N_clumps} clumps, {N_vertices} vertex spheres")
print(
    f"vertex contacts: total = {int(np.sum(vertex_contacts_per_clump)) // 2}, "
    f"mean per clump = {float(np.mean(vertex_contacts_per_clump)):.2f}"
)
print(
    f"clump contacts : total = {int(np.sum(clump_contacts_per_clump)) // 2}, "
    f"mean per clump = {float(np.mean(clump_contacts_per_clump)):.2f}"
)


# %%
# Rattlers
# --------
# A rigid clump with ``d_f = dim + dim_rot`` or fewer force-bearing
# vertex contacts cannot be mechanically stable: each contact
# contributes a positive normal stiffness and a negative tangential
# stiffness (``-k s / r`` from the spring potential's tangential
# softening), and only when the number of contacts strictly exceeds
# ``d_f`` can the sum of contributions be positive-definite for generic
# contact angles. The standard rattler threshold is therefore
# ``zc = d_f + 1 = dim + dim_rot + 1``.
# :func:`~jaxdem.utils.contacts.get_clump_rattler_ids` iteratively
# removes any clump with fewer than ``zc`` vertex contacts and re-checks
# the remaining graph (since removing one rattler may leave its
# neighbors under-coordinated), continuing until the set stabilizes.

state, system, rattler_ids, non_rattler_ids = get_clump_rattler_ids(state, system)
n_rattlers = int(rattler_ids.shape[0])
print(f"Rattlers: {n_rattlers} / {N_clumps}")

# The number of zero modes each rattler contributes depends on its
# force-bearing vertex-contact count ``k``: with ``k < d_f`` it has
# ``d_f - k`` floppy directions, while with ``k = d_f`` generic
# contacts it is rank-constrained (no floppy modes, though still
# mechanically marginal). We read each rattler's contact count off the
# per-clump array computed above.
vc_per_clump = np.asarray(vertex_contacts_per_clump)
rattler_contacts = vc_per_clump[np.asarray(rattler_ids)]
print(f"Force-bearing vertex contacts per rattler: {rattler_contacts.tolist()}")


# %%
# Dynamical matrix
# ----------------
# Now we will calculate the dynamical matrix for the entire system,
# including the rattler clumps, using
# :func:`~jaxdem.utils.dynamicalMatrix.clump_non_bonded_hessian`. It
# takes the hessian of the pair potential with respect to each clump's
# generalized coordinates :math:`(\delta r_c, \omega)` — a
# ``d_f``-dimensional coordinate per clump (``d_f = 3`` in 2D, ``6`` in
# 3D) — so for ``N_clumps`` clumps the matrix is
# ``(d_f N_clumps, d_f N_clumps)``. We expect each rattler with ``k``
# force-bearing vertex contacts to contribute ``max(0, d_f - k)`` zero
# modes (the directions orthogonal to its contact constraints), giving

#     n_zero  =  dim  +  Σ_rattlers max(0, d_f - k_i)

# on top of the ``dim`` global translational zero modes. The global
# translations arise because the potential only depends on differences
# between particle positions; a background potential would lift them.

state, system, H = clump_non_bonded_hessian(state, system)
H_np = np.asarray(H)
# Make exactly symmetric before eigendecomposition (autograd symmetry
# holds to roundoff; symmetrize to avoid complex eigenvalues from any
# floating-point asymmetry).
H_np = 0.5 * (H_np + H_np.T)
eigenvalues = np.sort(np.linalg.eigvalsh(H_np))

# Zero modes will not come out exactly zero from the eigendecomposition —
# they land around machine precision times the largest eigenvalue. In a
# jammed packing there is typically a very large gap (many orders of
# magnitude) between these numerical zeros and the smallest truly
# finite mode, so we can identify them by finding the gap.
# :func:`~jaxdem.utils.dynamicalMatrix.zero_mode_mask` does this for us
# and returns a boolean mask we can apply to both eigenvalues and
# eigenvectors.

zero_mask = np.asarray(zero_mode_mask(eigenvalues))
n_zero = int(zero_mask.sum())

print("\nEigenvalue spectrum (low end):")
for i, lam in enumerate(eigenvalues[:12]):
    mark = "  (zero)" if zero_mask[i] else ""
    print(f"  λ[{i:3d}] = {lam: .3e}{mark}")
print("  ...")
print(f"  λ[-1] = {eigenvalues[-1]: .3e}")

rot_dim = 1  # 2D
d_f = dim + rot_dim  # 3 DOFs per clump in 2D
print(f"\n# zero modes   : {n_zero}")
rattler_floppy = int(np.sum(np.maximum(0, d_f - rattler_contacts)))
expected_zero = dim + rattler_floppy
print(
    f"# expected     : {expected_zero} = {dim} (global translations) + "
    f"{rattler_floppy} (Σ max(0, d_f - k_i) over rattlers)"
)
# assert n_zero == expected_zero, (
#     f"zero-mode count {n_zero} != expected {expected_zero}"
# )


# %%
# Remove rattlers and re-analyze
# ------------------------------
# We will now remove the rattler clumps using
# :func:`~jaxdem.utils.contacts.remove_rattlers`, which drops their
# vertex spheres from the state and returns a matching system
# re-initialized for the reduced particle count — no manual system
# reconstruction needed. (For a clump system the rattler IDs are
# already clump IDs, so we pass ``rattler_ids`` directly.)

state_nr, system_nr = remove_rattlers(state, system, rattler_ids)
print(f"After rattler removal: {int(state_nr.N)} vertex spheres "
      f"in {int(state_nr.clump_id.max()) + 1} clumps")


# %%
# Recompute the dynamical matrix
# ------------------------------
# With the rattlers gone we expect only the ``dim`` global translational
# zero modes to remain.

state_nr, system_nr, H_nr = clump_non_bonded_hessian(state_nr, system_nr)
H_nr_np = np.asarray(H_nr)
H_nr_np = 0.5 * (H_nr_np + H_nr_np.T)
eigenvalues_nr = np.sort(np.linalg.eigvalsh(H_nr_np))

zero_mask_nr = np.asarray(zero_mode_mask(eigenvalues_nr))
n_zero_nr = int(zero_mask_nr.sum())

print("\nAfter removing rattlers — eigenvalue spectrum (low end):")
for i, lam in enumerate(eigenvalues_nr[:8]):
    mark = "  (zero)" if zero_mask_nr[i] else ""
    print(f"  λ[{i:3d}] = {lam: .3e}{mark}")
print("  ...")
print(f"  λ[-1] = {eigenvalues_nr[-1]: .3e}")

print(f"\n# zero modes (no rattlers)  : {n_zero_nr}")
print(f"# expected                   : {dim} (global translations only)")
assert n_zero_nr == dim, (
    f"post-rattler zero-mode count {n_zero_nr} != {dim}"
)


# %%
# Friction in every contact
# -------------------------
# For each contacting clump pair :math:`(I, J)`,
# :func:`~jaxdem.utils.contacts.compute_clump_pair_friction` sums the
# per-vertex contact forces between spheres of :math:`I` and spheres of
# :math:`J`, decomposes the total along the COM-to-COM direction, and
# reports the ratio :math:`\mu_{IJ} = |F^t_{IJ}| / |F^n_{IJ}|`. For a
# single sphere-sphere contact along the center line :math:`\mu = 0`.
# Off-axis multi-vertex contacts, which only arise for non-spherical
# clumps, can give :math:`\mu > 0`: this is the purely geometric
# "friction" a rigid clump exhibits due to its shape, with no
# tangential force law involved.

state_nr, system_nr, F_clumps, mu, contact_mask = compute_clump_pair_friction(
    state_nr, system_nr
)
mu_np = np.asarray(mu)
mask_np = np.asarray(contact_mask)

# Extract upper-triangular entries of contacting clump pairs.
ij = np.argwhere(np.triu(mask_np, k=1))
mu_values = mu_np[ij[:, 0], ij[:, 1]]
print(f"\n{len(mu_values)} clump-clump contacts in the rattler-free contact network")
print(f"μ statistics:")
print(f"  min    = {float(np.min(mu_values)):.4f}")
print(f"  mean   = {float(np.mean(mu_values)):.4f}")
print(f"  median = {float(np.median(mu_values)):.4f}")
print(f"  max    = {float(np.max(mu_values)):.4f}")
