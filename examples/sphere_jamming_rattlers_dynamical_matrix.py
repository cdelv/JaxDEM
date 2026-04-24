# %%
"""Jammed bidisperse packing: contacts, rattlers, and the dynamical matrix
=========================================================================

This example ties together :mod:`jaxdem.utils.contacts` and
:mod:`jaxdem.utils.dynamicalMatrix` in the classic jamming / isostaticity
setting. We build a small bidisperse 2D packing, compress it to its nearest
jammed state, count the inter-particle contacts, identify the rattlers
(particles with fewer force-bearing contacts than mechanical stability
requires), and compute the dynamical matrix.

The zero-mode count of the dynamical matrix tracks the system's floppy
directions. A rattler with ``k`` force-bearing contacts contributes
``max(0, dim - k)`` zero modes to the hessian (the directions
perpendicular to its contact constraints), so

    n_zero  =  dim  +  Σ_rattlers max(0, dim - k_i)

The leading ``dim`` is the rigid-translation null space of the pair
potential (it depends only on differences). A rattler with zero
contacts gives the full ``dim`` zero modes; one with a single contact
gives ``dim - 1``; one with ``dim`` generic contacts gives none (though
the default ``zc = dim + 1`` still flags it as a rattler because
finite-overlap tangential softening can make it mechanically unstable).

After we remove the rattlers, the isostatic count reduces to just the
``dim`` global translations — the remaining network is rigid.
"""

# %%
# Imports
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

from jaxdem.utils.particleCreation import build_sphere_system
from jaxdem.utils.jamming import bisection_jam
from jaxdem.utils.contacts import (
    count_vertex_contacts,
    get_sphere_rattler_ids,
    remove_rattlers,
)
from jaxdem.utils.dynamicalMatrix import non_bonded_hessian, zero_mode_mask


# %%
# Parameters
# -----------
# Small bidisperse 2D packing: 20 small + 20 large spheres, radius ratio 1.4,
# smallest radius 0.5. The seed is fixed so the whole pipeline is reproducible.
#
# We use a bidisperse mix because monodisperse 2D disks crystallize.
# A 1 : 1.4 radius ratio with roughly equal populations is the standard 
# jamming-literature choice: large enough to fully suppress 
# crystallization, small enough to keep the packing structurally isotropic.

rng = np.random.default_rng(seed=20260422)
dim = 2
n_small = 20
n_large = 20
r_small = 0.5
r_large = r_small * 1.4

radii = np.concatenate(
    [np.full(n_small, r_small), np.full(n_large, r_large)]
)
N = int(radii.shape[0])

# %%
# Build and jam
# -------------
# :func:`~jaxdem.utils.particleCreation.build_sphere_system` places the
# spheres uniformly and quasistatically compresses to the given target
# phi. :func:`~jaxdem.utils.jamming.bisection_jam` then drives the
# system to its nearest jammed state via bisection on the packing
# fraction — it handles the minimization internally, so no extra
# FIRE/minimize call is needed. We use the naive collider because the
# system is small.

state, system = build_sphere_system(
    particle_radii=radii.tolist(),
    phi=0.5,  # loose initial target (below 2D jamming ~0.84)
    dim=dim,
    seed=int(rng.integers(0, 2**31 - 1)),
    collider_type="naive",
    domain_type="periodic",
    linear_integrator_type="linearfire",  # bisection_jam uses FIRE internally
    rotation_integrator_type="",
    dt=1e-2,
)

state, system, phi_jam, pe_jam = bisection_jam(state, system)  # jam the system
print(f"Jammed: phi = {float(phi_jam):.6f}, residual PE = {float(pe_jam):.3e}")


# %%
# Count contacts
# --------------
# A contact is a sphere pair with nonzero force.
# :func:`~jaxdem.utils.contacts.count_vertex_contacts` returns the
# force-bearing contact count for each clump (for a pure sphere system
# each sphere is its own clump of size 1, so this is just the contact
# count per sphere). Each unique sphere-pair contact is counted once
# per endpoint, so summing over clumps and dividing by 2 gives the
# number of distinct inter-particle contacts.

state, system, contacts_per_sphere = count_vertex_contacts(state, system)
contacts_per_sphere = np.asarray(contacts_per_sphere)
n_contacts = int(contacts_per_sphere.sum()) // 2
print(f"{N} particles, {n_contacts} inter-particle contacts")


# %%
# Rattlers
# --------
# A sphere with ``dim`` or fewer force-bearing contacts cannot be
# mechanically stable: each contact contributes a positive normal
# stiffness and a negative tangential stiffness (``-k s / r`` from the
# spring potential's tangential softening), and only when the number
# of contacts strictly exceeds ``dim`` can the sum of contributions
# be positive-definite for generic contact angles. The standard
# rattler threshold is therefore ``zc = dim + 1``.
# :func:`~jaxdem.utils.contacts.get_sphere_rattler_ids` iteratively
# removes any particle with fewer than ``zc`` contacts and re-checks
# the remaining graph (since removing one rattler may leave its
# neighbors under-coordinated), continuing until the set stabilizes.

state, system, rattler_ids, non_rattler_ids = get_sphere_rattler_ids(
    state, system
)
n_rattlers = int(rattler_ids.shape[0])
print(f"Rattlers: {n_rattlers} / {N}")

# The number of zero modes each rattler contributes depends on its
# force-bearing contact count ``k``: with ``k < dim`` contacts it has
# ``dim - k`` floppy directions, while with ``k = dim`` contacts it is
# generically rank-constrained (no floppy modes, though possibly still
# mechanically unstable). We read each rattler's contact count off the
# same per-sphere array we just printed.
rattler_contacts = contacts_per_sphere[np.asarray(rattler_ids)]
print(f"Force-bearing contacts per rattler: {rattler_contacts.tolist()}")


# %%
# Dynamical matrix
# ----------------
# Now we will calculate the dynamical matrix for the entire system,
# including the rattler particles, using
# :func:`~jaxdem.utils.dynamicalMatrix.non_bonded_hessian`. We expect
# each rattler with ``k`` force-bearing contacts to contribute
# ``max(0, dim - k)`` zero modes (the directions orthogonal to its
# contact constraints), giving

#     n_zero  =  dim  +  Σ_rattlers max(0, dim - k_i)

# on top of the ``dim`` zero modes from global translations. The
# global translations arise because the potential only depends on
# the difference between particle positions — adding a background
# potential would lift them.

state, system, H = non_bonded_hessian(state, system)
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

print(f"\n# zero modes   : {n_zero}")
rattler_floppy = int(np.sum(np.maximum(0, dim - rattler_contacts)))
expected_zero = dim + rattler_floppy
print(
    f"# expected     : {expected_zero} = {dim} (global translations) + "
    f"{rattler_floppy} (Σ max(0, dim - k_i) over rattlers)"
)
assert n_zero == expected_zero, (
    f"zero-mode count {n_zero} != expected {expected_zero}"
)


# %%
# Remove rattlers and re-analyze
# ------------------------------
# We will now remove the rattlers using
# :func:`~jaxdem.utils.contacts.remove_rattlers`, which drops the
# rattler spheres from the state and returns a matching system
# re-initialized for the reduced particle count — no manual system
# reconstruction needed.

rattler_clump_ids = state.clump_id[rattler_ids]
state_nr, system_nr = remove_rattlers(state, system, rattler_clump_ids)
print(f"After rattler removal: {int(state_nr.N)} particles")


# %%
# Recompute the dynamical matrix
# ------------------------------
# With the rattlers gone we expect only the ``dim`` global translational
# zero modes to remain; the rest of the spectrum should be identical to
# the finite modes of the full system.

state_nr, system_nr, H_nr = non_bonded_hessian(state_nr, system_nr)
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
print("\nWith the rattlers removed, every zero mode of the dynamical matrix "
      "corresponds to a global translation of the packing.")
