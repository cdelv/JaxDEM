# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Jammed rigid clumps: contacts, Hessian, and geometric friction
==================================================================

This is the rigid-clump counterpart of
:doc:`sphere_jamming_rattlers_dynamical_matrix`. It focuses on the APIs
specific to multi-sphere rigid bodies (which carry :math:`d_f = 3` degrees of
freedom per body in 2D and :math:`6` in 3D):

1. **Vertex vs clump contacts:**
   :func:`~jaxdem.utils.contacts.count_vertex_contacts` counts constituent
   sphere contacts, whereas :func:`~jaxdem.utils.contacts.count_clump_contacts`
   counts distinct neighboring bodies.
2. **Rigid-body Hessian:**
   :func:`~jaxdem.utils.dynamical_matrix.clump_non_bonded_hessian` builds the
   :math:`(d_f N_b \times d_f N_b)` Hessian with optional ``rotation_scale`` to
   put translational and rotational coordinates on comparable length scales.
3. **Geometric friction:**
   :func:`~jaxdem.utils.contacts.get_group_contacts` decomposes each clump-pair
   resultant force along the COM axis into :math:`\mu = |F_t| / |F_n|` —
   shape-induced friction even when constituent contacts are frictionless.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem
from jaxdem.utils import (
    count_clump_contacts,
    count_vertex_contacts,
    get_clump_rattler_ids,
    get_contacts,
    get_group_contacts,
    remove_rattlers,
)
from jaxdem.utils.dynamical_matrix import clump_non_bonded_hessian, zero_mode_mask
from jaxdem.utils.jamming import bisection_jam
from jaxdem.utils.particle_creation import build_ga_system

# %%
# 1) Build and Jam a Bidisperse 2D Clump Packing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Eight mixed 3- and 4-asperity clumps with ``core_type="phantom"`` (central
# core sets mass/inertia/volume without entering contacts):

dim = 2
state, system = build_ga_system(
    particle_radii=[0.5] * 4 + [0.7] * 4,
    vertex_counts=[3] * 4 + [4] * 4,
    asperity_radius=0.3,
    phi=0.50,
    dim=dim,
    particle_type="clump",
    core_type="phantom",
    domain_type="periodic",
    randomize_orientation=True,
    n_property_samples=256,
    mesh_type="fibonacci",
    compression_step=0.05,
    compression_force_tol=1e-6,
    max_n_min_steps_per_outer=3_000,
    fire_dt=0.05,
    dt=0.05,
    minimizer=jdem.fire,
    minimizer_kw={"dt": 0.05},
    collider_type="naive",
    seed=0,
)

result = bisection_jam(
    state,
    system,
    pe_tol=1e-8,
    pe_band_factor=4.0,
    n_jamming_steps=80,
    packing_fraction_increment=0.04,
    packing_fraction_tolerance=1e-4,
    minimizer_kwargs={"max_steps": 5_000, "force_tol": 1e-7, "torque_tol": 1e-7},
    verbose=False,
)
if not bool(result.converged):
    raise RuntimeError(
        f"Jamming failed: reason={result.reason}, minimizer={result.minimizer_reason}"
    )
state, system = result.jammed_state, result.jammed_system
print(
    f"Jammed nominal phi_J = {float(result.packing_fraction):.6f}, "
    f"E / vertex sphere = {float(result.potential_energy):.3e}"
)

# %%
# 2) Vertex Contacts, Clump Contacts, and Rattler Removal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``get_clump_rattler_ids`` already returns clump IDs (ready to pass straight
# to :func:`~jaxdem.utils.contacts.remove_rattlers`), removing bodies with
# fewer than ``dim + dim_rot + 1 = 4`` vertex contacts in 2D:

state, system, contacts = get_contacts(state, system)
state, system, vertex_counts = count_vertex_contacts(state, system, contacts=contacts)
state, system, clump_counts = count_clump_contacts(state, system, contacts=contacts)
state, system, rattler_ids, _ = get_clump_rattler_ids(state, system, contacts=contacts)
print("Vertex contacts per clump   :", np.asarray(vertex_counts))
print("Neighboring clumps per clump:", np.asarray(clump_counts))
print("Rattler clump IDs           :", np.asarray(rattler_ids))

state_nr, system_nr = remove_rattlers(state, system, rattler_ids)

# %%
# 3) Rigid-Body Hessian (with ``rotation_scale``)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Passing a per-clump ``rotation_scale`` array scales angular coordinates as
# :math:`(\delta x, \delta y, R\omega)` so translation and rotation share
# length units:

n_clumps = int(jnp.max(state.clump_id)) + 1
state, system, H = clump_non_bonded_hessian(
    state, system, rotation_scale=jnp.full(n_clumps, 0.6)
)
evals = np.linalg.eigvalsh(0.5 * (np.asarray(H) + np.asarray(H).T))
zero_mask = np.asarray(zero_mode_mask(evals))

n_clumps_nr = int(jnp.max(state_nr.clump_id)) + 1
state_nr, system_nr, H_nr = clump_non_bonded_hessian(
    state_nr, system_nr, rotation_scale=jnp.full(n_clumps_nr, 0.6)
)
evals_nr = np.linalg.eigvalsh(0.5 * (np.asarray(H_nr) + np.asarray(H_nr).T))
zero_mask_nr = np.asarray(zero_mode_mask(evals_nr))
print(
    f"Zero modes: {int(zero_mask.sum())} (all clumps) -> "
    f"{int(zero_mask_nr.sum())} (after pruning; {dim} global translations expected)"
)

# %%
# 4) Geometric Friction Between Contacting Clumps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :func:`~jaxdem.utils.contacts.get_group_contacts` returns directed clump
# pairs; filter ``I < J`` to count each physical pair once:

state_nr, system_nr, groups = get_group_contacts(state_nr, system_nr)
pairs = np.asarray(groups.pair_ids)
unique = pairs[:, 0] < pairs[:, 1]
mu_values = np.asarray(groups.friction)[unique]
finite_mu = np.sort(mu_values[np.isfinite(mu_values)])
print("Unique contacting clump pairs:", len(pairs[unique]))
print("Participating spheres per side:", np.asarray(groups.sphere_counts)[unique].tolist())
if len(finite_mu):
    print(f"Geometric friction mu: mean={finite_mu.mean():.3f}, max={finite_mu.max():.3f}")

# %%
# Summary Plots
# ~~~~~~~~~~~~~
scene_states = {"Before rattler removal": state, "After rattler removal": state_nr}
scene_systems = {"Before rattler removal": system, "After rattler removal": system_nr}

scale = max(float(np.max(np.abs(evals))), 1.0)
linthresh = 1e-7 * scale

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
axes[0].plot(
    np.arange(len(evals)),
    evals,
    "o-",
    ms=4,
    label=f"All clumps ({zero_mask.sum()} zero modes)",
)
axes[0].plot(
    np.arange(len(evals_nr)),
    evals_nr,
    "s--",
    ms=4,
    label=f"Pruned ({zero_mask_nr.sum()} zero modes)",
)
axes[0].set_yscale("symlog", linthresh=linthresh)
axes[0].set_yticks([t for t in axes[0].get_yticks() if t == 0 or abs(t) >= linthresh])
axes[0].axhline(0.0, color="0.5", lw=0.8)
axes[0].set(
    xlabel="Sorted mode index",
    ylabel="Clump Hessian eigenvalue",
    title="Rigid-body Hessian spectrum",
)
axes[0].legend(fontsize="small")
axes[0].grid(alpha=0.2)

if len(finite_mu):
    y = np.arange(1, len(finite_mu) + 1) / len(finite_mu)
    axes[1].step(np.r_[0.0, finite_mu], np.r_[0.0, y], where="post", color="tab:purple")
    axes[1].plot(finite_mu, y, "o", color="tab:purple", label=f"{len(finite_mu)} clump pairs")
    axes[1].legend(fontsize="small")
axes[1].set(
    xlabel=r"Geometric friction $\mu = |F_t| / |F_n|$",
    ylabel="Cumulative fraction of pairs",
    ylim=(0.0, 1.05),
    title="Shape-induced tangential force ratio",
)
axes[1].grid(alpha=0.2)
plt.show()
