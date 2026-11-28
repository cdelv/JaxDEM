# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Jammed bidisperse spheres: contacts, rattlers, and the dynamical matrix
==========================================================================

Take a periodic 2D bidisperse disk packing from
:func:`~jaxdem.utils.jamming.bisection_jam` through contact counting, rattler
pruning, and energy-Hessian spectral analysis using
:func:`~jaxdem.utils.dynamical_matrix.non_bonded_hessian`.

For the rigid-clump counterpart, see
:doc:`clump_jamming_dynamical_matrix_and_friction`; for a one-call wrapper that
computes contacts, pressure, rattlers, and eigenvalues together, see
:doc:`packing_analysis`.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem
from jaxdem.utils import (
    count_sphere_contacts,
    get_contacts,
    get_sphere_rattler_ids,
    remove_rattlers,
)
from jaxdem.utils.dynamical_matrix import non_bonded_hessian, zero_mode_mask
from jaxdem.utils.jamming import bisection_jam

# %%
# 1) Jam a Bidisperse 2D Disk Packing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build 16 bidisperse disks (radius ratio 1.4) in a periodic box and find a
# jammed configuration with :func:`~jaxdem.utils.jamming.bisection_jam`:

dim = 2
N = 16
radii = jnp.array([0.5] * (N // 2) + [0.7] * (N // 2))
volume = jnp.pi * radii**2
L = jnp.sqrt(jnp.sum(volume) / 0.65)

state = jdem.State.create(
    pos=jax.random.uniform(jax.random.key(0), (N, dim)) * L,
    rad=radii,
    mass=jnp.ones(N),
    volume=volume,
)
mat_table = jdem.MaterialTable.from_materials(
    [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
)
system = jdem.System.create(
    state=state,
    dt=0.05,
    rotation_integrator_type=None,
    minimizer=jdem.fire,
    minimizer_kw={"dt": 0.05},
    domain_type="periodic",
    domain_kw={"box_size": jnp.full(dim, L)},
    collider_type="naive",
    mat_table=mat_table,
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
    f"Jammed phi_J = {float(result.packing_fraction):.6f}, "
    f"E/N = {float(result.potential_energy):.3e}, trials = {int(result.steps)}"
)

# %%
# 2) Count Contacts and Identify Rattlers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Capture one contact snapshot with :func:`~jaxdem.utils.contacts.get_contacts`
# and reuse it across :func:`~jaxdem.utils.contacts.count_sphere_contacts` and
# :func:`~jaxdem.utils.contacts.get_sphere_rattler_ids`. In 2D, the default
# recursive rattler screen removes disks with fewer than ``dim + 1 = 3``
# force-bearing contacts:

state, system, contacts = get_contacts(state, system)
state, system, counts = count_sphere_contacts(state, system, contacts=contacts)
state, system, rattler_ids, non_rattler_ids = get_sphere_rattler_ids(
    state, system, contacts=contacts
)
print("Contacts per disk:", np.asarray(counts))
print("Distinct contacts:", int(np.asarray(counts).sum()) // 2)
print("Rattler disk indices:", np.asarray(rattler_ids))

# %%
# 3) Energy Hessian, Mass-Weighted Dynamical Matrix, and Rattler Pruning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :func:`~jaxdem.utils.dynamical_matrix.non_bonded_hessian` computes the
# :math:`(dN \times dN)` second derivative of pair energy
# :math:`H_{\alpha\beta} = \partial^2 U / \partial q_\alpha \partial q_\beta`
# (including both stiffness and compressive prestress). For unequal masses,
# form the mass-weighted dynamical matrix :math:`D = M^{-1/2} H M^{-1/2}` so
# eigenvalues give squared vibrational frequencies :math:`\omega^2`.
#
# Note the ID space difference when pruning: ``get_sphere_rattler_ids`` returns
# particle row indices, whereas :func:`~jaxdem.utils.contacts.remove_rattlers`
# takes **clump IDs** (``state.clump_id[rattler_ids]``) and returns a new
# initialized ``(state_nr, system_nr)`` pair:


def hessian_spectrum(st: jdem.State, sys: jdem.System):
    st, sys, H = non_bonded_hessian(st, sys)
    H_sym = 0.5 * (np.asarray(H) + np.asarray(H).T)
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(np.asarray(st.mass), dim))
    D = (inv_sqrt_m[:, None] * H_sym) * inv_sqrt_m[None, :]
    evals, evecs = np.linalg.eigh(D)
    zmask = np.asarray(zero_mode_mask(evals))
    return st, sys, evals, evecs, zmask


state, system, evals, evecs, zero_mask = hessian_spectrum(state, system)
rattler_clump_ids = state.clump_id[rattler_ids]
state_nr, system_nr = remove_rattlers(state, system, rattler_clump_ids)
state_nr, system_nr, evals_nr, evecs_nr, zero_mask_nr = hessian_spectrum(
    state_nr, system_nr
)

print(
    f"Before pruning: N={state.N}, zero modes={int(zero_mask.sum())}, "
    f"lowest nonzero={evals[~zero_mask][0]:.3e}"
)
print(
    f"After pruning : N={state_nr.N}, zero modes={int(zero_mask_nr.sum())} "
    f"(periodic symmetry expects {dim}), lowest nonzero={evals_nr[~zero_mask_nr][0]:.3e}"
)

# Inspect the lowest non-trivial vibrational mode vector (shape (N_nr, dim)):
lowest_mode = evecs_nr[:, int(zero_mask_nr.sum())].reshape(state_nr.N, dim)
print("Max component of lowest vibrational eigenmode:", float(np.max(np.abs(lowest_mode))))

# %%
# Compare Spectra Before and After Pruning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scene_states = {"Before rattler removal": state, "After rattler removal": state_nr}
scene_systems = {"Before rattler removal": system, "After rattler removal": system_nr}

scale = max(float(np.max(np.abs(evals))), 1.0)
linthresh = 1e-7 * scale

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(
    np.arange(len(evals)),
    evals,
    "o-",
    markersize=4,
    label=f"All disks (N={state.N}, {zero_mask.sum()} zero modes)",
)
ax.plot(
    np.arange(len(evals_nr)),
    evals_nr,
    "s--",
    markersize=4,
    label=f"After pruning (N={state_nr.N}, {zero_mask_nr.sum()} zero modes)",
)
ax.set_yscale("symlog", linthresh=linthresh)
ax.set_yticks([t for t in ax.get_yticks() if t == 0 or abs(t) >= linthresh])
ax.axhline(0.0, color="0.5", linewidth=0.8)
ax.set(
    xlabel="Sorted mode index",
    ylabel=r"Dynamical-matrix eigenvalue $\omega^2$",
    title="Effect of rattler pruning on the vibrational spectrum",
)
ax.grid(alpha=0.2)
ax.legend(fontsize="small")
plt.show()
