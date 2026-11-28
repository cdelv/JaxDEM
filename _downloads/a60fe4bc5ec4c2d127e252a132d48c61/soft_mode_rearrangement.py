# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Watch a soft mode develop into a rearrangement
==================================================

Track the lowest vibrational eigenmode of a sheared 2D bidisperse disk packing
with ``jax.hessian`` across athermal quasistatic (AQS) shear steps, and measure
its alignment with the nonaffine displacement during a plastic rearrangement.

Starting from the prepared 16-disk configuration in
``data/soft_mode_packing.json`` (see :doc:`lees_edwards_shear` for how AQS
packings are prepared), this example recomputes equilibrium relaxation, exact
Hessians, mode softening, and the rearrangement event from scratch. Requires
Matplotlib.
"""

from dataclasses import replace
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem

# %%
# 1) Load the Prepared Packing and Define Exact ``jax.hessian`` Diagnostics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Three JAX/JaxDEM details make exact second derivatives work cleanly:
#
# * Differentiate with respect to the stored center coordinates ``pos_c`` (not
#   the derived ``pos`` property).
# * Use ``collider_type="naive"`` (smooth, branchless all-pairs energy).
# * :meth:`~jaxdem.colliders.Collider.compute_potential_energy` returns
#   ``(state, system, pe)``, so index ``[2]`` extracts the scalar potential energy.
#
# To remove the two uniform periodic translations without discarding genuine
# soft modes, build an orthonormal complement :math:`Q` via complete QR
# factorization and diagonalize :math:`Q^T H Q`.

example_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
fixture = json.loads((example_dir / "data" / "soft_mode_packing.json").read_text())
N = len(fixture["radii"])
STRAIN_INCREMENT = 5e-5
MAX_STRAIN_STEPS = 36
FORCE_TOL = 2e-10

state = jdem.State.create(
    pos=jnp.asarray(fixture["positions"]),
    rad=jnp.asarray(fixture["radii"]),
    mass=jnp.ones(N),
)
system = jdem.System.create(
    state=state,
    dt=0.01,
    force_model_type="spring",
    collider_type="naive",
    rotation_integrator_type=None,
    minimizer=jdem.minimizers.fire,
    minimizer_kw={"dt": 0.05},
    domain_type="leesedwards",
    domain_kw={"box_size": fixture["box_size"], "gamma": fixture["gamma"]},
    mat_table=jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", **fixture["material"])]
    ),
)


def total_energy(flat_positions: jax.Array, st: jdem.State, sys: jdem.System) -> jax.Array:
    trial = replace(st, pos_c=flat_positions.reshape(st.pos_c.shape))
    return sys.collider.compute_potential_energy(trial, sys)[2]


energy = jax.jit(total_energy)
hessian = jax.jit(jax.hessian(total_energy, argnums=0))
translations = np.tile(np.eye(2), (N, 1))
Q = np.linalg.qr(translations, mode="complete")[0][:, 2:]


def spectrum(st: jdem.State, sys: jdem.System):
    H = np.asarray(hessian(st.pos_c.reshape(-1), st, sys))
    H = 0.5 * (H + H.T)
    values, vectors = np.linalg.eigh(Q.T @ H @ Q)
    return H, values, Q @ vectors[:, 0]


def relax(st: jdem.State, sys: jdem.System):
    res = jdem.System.minimize(
        st, sys, max_steps=8_000, force_tol=FORCE_TOL, torque_tol=FORCE_TOL
    )
    if not bool(res.converged):
        raise RuntimeError(f"AQS relaxation failed: {res.info}")
    return res


# %%
# 2) Shear Quasistatically Until a Plastic Rearrangement Occurs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# At each strain step, apply :meth:`~jaxdem.domains.lees_edwards.LeesEdwardsDomain.shear`,
# relax to force balance, and measure the normalized overlap
# :math:`C = |\mathbf{e}_{\min}^{n} \cdot \mathbf{u}^{\mathrm{NA}}| / \|\mathbf{u}^{\mathrm{NA}}\|`
# between the preceding lowest eigenmode and the nonaffine displacement:

initial = relax(state, system)
state, system = initial.state, initial.system
_, eigenvalues, mode = spectrum(state, system)

strain_values = [float(system.domain.gamma)]
energies_per_disk = [float(initial.energy)]
released_per_disk = [0.0]
lowest_values = [eigenvalues[:2]]
overlaps = [0.0]
scene_states = [state]
scene_systems = [system]
event = None

for _ in range(MAX_STRAIN_STEPS):
    prev_mode = mode
    affine, strained_sys = system.domain.shear(state, system, STRAIN_INCREMENT)
    affine_energy = float(energy(affine.pos_c.reshape(-1), affine, strained_sys))
    res = relax(affine, strained_sys)
    state, system = res.state, res.system

    disp = np.asarray(system.domain.displacement(state.pos, affine.pos, system))
    disp = disp - disp.mean(axis=0)
    disp_norm = float(np.linalg.norm(disp))
    overlap = float(abs(prev_mode @ disp.reshape(-1)) / disp_norm)
    _, eigenvalues, mode = spectrum(state, system)
    energy_drop = energies_per_disk[-1] - float(res.energy)

    strain_values.append(float(system.domain.gamma))
    energies_per_disk.append(float(res.energy))
    released_per_disk.append(affine_energy / N - float(res.energy))
    lowest_values.append(eigenvalues[:2])
    overlaps.append(overlap)
    scene_states.append(state)
    scene_systems.append(system)

    if disp_norm > 0.15 and energy_drop > 1e-4:
        event = (disp_norm, energy_drop, overlap)
        break

if event is None:
    raise RuntimeError("No energy-releasing rearrangement found in strain interval.")

event_norm, drop, alignment = event
lowest_values = np.asarray(lowest_values)
print(f"Rearrangement at gamma = {strain_values[-1]:.5f}")
print(f"Energy drop / disk = {drop:.4e} | nonaffine norm = {event_norm:.4f}")
print(f"Overlap with preceding soft mode: C = {alignment:.4f}")
print(
    f"Lowest eigenvalue softened from {lowest_values[0, 0]:.4g} "
    f"to {lowest_values[:-1, 0].min():.4g}"
)

# %%
# 3) Plot Softening, Mode Overlap, and Energy Release
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

strain_offset = 1_000 * (np.asarray(strain_values) - strain_values[0])
fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), layout="constrained")

axes[0, 0].plot(strain_offset, energies_per_disk, "o-", ms=3, color="#275e94")
axes[0, 0].set(ylabel="Relaxed energy U / N", title="Equilibrium energy")

axes[0, 1].semilogy(
    strain_offset, lowest_values[:, 0], "o-", ms=3, color="#258685", label=r"lowest $\lambda$"
)
axes[0, 1].semilogy(strain_offset, lowest_values[:, 1], "--", color="0.55", label=r"next $\lambda$")
axes[0, 1].set(ylabel="Stiffness eigenvalue", title="Soft mode precedes rearrangement")
axes[0, 1].legend(fontsize=8)

axes[1, 0].plot(strain_offset[1:], overlaps[1:], "o-", ms=3, color="#275e94")
axes[1, 0].axhline(alignment, color="0.5", ls="--", lw=1)
axes[1, 0].set(
    ylim=(0, 1.05),
    ylabel="Mode overlap C",
    title=f"Soft-mode alignment at event: {alignment:.3f}",
)

axes[1, 1].plot(strain_offset, released_per_disk, "o-", ms=3, color="#d07a28")
axes[1, 1].set(ylabel="Released energy / N", title="Relaxation energy drop")

for ax in axes.flat:
    ax.axvline(strain_offset[-1], color="0.45", ls=":", lw=1)
    ax.set_xlabel(r"Strain increment from fixture, $1000\,\Delta\gamma$")
    ax.grid(alpha=0.2)
fig.suptitle("Soft collective mode developing into a plastic rearrangement", fontsize=13)
plt.show()

# %%
# Preparation Provenance & References
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Full generation parameters for ``data/soft_mode_packing.json`` are stored in
# its ``"provenance"`` field (built with the AQS protocol from
# :doc:`lees_edwards_shear`). For background on soft modes and AQS plasticity,
# see Maloney & Lemaître (`arXiv:cond-mat/0510677
# <https://arxiv.org/abs/cond-mat/0510677>`_) and Manning & Liu
# (`arXiv:1012.4822 <https://arxiv.org/abs/1012.4822>`_).
