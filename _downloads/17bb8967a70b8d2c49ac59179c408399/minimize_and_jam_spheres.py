# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Batched energy minimization and bisection jamming
=====================================================

Minimize overlapping bidisperse 2D disk packings with FIRE and bracket their
jamming thresholds with :py:func:`~jaxdem.utils.jamming.bisection_jam`,
running an entire ensemble in parallel via ``jax.vmap``.

Harmonic spring contacts have zero energy when separated and
:math:`U = \frac{1}{2}\sum_{i<j} k_{ij}\delta_{ij}^2` under overlap. Below
jamming, FIRE relaxes all overlaps to zero; above jamming, a compressed
contact network retains positive energy at force balance. Bisection brackets
this onset between an unjammed state (``unjammed_state``, :math:`e \le e_{\mathrm{tol}}`)
and a jammed state (``jammed_state``, :math:`e_{\mathrm{tol}} < e \le b\,e_{\mathrm{tol}}`).

For contact, rattler, and Hessian rigidity checks on jammed packings, see
:doc:`sphere_jamming_rattlers_dynamical_matrix`.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem
from jaxdem.utils.thermal import compute_potential_energy

# %%
# Build an Ensemble of Overlapping Bidisperse Disk Packings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use ``jax.vmap`` to create ``n_systems`` independent realizations of 16
# bidisperse disks (radius ratio 1.4) at an initial packing fraction ``phi=0.65``:

n_systems = 4
n_disks = 16
phi_init = 0.65
dim = 2
dt_fire = 5e-2

mats = [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
mat_table = jdem.MaterialTable.from_materials(
    mats, matcher=jdem.MaterialMatchmaker.create("harmonic")
)


def build_microstate(seed: jax.Array) -> tuple[jdem.State, jdem.System]:
    rad = jnp.concatenate([jnp.full(n_disks // 2, 0.5), jnp.full(n_disks // 2, 0.7)])
    volume = jnp.pi * rad**2
    box_len = jnp.sqrt(jnp.sum(volume) / phi_init)
    pos = jax.random.uniform(jax.random.key(seed), (n_disks, dim), minval=0.0, maxval=box_len)
    state = jdem.State.create(pos=pos, rad=rad, mass=jnp.ones(n_disks), volume=volume)
    system = jdem.System.create(
        state_shape=state.shape,
        dt=dt_fire,
        minimizer=jdem.minimizers.fire,
        minimizer_kw={"dt": dt_fire},
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={"box_size": jnp.full(dim, box_len)},
    )
    return state, system


init_states, init_systems = jax.vmap(build_microstate)(jnp.arange(n_systems))

# %%
# 1) Batched Energy Minimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``jax.vmap(jdem.System.minimize)`` relaxes every realization in parallel
# (waiting until the slowest member converges). Always inspect ``result.converged``
# and ``result.info.force_max``:

initial_pe = jax.vmap(compute_potential_energy)(init_states, init_systems) / n_disks
min_res = jax.vmap(
    lambda st, sys: jdem.System.minimize(
        st, sys, max_steps=2_000, force_tol=1e-7, torque_tol=1e-7
    )
)(init_states, init_systems)
if not bool(jnp.all(min_res.converged)):
    raise RuntimeError(f"Minimization failed: reason={min_res.reason}")

print("Initial PE / disk  :", np.asarray(initial_pe))
print("Minimized PE / disk:", np.asarray(min_res.energy))
print("Max force residual :", np.asarray(min_res.info.force_max))
print("FIRE steps         :", np.asarray(min_res.steps))

# %%
# 2) Batched Bisection Jamming
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:func:`~jaxdem.utils.jamming.bisection_jam` rescales the periodic box and
# minimizes at each trial until the packing fraction bracket closes within
# ``packing_fraction_tolerance``. Both the jammed upper state and the unjammed
# lower bracket state are returned on the result object:

pe_tol = 1e-8
jam_res = jax.vmap(
    lambda st, sys: jdem.utils.jamming.bisection_jam(
        st,
        sys,
        pe_tol=pe_tol,
        pe_band_factor=4.0,
        n_jamming_steps=80,
        packing_fraction_increment=0.04,
        packing_fraction_tolerance=1e-4,
        minimizer_kwargs=dict(max_steps=5_000, force_tol=1e-7, torque_tol=1e-7),
        verbose=False,
    )
)(min_res.state, min_res.system)
if not bool(jnp.all(jam_res.converged)):
    raise RuntimeError(
        f"Jamming failed: reason={jam_res.reason}, minimizer={jam_res.minimizer_reason}"
    )

unjammed_phi = jax.vmap(jdem.utils.packing_utils.compute_packing_fraction)(
    jam_res.unjammed_state, jam_res.unjammed_system
)
print("Unjammed bracket phi:", np.round(np.asarray(unjammed_phi), 4))
print("Jammed bracket phi  :", np.round(np.asarray(jam_res.packing_fraction), 4))
print("Jammed PE / disk    :", np.asarray(jam_res.potential_energy))
print("Bisection trials    :", np.asarray(jam_res.steps))

# Slice replica 0 out of the batched pytrees for ParaView visualization.
pick0 = lambda tree: jax.tree.map(lambda x: x[0], tree)
scene_states = {
    "Initial random": pick0(init_states),
    "Minimized (phi=0.65)": pick0(min_res.state),
    "Jammed (bisection)": pick0(jam_res.jammed_state),
}
scene_systems = {
    "Initial random": pick0(init_systems),
    "Minimized (phi=0.65)": pick0(min_res.system),
    "Jammed (bisection)": pick0(jam_res.jammed_system),
}

# %%
# Summary Plot Across Replicas
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

replicas = np.arange(n_systems)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
axes[0].semilogy(replicas, np.asarray(initial_pe), "o-", label="Initial random")
axes[0].semilogy(
    replicas,
    np.maximum(np.asarray(min_res.energy), 1e-16),
    "s-",
    label="After FIRE (floor 1e-16)",
)
axes[0].semilogy(replicas, np.asarray(jam_res.potential_energy), "^-", label="At jamming")
axes[0].set(xlabel="Replica seed", ylabel="PE per disk", title="Energy across preparation stages")
axes[0].set_xticks(replicas)
axes[0].legend(fontsize="small")
axes[0].grid(alpha=0.2, which="both")

axes[1].bar(replicas, np.asarray(jam_res.packing_fraction), width=0.5, color="tab:blue")
axes[1].set(
    xlabel="Replica seed",
    ylabel=r"Jammed packing fraction $\phi_J$",
    title=f"Finite-size jamming thresholds (N={n_disks})",
)
axes[1].set_xticks(replicas)
axes[1].set_ylim(0.7, 0.95)
axes[1].grid(alpha=0.2, axis="y")
plt.show()
