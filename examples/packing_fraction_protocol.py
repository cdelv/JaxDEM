# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Packing-fraction protocol: integrate with a scheduled box rescale
====================================================================

Drive a 3D sphere packing along a time-varying packing-fraction schedule using
:func:`~jaxdem.utils.dynamics_routines.run_packing_fraction_protocol` and a
pseudo-logarithmic save schedule from
:func:`~jaxdem.utils.rollout_schedules.make_save_steps_pseudolog`.

The protocol integrates each stride using whatever integrator, bonded model,
and collider live on ``system``, then affinely rescales the box to the next
target ``phi`` before saving the frame. To combine a density ramp with
temperature control, see :doc:`temperature_density_control`.
"""

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdem.utils.dynamics_routines import run_packing_fraction_protocol
from jaxdem.utils.packing_utils import compute_packing_fraction
from jaxdem.utils.particle_creation import build_sphere_system
from jaxdem.utils.rollout_schedules import make_save_steps_pseudolog
from jaxdem.utils.thermal import compute_translational_kinetic_energy

# %%
# 1) Build the Starting Packing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :func:`~jaxdem.utils.particle_creation.build_sphere_system` returns a relaxed
# initial state with zero velocities; kinetic energy rises only as compression
# drives spheres into contact:

N = 16
phi_start = 0.35
phi_end = 0.55

state, system = build_sphere_system(
    particle_radii=[0.1] * N,
    phi=phi_start,
    dim=3,
    dt=1e-3,
    collider_type="naive",
    seed=0,
    compression_step=0.05,
    compression_force_tol=1e-7,
    max_n_min_steps_per_outer=2_000,
    fire_dt=0.05,
)
state, system = system.initialize(state, system)
print(f"Initial phi = {float(compute_packing_fraction(state, system)):.4f}")

# %%
# 2) Build a Pseudolog Save Schedule and Ramp Profile
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set ``include_step0=False`` because the protocol records each frame *after*
# its stride and rescale; ``np.diff`` turns absolute step indices into
# per-frame strides:

num_steps = 400
save_steps = make_save_steps_pseudolog(
    num_steps=num_steps,
    reset_save_decade=200,
    min_save_decade=10,
    decade=10,
    include_step0=False,
)
strides = np.diff(np.concatenate([[0], save_steps]))
n_frames = int(strides.size)

t_frac = save_steps / float(num_steps)
ramp_end = 0.6
phi_at_frames = np.where(
    t_frac < ramp_end,
    phi_start + (phi_end - phi_start) * (t_frac / ramp_end),
    phi_end,
).astype(float)
print(f"Scheduled {n_frames} frames (first 5 steps: {save_steps[:5]})")

# %%
# 3) Run the Protocol and Read Back Trajectory Observables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``traj_state`` and ``traj_system`` carry a leading time axis of length
# ``n_frames`` (positions shape ``(n_frames, N, dim)``), so ``jax.vmap``
# evaluates per-frame observables in one compiled pass:

state, system, (traj_state, traj_system) = run_packing_fraction_protocol(
    state,
    system,
    strides=jnp.asarray(strides, dtype=int),
    phi_at_frames=jnp.asarray(phi_at_frames, dtype=float),
)
system.check_overflow()

phi_trace = np.asarray(jax.vmap(compute_packing_fraction)(traj_state, traj_system))
ke_trace = np.asarray(jax.vmap(compute_translational_kinetic_energy)(traj_state))

print("Trajectory pos shape (T, N, dim):", traj_state.pos.shape)
print("idx   step      phi         KE")
for i in (0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1):
    print(f"{i:3d}  {save_steps[i]:5d}   {phi_trace[i]:.4f}   {ke_trace[i]:.3e}")

# %%
# Plot Imposed Schedule and Kinetic Energy Response
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

times = np.asarray(traj_system.time)
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
axes[0].plot(times, phi_at_frames, "--", color="0.4", label="Requested schedule")
axes[0].plot(times, phi_trace, "o", markersize=3, label="Measured frames")
axes[0].set(ylabel=r"Packing fraction $\phi$", title="Compression ramp followed by hold")
axes[0].legend(fontsize="small")

axes[1].plot(times, ke_trace, "o-", color="tab:orange", markersize=3)
axes[1].set(xlabel="Physical time", ylabel="Translational kinetic energy")
for ax in axes:
    ax.axvspan(ramp_end * num_steps * float(system.dt), times[-1], color="0.5", alpha=0.1)
    ax.grid(alpha=0.2)
plt.show()
