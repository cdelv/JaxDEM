# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Packing-fraction protocol: integrate with a scheduled box rescale
======================================================================

:func:`~jaxdem.utils.dynamics_routines.run_packing_fraction_protocol`
wraps :meth:`~jaxdem.system.System.step` and calls
:func:`~jaxdem.utils.packing_utils.scale_to_packing_fraction` on a
user-supplied, per-frame schedule. Temperature control, bonded forces,
and the collider come from the ``System`` you pass in. The protocol
does not change them.

Here we:

1. Build a sphere packing at ``phi = 0.35`` with
   :func:`~jaxdem.utils.particle_creation.build_sphere_system`.
2. Ramp ``phi`` linearly up to ``0.55`` and then hold it there. We save
   frames on a pseudolog schedule, which samples the fast early changes
   densely and the slow late drift sparsely.
3. Read back per-frame ``phi`` and total kinetic energy along the ramp.

Sixteen particles and 400 integration steps make this a quick protocol
demonstration, not an equilibrated equation-of-state measurement. All
parameters use consistent reduced units.
"""

# %%
# Imports
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

from jaxdem.utils.particle_creation import build_sphere_system
from jaxdem.utils.dynamics_routines import run_packing_fraction_protocol
from jaxdem.utils.packing_utils import compute_packing_fraction
from jaxdem.utils.rollout_schedules import make_save_steps_pseudolog
from jaxdem.utils.thermal import compute_translational_kinetic_energy

# %%
# 1) Build the starting system
# ----------------------------
N = 16
phi_start = 0.35
phi_end = 0.55
dim = 3

state, system = build_sphere_system(
    particle_radii=[0.1] * N,
    phi=phi_start,
    dim=dim,
    dt=1e-3,
    collider_type="naive",
    seed=0,
    compression_step=0.05,
    compression_force_tol=1e-7,
    max_n_min_steps_per_outer=2_000,
    fire_dt=0.05,
)
state, system = system.initialize(state, system)
print(f"initial phi = {float(compute_packing_fraction(state, system)):.4f}")


# %%
# 2) Build the schedule
# ---------------------
# ``make_save_steps_pseudolog`` gives us non-uniform save points that
# densely sample the early transient and thin out later. ``np.diff``
# turns those absolute step indices into the per-frame strides the
# protocol wants.

num_steps = 400
save_steps = make_save_steps_pseudolog(
    num_steps=num_steps,
    reset_save_decade=200,
    min_save_decade=10,
    decade=10,
    # The protocol records each frame *after* its integration stride and
    # box rescale, so a step-0 entry would be a zero-length first
    # stride. The initial state is the ``state`` we already hold.
    include_step0=False,
)
strides = np.diff(
    np.concatenate([[0], save_steps])
)  # stride from step 0 to save_steps[0], then between frames
n_frames = int(strides.size)

# Target phi at each frame: linear ramp from phi_start -> phi_end over
# the first 60% of the protocol, then hold at phi_end.
t_frac = save_steps / float(num_steps)
ramp_end = 0.6
phi_at_frames = np.where(
    t_frac < ramp_end,
    phi_start + (phi_end - phi_start) * (t_frac / ramp_end),
    phi_end,
).astype(float)

print(f"n_frames = {n_frames}  (save_steps[:5] = {save_steps[:5]})")


# %%
# 3) Run the protocol
# -------------------
# The imposed quantity is :math:`\phi=V_p/V_{\mathrm{box}}`, with the
# particle volume :math:`V_p` held fixed during each affine rescale. To
# change from :math:`\phi` to :math:`\phi'` without changing box aspect ratio,
#
# .. math::
#
#    s=\left(\frac{\phi}{\phi'}\right)^{1/d},\qquad
#    L'_\alpha=sL_\alpha,\qquad
#    \mathbf{c}'_I=\mathbf{a}+s(\mathbf{c}_I-\mathbf{a}).
#
# Here :math:`\mathbf{a}` is the domain anchor and :math:`\mathbf{c}_I` a
# body center. Radii and internal body geometry remain fixed, so increasing
# :math:`\phi` decreases separation rather than inflating the particles.
# The protocol integrates a chunk, applies that frame's rescale, and then
# records the frame. Consequently the saved :math:`\phi_i` should equal
# the schedule at :math:`t_i=\mathrm{dt}\,\mathrm{save\_steps}_i`.
# The pseudolog schedule changes sampling density, not the physical timestep.
#
# The protocol runs pure Verlet integration between rescale events. There is no thermostat, so
# box rescaling can do work on the particles. KE need not grow monotonically:
# energy can exchange between kinetic and contact-potential contributions.
state, system, (traj_state, traj_system) = run_packing_fraction_protocol(
    state,
    system,
    strides=jnp.asarray(strides, dtype=int),
    phi_at_frames=jnp.asarray(phi_at_frames, dtype=float),
)


# %%
# 4) Read back the trajectory
# ---------------------------
phi_trace = np.asarray(jax.vmap(compute_packing_fraction)(traj_state, traj_system))
ke_trace = np.asarray(jax.vmap(compute_translational_kinetic_energy)(traj_state))

np.testing.assert_allclose(phi_trace, phi_at_frames, rtol=1e-10, atol=1e-12)
assert np.all(np.isfinite(ke_trace))
print("idx  step      phi       KE")
for i in (0, 1, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1):
    print(f"{i:3d}  {save_steps[i]:6d}    {phi_trace[i]:.4f}    {ke_trace[i]:.3e}")


# %%
# Check both the imposed schedule and the mechanical response
# -----------------------------------------------------------
# The saved markers should follow the requested fraction exactly. The
# kinetic energy is a response of this short, unthermostatted run; holding
# the box fixed after the ramp does not impose zero kinetic energy or
# establish that the packing has equilibrated. Marker spacing also shows
# the nonuniform recording cadence.
times = np.asarray(traj_system.time)
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
axes[0].plot(times, phi_at_frames, "--", color="0.4", label="Requested fraction")
axes[0].plot(times, phi_trace, "o", markersize=3, label="Saved fraction")
axes[0].set(
    ylabel=r"Packing fraction $\phi$",
    title="Ramp the packing fraction, then hold the box",
)
axes[0].legend(fontsize="small")
axes[1].plot(times, ke_trace, "o-", color="tab:orange", markersize=3)
axes[1].set(
    xlabel="Physical time (reduced units)", ylabel="Total translational kinetic energy"
)
for ax in axes:
    ax.axvspan(
        ramp_end * num_steps * float(system.dt), times[-1], color="0.5", alpha=0.1
    )
    ax.grid(alpha=0.2)
plt.show()


# %%
# To keep the kinetic temperature clamped while phi still ramps, build
# the system with a velocity-rescaling thermostat instead. The builder
# forwards integrator keyword arguments directly:
#
# .. code-block:: python
#
#     state, system = build_sphere_system(
#         particle_radii=[0.1] * N,
#         phi=phi_start,
#         dim=dim,
#         dt=1e-3,
#         collider_type="naive",
#         linear_integrator_type="verlet_rescaling",
#         linear_integrator_kw={"temperature": 1.0},
#         seed=0,
#     )
#
# The protocol call above then runs unchanged. The protocol works with
# any integrator.
