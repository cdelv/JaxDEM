# %%
"""Packing-fraction protocol: integrate with a scheduled box rescale
======================================================================

:func:`~jaxdem.utils.dynamicsRoutines.run_packing_fraction_protocol`
is a thin wrapper around :meth:`System.step` that interleaves
:func:`~jaxdem.utils.packingUtils.scale_to_packing_fraction` on a
user-supplied, per-frame schedule. Temperature control, bonded forces,
collider, etc. are whatever the ``System`` you pass in already has —
the protocol just delegates.

Here we:

1. Build a sphere packing at a modest ``phi = 0.35`` with
   :func:`build_sphere_system`.
2. Drive ``phi`` up to ``0.55`` along a linear ramp and then hold for a
   while, saving a frame at each step on a pseudolog schedule (so
   early, fast changes are well-resolved and late, slow drift is
   sparsely sampled).
3. Read back per-frame ``phi`` and total kinetic energy along the ramp.
"""

# %%
# Imports
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp
import numpy as np

from jaxdem.utils.particleCreation import build_sphere_system
from jaxdem.utils.dynamicsRoutines import run_packing_fraction_protocol
from jaxdem.utils.packingUtils import compute_packing_fraction
from jaxdem.utils.rollout_schedules import make_save_steps_pseudolog
from jaxdem.utils.thermal import compute_translational_kinetic_energy


# %%
# 1) Build the starting system
# ----------------------------
N = 128
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
)
print(f"initial phi = {float(compute_packing_fraction(state, system)):.4f}")


# %%
# 2) Build the schedule
# ---------------------
# ``make_save_steps_pseudolog`` gives us non-uniform save points that
# densely sample the early transient and thin out later. ``np.diff``
# turns those absolute step indices into the per-frame strides the
# protocol wants.

num_steps = 20_000
save_steps = make_save_steps_pseudolog(
    num_steps=num_steps,
    reset_save_decade=2_000,
    min_save_decade=50,
    decade=10,
    include_step0=False,  # first recorded state is after the first stride, not the initial state
)
strides = np.diff(np.concatenate([[0], save_steps]))  # stride from step 0 to save_steps[0], then between frames
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
# Pure Verlet integration between rescale events — no thermostat, so
# compression pumps energy into the system and KE grows with phi.
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

print("idx  step      phi       KE")
for i in (0, 1, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1):
    print(f"{i:3d}  {save_steps[i]:6d}    {phi_trace[i]:.4f}    {ke_trace[i]:.3e}")


# %%
# Swap the integrator to ``verlet_rescaling`` at build time and the
# same call above will keep the kinetic temperature clamped while
# phi still ramps — the protocol itself is agnostic to the
# integrator choice.
