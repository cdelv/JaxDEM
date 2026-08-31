"""Packing-fraction protocol: integrate with a scheduled box rescale
======================================================================

:func:`~jaxdem.utils.dynamics_routines.run_packing_fraction_protocol`
wraps :meth:`System.step` and calls
:func:`~jaxdem.utils.packing_utils.scale_to_packing_fraction` on a
user-supplied, per-frame schedule. Temperature control, bonded forces,
and the collider come from the ``System`` you pass in. The protocol
does not change them.

Here we:

1. Build a sphere packing at ``phi = 0.35`` with
   :func:`build_sphere_system`.
2. Ramp ``phi`` linearly up to ``0.55`` and then hold it there. We save
   frames on a pseudolog schedule, which samples the fast early changes
   densely and the slow late drift sparsely.
3. Read back per-frame ``phi`` and total kinetic energy along the ramp.
"""

# %%
# Imports
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp
import numpy as np

from jaxdem.utils.particle_creation import build_sphere_system
from jaxdem.utils.dynamics_routines import run_packing_fraction_protocol
from jaxdem.utils.packing_utils import compute_packing_fraction
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
# The protocol runs pure Verlet integration between rescale events. There is no thermostat, so
# compression adds energy to the system and KE grows with phi.
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
