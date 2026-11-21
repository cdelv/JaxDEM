# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Temperature and density control
==================================

JaxDEM separates temperature control and density control into two independent
axes:

* **Temperature control** is an integrator choice:
  ``linear_integrator_type="verlet_rescaling"`` rescales velocities to hold
  kinetic temperature :math:`T = 2K / (f k_B)` fixed on a chosen cadence,
  while ``"verlet"`` lets temperature fluctuate freely.
* **Density control** is an outer protocol:
  :func:`~jaxdem.utils.dynamics_routines.run_packing_fraction_protocol`
  integrates in chunks and rescales the periodic box along a per-frame
  :math:`\phi(t)` schedule.

Below we apply the same sinusoidal packing-fraction cycle to both integrators
and compare their kinetic-temperature histories.
"""

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem
from jaxdem.utils.dynamics_routines import run_packing_fraction_protocol
from jaxdem.utils.packing_utils import compute_packing_fraction
from jaxdem.utils.thermal import compute_temperature, set_temperature

# %%
# Setup and Sinusoidal Density Schedule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Align ``save_stride`` with ``rescale_every`` so samples do not alias against
# the velocity-rescaling cadence (temperature can still drift slightly between
# rescaling events).

N = 16
phi0 = 0.70
dt = 2e-3
young = 1_000.0
target_T = 1.0
n_steps = 1_000
save_stride = 50
n_frames = n_steps // save_stride
can_rotate = False
subtract_drift = True

t_frac = (1 + np.arange(n_frames)) / n_frames
phi_at_frames = phi0 - 0.05 * np.sin(2.0 * np.pi * t_frac)
strides = np.full(n_frames, save_stride, dtype=int)


def build_state_system(
    linear_integrator_type: str, linear_integrator_kw: dict | None = None
) -> tuple[jdem.State, jdem.System]:
    rad = jnp.ones(N)
    side = jnp.sqrt(jnp.sum(jnp.pi * rad**2) / phi0)
    axis = (jnp.arange(4) + 0.5) * side / 4.0
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    pos = jnp.stack((x.ravel(), y.ravel()), axis=-1)

    state = jdem.State.create(pos=pos, rad=rad, mass=jnp.ones(N))
    state = set_temperature(
        state,
        target_T,
        can_rotate=can_rotate,
        subtract_drift=subtract_drift,
        seed=0,
    )
    mat_table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=young, poisson=0.5, density=1.0)],
        matcher=jdem.MaterialMatchmaker.create("harmonic"),
    )
    system = jdem.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type=linear_integrator_type,
        linear_integrator_kw=linear_integrator_kw or {},
        rotation_integrator_type=None,
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={"box_size": jnp.full(2, side)},
    )
    return jdem.System.initialize(state, system)


def measure_traces(traj_state: jdem.State, traj_system: jdem.System):
    temp = np.asarray(
        jax.vmap(
            lambda s: compute_temperature(
                s, can_rotate=can_rotate, subtract_drift=subtract_drift
            )
        )(traj_state)
    )
    phi = np.asarray(jax.vmap(compute_packing_fraction)(traj_state, traj_system))
    return temp, phi


# %%
# Run 1: Bare Verlet vs Run 2: Velocity-Rescaling Thermostat
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st_bare, sys_bare = build_state_system("verlet")
_, _, (traj_bare_s, traj_bare_sys) = run_packing_fraction_protocol(
    st_bare, sys_bare, strides=strides, phi_at_frames=phi_at_frames
)
bare_T, bare_phi = measure_traces(traj_bare_s, traj_bare_sys)

st_ctrl, sys_ctrl = build_state_system(
    "verlet_rescaling",
    linear_integrator_kw=dict(
        temperature=target_T,
        rescale_every=50,
        can_rotate=can_rotate,
        subtract_drift=subtract_drift,
        k_B=1.0,
    ),
)
_, _, (traj_ctrl_s, traj_ctrl_sys) = run_packing_fraction_protocol(
    st_ctrl, sys_ctrl, strides=strides, phi_at_frames=phi_at_frames
)
ctrl_T, ctrl_phi = measure_traces(traj_ctrl_s, traj_ctrl_sys)

print(f"Bare Verlet     : T in [{bare_T.min():.3f}, {bare_T.max():.3f}]")
print(f"Verlet rescaling: T in [{ctrl_T.min():.3f}, {ctrl_T.max():.3f}]")

scene_states = {"Bare Verlet": traj_bare_s, "Velocity rescaling": traj_ctrl_s}
scene_systems = {"Bare Verlet": traj_bare_sys, "Velocity rescaling": traj_ctrl_sys}

# %%
# Plot Imposed Density Cycle and Temperature Response
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

times = np.cumsum(strides) * dt
fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True, layout="constrained")
axes[0].plot(times, bare_phi, "o-", ms=3, color="tab:green", label=r"Measured $\phi(t)$")
axes[0].set(ylabel=r"Packing fraction $\phi$", title="Imposed sinusoidal density cycle")
axes[0].legend(fontsize="small")

axes[1].plot(times, bare_T, "o-", ms=3, label="Bare Verlet")
axes[1].plot(times, ctrl_T, "s--", ms=3, label="Velocity rescaling")
axes[1].set(
    xlabel="Physical time",
    ylabel="Kinetic temperature T",
    title="Temperature floats under Verlet, stays clamped under rescaling",
)
axes[1].legend(fontsize="small")
for ax in axes:
    ax.grid(alpha=0.2)
plt.show()
