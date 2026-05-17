"""Temperature and density control via the modern stack.

This script runs the same physical scenario twice to demonstrate the
two orthogonal control axes available in JaxDEM today:

* **Temperature control** is a choice of integrator. Picking
  ``linear_integrator_type="verlet_rescaling"`` at ``System.create`` time
  clamps the kinetic temperature to a fixed value on a configurable
  cadence; picking ``"verlet"`` lets temperature float.

* **Density / box control** is a protocol on top of whatever integrator
  you chose. :func:`run_packing_fraction_protocol` integrates via
  ``system.step`` in chunks and calls :func:`scale_to_packing_fraction`
  on a user-supplied per-frame schedule.

We impose a single-period sinusoidal modulation of the packing fraction
and compare:

1. bare Verlet + phi modulation — temperature rises on compression, falls
   on expansion, as expected.
2. velocity-rescaling Verlet + phi modulation — temperature stays clamped
   while phi oscillates.
"""

# %%
# Imports
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jaxdem as jd
from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration
from jaxdem.utils.dynamicsRoutines import run_packing_fraction_protocol
from jaxdem.utils.packingUtils import compute_packing_fraction
from jaxdem.utils.thermal import compute_potential_energy, compute_temperature, set_temperature


# %%
# Parameters
N = 100
phi0 = 0.70
dim = 2
dt = 1e-2
initial_temperature = 1e-4
phi_amplitude = -0.05          # phi(t) = phi0 + amp * sin(2*pi*t), one full period
n_steps = 10_000
save_stride = 100              # one saved frame per 100 integration steps
n_frames = n_steps // save_stride

can_rotate = False             # smooth spheres, no rotation DOF
subtract_drift = True
seed = 0


# %%
# Precompute the phi schedule
# ---------------------------
# One phi value per saved frame. Each frame covers ``save_stride`` steps,
# after which the box is rescaled to the frame's target phi.
t_frac = (1 + np.arange(n_frames)) / n_frames    # (0, 1]
phi_at_frames = phi0 + phi_amplitude * np.sin(2.0 * np.pi * t_frac)
strides = np.full(n_frames, save_stride, dtype=int)


# %%
# System builder — two variants differ only in the linear integrator choice.
def build_state_system(linear_integrator_type, linear_integrator_kw=None):
    if dim == 2:
        particle_radii = jd.utils.dispersity.get_polydisperse_radii(
            N, [0.5, 0.5], [1.0, 1.4]
        )
    else:
        particle_radii = jd.utils.dispersity.get_polydisperse_radii(N, [1.0], [1.0])
    pos, box_size = random_sphere_configuration(
        particle_radii.tolist(), phi0, dim, seed
    )
    state = jd.State.create(pos=pos, rad=particle_radii, mass=jnp.ones(N))
    state = set_temperature(
        state, initial_temperature,
        can_rotate=can_rotate, subtract_drift=subtract_drift, seed=seed,
    )
    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    mat_table = jd.MaterialTable.from_materials(
        mats, matcher=jd.MaterialMatchmaker.create("harmonic")
    )
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type=linear_integrator_type,
        linear_integrator_kw=linear_integrator_kw or {},
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={"box_size": box_size},
    )
    return state, system


def summarize(label, traj_state, traj_system):
    temperature = jax.vmap(
        lambda s: compute_temperature(s, can_rotate=can_rotate, subtract_drift=subtract_drift)
    )(traj_state)
    phi_series = jax.vmap(compute_packing_fraction)(traj_state, traj_system)
    pe = jax.vmap(compute_potential_energy)(traj_state, traj_system)
    T_arr = np.asarray(temperature)
    phi_arr = np.asarray(phi_series)
    pe_arr = np.asarray(pe)
    print(
        f"[{label}]  T: min={T_arr.min():.3e} max={T_arr.max():.3e} mean={T_arr.mean():.3e}  "
        f"phi: min={phi_arr.min():.4f} max={phi_arr.max():.4f}  "
        f"PE mean={pe_arr.mean():.3e}"
    )


# %%
# 1) Bare Verlet + phi modulation: temperature is free to fluctuate.
state, system = build_state_system("verlet")
state, system, (traj_state, traj_system) = run_packing_fraction_protocol(
    state, system, strides=strides, phi_at_frames=phi_at_frames,
)
summarize("bare verlet", traj_state, traj_system)


# %%
# 2) verlet_rescaling thermostat + phi modulation: temperature clamped.
state, system = build_state_system(
    "verlet_rescaling",
    linear_integrator_kw=dict(
        temperature=jnp.asarray(initial_temperature, dtype=float),
        rescale_every=jnp.asarray(50, dtype=int),
        can_rotate=jnp.asarray(int(can_rotate), dtype=int),
        subtract_drift=jnp.asarray(int(subtract_drift), dtype=int),
        k_B=jnp.asarray(1.0, dtype=float),
    ),
)
state, system, (traj_state, traj_system) = run_packing_fraction_protocol(
    state, system, strides=strides, phi_at_frames=phi_at_frames,
)
summarize("verlet_rescaling", traj_state, traj_system)
