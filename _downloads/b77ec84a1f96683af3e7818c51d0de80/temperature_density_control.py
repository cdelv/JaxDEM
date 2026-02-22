"""
Example: temperature and density control via simple rescaling.

This script builds a small periodic system of spheres, initialized at a low temperature,
and then runs dynamics while imposing a *time-dependent packing fraction*
using `jd.utils.control_nvt_density_rollout`.  This function can modulate both the
temperature and density of the system using very simple rescaling.
It comes in two variants, one with _rollout and one without.  The _rollout
variant provides access to the intermediate trajectory data, whereas the other
does not.  Although it may be convenient to get the trajectory data,
the non _rollout option is higher performance.

In either, we can set a target temperature and/or density and the control_nvt_density
function will use a linear scheduler to reach the desired set point after a set
number of steps.

You can also control either variable using a custom scheduler.  Here, we show
an example of that.  We modulate the density according to a single sine wave.

In this example we:
- **Modulate density** (packing fraction) with a prescribed schedule.
- **Do not thermostat** (temperature is allowed to fluctuate in response to compression).
"""

# %%
# Imports
# ~~~~~~~~~~~~~~~~~~~~~
import jax
import jax.numpy as jnp
import jaxdem as jd
from functools import partial
from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

# We need to enable double precision for accuracy in cold systems
jax.config.update("jax_enable_x64", True)

# %%
# Parameters
# ~~~~~~~~~~~~~~~~~~~~~
# Background:
# We build a single system in the same spirit as `examples/jam_spheres.py`:
# - choose a polydisperse radius distribution
# - generate a random, overlap-free configuration at the desired packing fraction `phi`
# - create a periodic `System` with a simple spring contact model
N = 100
phi = 0.7
dim = 2
e_int = 1.0
dt = 1e-2

can_rotate = False  # smooth spheres cannot rotate
subtract_drift = True
seed = 0

def build_microstate(i):
    mass = 1.0
    e_int = 1.0
    dt = 1e-2
    if dim == 2:
        cr = [0.5, 0.5]
        sr = [1.0, 1.4]
    else:
        cr = [1.0]
        sr = [1.0]
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N, cr, sr)
    pos, box_size = random_sphere_configuration(particle_radii, phi, dim, seed)
    state = jd.State.create(
        pos=pos,
        rad=particle_radii,
        mass=jnp.ones(N) * mass
    )
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system


# %%
# Run dynamics while imposing a sine-wave compression
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We impose a single-period sinusoidal modulation of the packing fraction over the
# protocol duration (a compression/decompression cycle). We intentionally do **not**
# set a temperature schedule here, so temperature is free to fluctuate.
initial_temperature = 1e-4
packing_fraction_amplitude = -0.05
n_steps = 10_000
save_stride = 100
n_snapshots = int(n_steps) // int(save_stride)

state, system = build_microstate(0)
state = jd.utils.thermal.set_temperature(state, initial_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift, seed=0)

def sine_dens_schedule(k, K, start, target):
    x = k / jnp.maximum(K, 1)  # 0..1
    return start + packing_fraction_amplitude * jnp.sin(2.0 * jnp.pi * x)  # one full period

state, system, (traj_state, traj_system) = jd.utils.control_nvt_density_rollout(
    state,
    system,
    n=n_snapshots,
    stride=save_stride,
    rescale_every=100,
    # NOTE: Density control is enabled when either packing_fraction_target or
    # packing_fraction_delta is provided. Our schedule ignores the "target", so we
    # pass delta=0.0 simply to enable control.
    packing_fraction_delta=0.0,
    density_schedule=sine_dens_schedule,
    can_rotate=can_rotate,
    subtract_drift=subtract_drift,
)

temperature = jax.vmap(partial(jd.utils.thermal.compute_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift))(traj_state)
phi = jax.vmap(partial(jd.utils.packingUtils.compute_packing_fraction))(traj_state, traj_system)
pe = jax.vmap(partial(jd.utils.thermal.compute_potential_energy))(traj_state, traj_system)

print('The recorded temperature profile is: ', temperature)
print('The recorded density profile is: ', phi)
print('The recorded potential-energy profile is: ', pe)

# %%
