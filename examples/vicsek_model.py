# %%
"""
This is an implementation of the Vicsek model in 2D.
A state is created with 200 particles in periodic boundaries,
each interacting via a pairwise purely-repulsive harmonic potential.
Particles move with a constant velocity ```v0``` in a direction that
is set by a random component proportional to ```eta``` and another
component proportional to the average velocity of all neighboring
particles within a distance ```neighbor_radius```.  In this
implementation, the random vector noise characterizes this
as an "extrinsic" noise Vicsek model, as opposed to the "intrinsic"
noise variant.  We use a ```trajectory_rollout``` to run the dynamics
using a jit-compiled loop, which also returns the trajectory data.
We then calculate the polarization order parameter (norm of the
average velocity vectors) for each saved frame.
"""

import jax
jax.config.update("jax_enable_x64", True)  # set f64 precision!
import jax.numpy as jnp
import jaxdem as jd
import numpy as np

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

# %%
# This function sets the initial data

def build_microstate(N, phi, dim, dt, neighbor_radius, eta, v0, seed):
    # important to set this to be large enough such that the collider does not overflow
    max_neighbors = 64

    # Mono-disperse radii
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N, [1.0], [1.0])

    pos, box_size = random_sphere_configuration(particle_radii, phi, dim, seed)

    state = jd.State.create(
        pos=pos,
        rad=particle_radii,
        mass=jnp.ones(N),
    )

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        # Vicsek integrator:
        linear_integrator_type="vicsek_extrinsic",
        linear_integrator_kw=dict(
            neighbor_radius=jnp.asarray(neighbor_radius, dtype=float),
            eta=jnp.asarray(eta, dtype=float),
            v0=jnp.asarray(v0, dtype=float),
            max_neighbors=max_neighbors,
        ),
        rotation_integrator_type="",
        domain_type="periodic",
        domain_kw=dict(box_size=box_size),
        force_model_type="spring",
        mat_table=mat_table,
        # here, we use the naive (double for-loop) collider since the system is small;
        # if you were to use a larger system, we recommend using the StaticCellList
        # or potentially the NeighborList
        collider_type="naive",
        seed=seed,
    )
    return state, system


# %%
# Build the initial data
state, system = build_microstate(
    N=200,  # 200 particles
    phi=0.65,  # just dense enough to order
    dim=2,  # 2D
    dt=1e-2,  # semi-arbitrary timestep
    neighbor_radius=1.0,  # particles will align within 2x their radii
    eta=0.2,  # small noise component
    v0=1.0,  # semi-arbitrary velocity
    seed=np.random.randint(0, 1e9)  # random seed
)

# Run the dynamics for 5K steps, saving every 50th
n_steps = 5_000
save_stride = 50
n_frames = n_steps // save_stride

state_f, system_f, (traj_state, traj_system) = jd.System.trajectory_rollout(
    state,
    system,
    n=n_frames,
    stride=save_stride,
)

polarization = jnp.linalg.norm(jnp.mean(traj_state.vel, axis=-2), axis=-1)
print("Polarization:")
print(polarization)
# %%
