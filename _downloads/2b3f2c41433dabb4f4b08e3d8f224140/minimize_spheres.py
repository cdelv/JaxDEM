"""Minimization of bidisperse spheres (or disks).
-------------------------------------------------

In this example, we minimize the energy of a set of random configurations of bidisperse spheres
(or disks). The particles sit in a 3D (or 2D) periodic box.

The particles use a purely repulsive harmonic interaction potential. The potential energy is zero
when the particles are not in contact. Otherwise it is proportional to the square of the overlap
distance.

The FIRE minimizer minimizes the energy.

"""

# %%
# Imports
# ~~~~~~~~~~~~~~~~~~~~~
import jax
import jax.numpy as jnp
import jaxdem as jdem

# We enable double precision to reach the accuracy that our tolerances need.
jax.config.update("jax_enable_x64", True)

# %%
# Parameters
# ~~~~~~~~~~~~~~~~~~~~~
# We minimize 10 systems of 10 particles in parallel.
# This shows system-level parallelism in JaxDEM.
# Note that the parallel run is only as fast as the slowest system.
# We place the particles randomly in the box at
# an initial packing fraction of 0.4.
N_systems = 10
N = 10
phi = 0.4
dim = 2
e_int = 1.0
dt = 1e-2


# For the one-call equivalent see
# :func:`~jaxdem.utils.particle_creation.build_sphere_system`
# (sphere_construction example).
def build_microstate(i):
    # assign bidisperse radii
    rad = jnp.ones(N)
    rad = rad.at[: N // 2].set(0.5)
    rad = rad.at[N // 2 :].set(0.7)

    # set the box size for the packing fraction and the radii
    volume = (jnp.pi ** (dim / 2) / jax.scipy.special.gamma(dim / 2 + 1)) * rad**dim
    L = (jnp.sum(volume) / phi) ** (1 / dim)
    box_size = jnp.ones(dim) * L

    # create microstate
    key = jax.random.PRNGKey(i)
    pos = jax.random.uniform(key, (N, dim), minval=0.0, maxval=L)
    mass = jnp.ones(N)
    mats = [jdem.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jdem.MaterialMatchmaker.create("harmonic")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)

    # create system and state
    state = jdem.State.create(pos=pos, rad=rad, mass=mass, volume=volume)
    system = jdem.System.create(
        state_shape=state.shape,
        dt=dt,
        minimizer=jdem.minimizers.fire,
        minimizer_kw={"dt": dt},
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={
            "box_size": box_size,
        },
    )
    return state, system


# %%
# Run the Minimization for Multiple Systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We first create the systems and states with JAX's vmap function.
# This creates 10 states and systems in parallel.
# We could also use the State.stack method to join a list of states and systems.
state, system = jax.vmap(build_microstate)(jnp.arange(N_systems))

# We run the minimization for up to 1M steps
n_steps = 1_000_000

# The minimizer stops as soon as ANY of the following conditions is met:
# 1. step_count >= max_steps (step budget exhausted)
# 2. |PE| / N <= pe_tol (per-particle energy is low enough)
# 3. the relative change in PE between steps drops below pe_diff_tol (energy stopped changing)
# 4. the maximum absolute gradient component drops to force_tol or below
# We set the tolerance for the potential energy and for its relative change to 1e-16.
# The minimizer returns the final state, system, number of steps taken, and the final
# potential energy. It reports the final potential energy PER PARTICLE (PE / N) when you
# do not set a custom target_fn.
state, system, steps, final_pe = jax.vmap(
    lambda st, sys: sys.minimize(
        st, sys, max_steps=n_steps, pe_tol=1e-16, pe_diff_tol=1e-16
    )
)(state, system)

print(f"Final potential energy: {final_pe}")
print(f"Number of steps taken: {steps}")

# %%
# Run the Minimization for a Single System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can also run the minimization on a single system by passing the state and system to the minimization function.
state, system = build_microstate(0)
state, system, steps, final_pe = system.minimize(
    state, system, max_steps=n_steps, pe_tol=1e-16, pe_diff_tol=1e-16
)

print(f"Final potential energy: {final_pe}")
print(f"Number of steps taken: {steps}")
# %%
