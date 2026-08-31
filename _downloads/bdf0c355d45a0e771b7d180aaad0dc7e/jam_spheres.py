"""Jamming of bidisperse spheres (or disks).
--------------------------------------------

In this example, we use a bisection search to find the nearest jammed state for a set of random
configurations of bidisperse spheres (or disks). The particles sit in a 3D (or 2D) periodic box.

The particles use a purely repulsive harmonic interaction potential. The potential energy is zero
when the particles are not in contact. Otherwise it is proportional to the square of the overlap
distance.

The bisection search works as follows. It compresses the box in small steps (equivalently, it grows
the radii of the particles) and minimizes the total potential energy after each step. If the potential
energy is nonzero after a compression, the state is possibly jammed. The algorithm then goes back to
the last unjammed state and runs a bisection search between the jammed and unjammed states. It finds
the largest packing fraction in these bounds whose total potential energy is near zero. Compressing
beyond this packing fraction increases the potential energy sharply.

:py:func:`~jaxdem.utils.jamming.bisection_jam` implements the bisection search.
"""

# %%
# Imports
# ~~~~~~~~~~~~~~~~~~~~~
import jax
import jax.numpy as jnp
import jaxdem as jdem

# We enable double precision to reach the accuracy that conventional jamming analysis needs.
jax.config.update("jax_enable_x64", True)

# %%
# Parameters
# ~~~~~~~~~~~~~~~~~~~~~
# We jam 10 systems of 10 particles in parallel.
# This shows system-level parallelism in JaxDEM.
# Note that the parallel run is only as fast as the slowest system.
# Close to jamming, the systems take longer to minimize,
# so the jamming algorithm can be slow when you parallelize over
# many systems.
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
# Run the Jamming Algorithm for Multiple Systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We first create the systems and states with JAX's vmap function.
# This creates 10 states and systems in parallel.
# We could also use the State.stack method to join a list of states and systems.
state, system = jax.vmap(build_microstate)(jnp.arange(N_systems))

# We then run the jamming algorithm on the systems with JAX's vmap function.
# This runs the jamming algorithm on each system in parallel.
# It returns a :class:`~jaxdem.utils.jamming.JamResult` named tuple with the last
# unjammed state/system, the jammed state/system, and the jammed state's
# packing fraction and potential energy.
# The final potential energy per particle should be less than the tolerance of 1e-16.
result = jax.vmap(lambda st, sys: jdem.utils.jamming.bisection_jam(st, sys))(
    state, system
)
state, system = result.jammed_state, result.jammed_system

print(f"Final potential energy: {result.potential_energy}")
print(f"Final packing fraction: {result.packing_fraction}")

# %%
# Run the Jamming Algorithm for a Single System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can also run the jamming algorithm on a single system by passing the state and system to the jamming function.
# This is slightly more convenient.
state, system = build_microstate(0)
result = jdem.utils.jamming.bisection_jam(state, system)
state, system = result.jammed_state, result.jammed_system

print(f"Final potential energy: {result.potential_energy}")
print(f"Final packing fraction: {result.packing_fraction}")
