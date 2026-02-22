"""
Jamming of bidisperse spheres (or disks)
----------------------------------------

In this example, we'll use the bisection search algorithm to find the nearest jammed state for a set of random
configurations of bidisperse spheres (or disks) in a 3D (or 2D) periodic box.

The particles use a purely repulsive harmonic interaction potential, meaning that the potential energy is zero
when the particles are not in contact, and is otherwise proportional to the square of the overlap distance.

The bisection search algorithm is a simple and efficient way to find the nearest jammed state for a given configuration.
It works by incrementally compressing the box (equivalently growing the radii of the particles) and minimizing the total
potential energy after each incremental compression.  If, after a compression, the potential energy is nonzero, the
state is said to be possibly jammed.  In this case, the system is reverted to the last unjammed state and a bisection
search is performed between the jammed and unjammed states.  The algorithm will then find the largest packing fraction
within the bounds which has a total potential energy near zero.  Compressing beyond this packing fraction will result
in a drastic increase in potential energy.

The bisection search algorithm is implemented in :py:func:`~jaxdem.utils.jamming.bisection_jam`.
"""

# %%
# Imports
# ~~~~~~~~~~~~~~~~~~~~~
import jax
import jax.numpy as jnp
import jaxdem as jd

# We need to enable double precision to reach the necessary accuracy for conventional jamming analysis.
jax.config.update("jax_enable_x64", True)

# %%
# Parameters
# ~~~~~~~~~~~~~~~~~~~~~
# We'll jam 10 systems of 10 particles in parallel.
# This highlights the utility of system-level parallelism in JaxDEM
# although it should be noted that the parallelized algorithm is only
# as fast as the slowest system.
# As we approach jamming, the systems will take longer to minimize,
# so the jamming algorithm can be quite slow if parallelized over
# many systems.
# We will place the particles down randomly in the box according
# to an initial packing fraction of 0.4.
N_systems = 10
N = 10
phi = 0.4
dim = 2
e_int = 1.0
dt = 1e-2

def build_microstate(i):
    # assign bidisperse radii
    rad = jnp.ones(N)
    rad = rad.at[: N // 2].set(0.5)
    rad = rad.at[N // 2:].set(0.7)
    
    # set the box size for the packing fraction and the radii
    volume = (jnp.pi ** (dim / 2) / jax.scipy.special.gamma(dim / 2 + 1)) * rad ** dim
    L = (jnp.sum(volume) / phi) ** (1 / dim)
    box_size = jnp.ones(dim) * L

    # create microstate
    key = jax.random.PRNGKey(i)
    pos = jax.random.uniform(key, (N, dim), minval=0.0, maxval=L)
    mass = jnp.ones(N)
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    
    # create system and state
    state = jd.State.create(pos=pos, rad=rad, mass=mass, volume=volume)
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="linearfire",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        # collider_type="celllist",
        # collider_kw=dict(state=state),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system

# %%
# Run the Jamming Algorithm for Multiple Systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We'll first create the systems and states using jax's vmap function.
# This will create 10 states and systems in parallel.
# We could also use the State.stack method to join a list of states and systems.
state, system = jax.vmap(build_microstate)(jnp.arange(N_systems))

# We'll then run the jamming algorithm on the systems using jax's vmap function.
# This will run the jamming algorithm on each system in parallel.
# It returns the final state and system as well as their final packing fraction and potential energy.
# The final potential energy per degree of freedom should be less than the tolerance of 1e-16.
state, system, final_pf, final_pe = jax.vmap(lambda st, sys: jd.utils.jamming.bisection_jam(st, sys))(state, system)

print(f"Final potential energy: {final_pe}")
print(f"Final packing fraction: {final_pf}")

# %%
# Run the Jamming Algorithm for a Single System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can also run the jamming algorithm on a single system by passing the state and system to the jamming function.
# This is likely slightly more convenient.
state, system = build_microstate(0)
state, system, final_pf, final_pe = jd.utils.jamming.bisection_jam(state, system)

print(f"Final potential energy: {final_pe}")
print(f"Final packing fraction: {final_pf}")
