r"""Minimization of bidisperse spheres (or disks).
--------------------------------------------------

In this example, we minimize the energy of a set of random configurations of bidisperse spheres
(or disks). The particles sit in a 3D (or 2D) periodic box.

The particles use a purely repulsive harmonic interaction potential. For a
pair with separation :math:`r_{ij}` and radii :math:`R_i,R_j`,

.. math::

    \delta_{ij}=\max(0,R_i+R_j-r_{ij}),\qquad
    U=\frac12\sum_{i<j}k_{ij}\delta_{ij}^2.

The energy is zero without overlaps. During an overlap, the normal force
magnitude is :math:`k_{ij}\delta_{ij}`; it pushes the centers apart. FIRE
uses these forces to find a stationary configuration at the fixed box size.
Minimization is a preparation procedure, not a trajectory at a prescribed
physical temperature or a measurement of a material relaxation time.

FIRE relaxes overlaps until the maximum free-body force and torque meet
the requested tolerances. A dilute minimum can have zero contact energy
and still have many floppy motions; minimization does not prove rigidity.

Two small, reproducible systems keep this demonstration quick on a CPU.
Lengths, masses, and stiffnesses use a consistent set of reduced units.

"""

# %%
# Imports
# ~~~~~~~~~~~~~~~~~~~~~
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jaxdem as jdem

# We enable double precision to reach the accuracy that our tolerances need.
jax.config.update("jax_enable_x64", True)

# %%
# Parameters
# ~~~~~~~~~~~~~~~~~~~~~
# We minimize two systems of eight particles in parallel.
# This shows system-level parallelism in JaxDEM.
# Note that the parallel run is only as fast as the slowest system.
# We place the particles randomly in the box at an initial packing fraction
# of 0.4. Each realization uses a fixed seed, so changes to tolerances or the
# minimizer can be compared against identical initial positions. At this low
# fraction, removing overlaps typically yields zero energy with many remaining
# free motions. A higher fraction can retain positive energy at force balance.
N_systems = 2
N = 8
phi = 0.4
dim = 2
e_int = 1.0
dt = 5e-2


# %%
# For the one-call equivalent see
# :py:func:`~jaxdem.utils.particle_creation.build_sphere_system`
# (:doc:`sphere_construction example </auto_examples/sphere_construction>`).
def build_microstate(i):
    # assign bidisperse radii
    rad = jnp.ones(N)
    rad = rad.at[: N // 2].set(0.5)
    rad = rad.at[N // 2 :].set(0.7)

    # Set the box from the nominal fraction phi = sum(V_i) / L**dim.
    # V_i = pi**(dim/2) R_i**dim / Gamma(dim/2+1), giving area in 2D
    # and volume in 3D. The equal 0.5/0.7 radius populations introduce
    # bidispersity without changing the mass convention (unit mass per disk).
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
# We first create the systems and states with JAX's vmap function. The seed
# index selects each independent configuration, while all members share the
# same particle count and dimensionality. The resulting arrays carry a leading
# system axis, e.g. ``state.pos.shape == (N_systems, N, dim)``. The second
# vmap below applies minimization independently to those State/System pairs.
# ``State.stack`` is another way to assemble compatible states prepared
# separately; the system layouts must be stacked consistently as well.
state, system = jax.vmap(build_microstate)(jnp.arange(N_systems))

# A finite budget makes failure explicit instead of permitting a long run.
n_steps = 2_000
force_tol = 1e-7

# %%
# FIRE adapts a fictitious velocity and step size using force information.
# Its internal integration is an optimization device; ``dt`` controls this
# search, not elapsed physical relaxation time. Accepted mechanical convergence
# requires finite free-body residuals satisfying
#
# .. math::
#
#    \max_{I\in\mathrm{free}}\|F_I\|\leq f_{\rm tol},\qquad
#    \max_{I\in\mathrm{free}}\|\tau_I\|\leq\tau_{\rm tol}.
#
# For these smooth spheres the contact law produces no torque; the torque
# criterion matters when the same workflow is applied to rigid clumps.
# Reaching a step limit or producing nonfinite residuals is a failure to
# converge. Low energy alone is not a replacement for checking these forces.
#
# The result contains the final state and system, step count, and energy,
# plus status and residual diagnostics. Four-value unpacking is convenient,
# but inspect ``result.converged`` first. With the physical target used here,
# FIRE evaluates the final potential energy after relaxation and reports
# **energy per constituent sphere**, ``U/N``. A custom ``target_fn`` instead
# defines its own objective and units, so that interpretation must be revisited.
from jaxdem.utils.thermal import compute_potential_energy

initial_pe = jax.vmap(compute_potential_energy)(state, system) / N
result = jax.vmap(
    lambda st, sys: sys.minimize(
        st, sys, max_steps=n_steps, force_tol=force_tol, torque_tol=force_tol
    )
)(state, system)
if not bool(jnp.all(result.converged)):
    raise RuntimeError(f"Minimization failed: reason={result.reason}")
state, system, steps, final_pe = result

print(f"Initial energy per particle: {initial_pe}")
print(f"Maximum residual force: {result.info.force_max}")
print(f"Converged: {result.converged}")
print(f"Final potential energy: {final_pe}")
print(f"Number of steps taken: {steps}")

# %%
# Separate energy reduction from convergence cost
# -----------------------------------------------
# At this dilute fraction both preparations remove their initial overlaps.
# The left panel retains zero final energies on a symmetric logarithmic
# scale. The right panel compares optimization work; force-residual labels
# are normalized by the requested tolerance, so values at or below one
# satisfy that criterion. These iterations are FIRE steps, not physical time.
seeds = np.arange(N_systems)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
for seed, initial, final in zip(
    seeds, np.asarray(initial_pe), np.asarray(final_pe), strict=True
):
    axes[0].plot([0, 1], [initial, final], "o-", label=f"Seed {seed}")
axes[0].set(
    xticks=[0, 1],
    xticklabels=["Initial", "Relaxed"],
    ylabel="Potential energy per sphere",
    title="FIRE removes the overlaps",
)
axes[0].set_yscale("symlog", linthresh=1e-10)
axes[0].set_ylim(-5e-11, 1.7 * max(float(jnp.max(initial_pe)), 1e-10))
axes[0].legend(fontsize="small")
bars = axes[1].bar(seeds, np.asarray(steps), width=0.5, color="tab:blue")
residual_ratio = np.asarray(result.info.force_max) / force_tol
axes[1].bar_label(
    bars,
    labels=[f"F/tol = {value:.2g}" for value in residual_ratio],
    padding=3,
    fontsize="small",
)
axes[1].set(
    xticks=seeds,
    xlabel="Preparation seed",
    ylabel="FIRE iterations",
    title="Cost and final force residual",
)
axes[1].margins(y=0.2)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
plt.show()

# %%
# Run the Minimization for a Single System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can also run the minimization on a single system by passing the state and system to the minimization function.
state, system = build_microstate(0)
result = system.minimize(
    state, system, max_steps=n_steps, force_tol=force_tol, torque_tol=force_tol
)
if not bool(result.converged):
    raise RuntimeError(f"Single-system minimization failed: reason={result.reason}")
state, system, steps, final_pe = result

print(f"Converged: {result.converged}")
print(f"Final potential energy: {final_pe}")
print(f"Number of steps taken: {steps}")
# %%
